"""Fixture canon: normalization, scanning, and golden-file access.

This module is the single source of truth for what the test suite is allowed to
assert about the corpus. It deliberately has **no third-party imports** so that
it runs in CI without NumPy, Triton or a GPU.

Two blocks in `fixtures/GOLDEN.json` with different provenance and different
edit rules:

  observations  machine-derived from the normalized fixture text.
                NEVER hand-edited. `make golden` rewrites this block.
  intent        hand-authored assertions about what each tier was *designed* to
                exhibit. `make golden` refuses to touch it; changes require
                review (see docs/team/testing-ci.md R2).

Anything a test needs to know about a fixture comes from one of those two
blocks. A literal typed into a test file is a defect.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "fixtures"
RAW = FIXTURES / "raw"
GOLDEN = FIXTURES / "GOLDEN.json"

SCHEMA_VERSION = 1

# A source dump embeds the absolute path of the file that produced it, e.g.
#   #loc = loc("/home/dev/proj/verify/dump.py":31:0)
# That makes the bytes machine-dependent, so two people generating the same
# kernel get different fixtures and every hash-based test fails for one of
# them. Normalization replaces the path with the literal token LOCFILE before
# anything is hashed. See docs/team/testing-ci.md R3.
_ABS_PATH_LOC = re.compile(r'loc\("(/[^"]*)":(\d+):(\d+)\)')

# An operation line: an optional SSA definition followed by a namespaced op
# name at the start of the line, e.g.
#     %c0 = arith.constant 0 : i32 loc(#loc1)
#     %acc:3 = scf.for ... -> (...) : i32 {
#     tt.func public @k(...) attributes {...} {
_OP = re.compile(r"^\s*(?:%[A-Za-z0-9_]+(?::\d+)?\s*=\s*)?([a-z][a-z0-9_]*\.[a-z0-9_]+)")
_DEF = re.compile(r"^\s*(%[A-Za-z0-9_]+(?::\d+)?)\s*=")
_MULTI_RESULT = re.compile(r"^\s*%[A-Za-z0-9_]+:(\d+)\s*=")
_LOC_DEF = re.compile(r"^#loc(\d*)\s*=\s*loc\(")
_LOC_USE = re.compile(r"loc\(#loc\d*\)")
_LOC_NAME = re.compile(r'^#loc\d*\s*=\s*loc\("([A-Za-z_][A-Za-z0-9_.]*)"\(')
_LOC_FILE = re.compile(r'^#loc\d*\s*=\s*loc\("([^"]*)":\d+:\d+\)')

# Facts that must be observable for the corpus to be the corpus. Keys here are
# copied verbatim into observations.ops.
OPS_OF_INTEREST = (
    "tt.dot",
    "tt.load",
    "tt.store",
    "tt.splat",
    "tt.broadcast",
    "tt.expand_dims",
    "tt.make_range",
    "tt.get_program_id",
    "tt.return",
    "arith.remsi",
    "scf.for",
    "scf.yield",
)


def normalize(text: str) -> str:
    """Return the canonical, machine-independent form of a source dump."""
    text = _ABS_PATH_LOC.sub(r'loc("LOCFILE":\2:\3)', text)
    lines = [line.rstrip() for line in text.splitlines()]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines) + "\n"


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def scan(text: str) -> dict:
    """Derive every observable fact about one normalized fixture."""
    lines = text.splitlines()
    dialects: dict[str, int] = {}
    ops: dict[str, int] = {}
    defs: list[str] = []
    multi_result: list[str] = []
    loc_defs = loc_uses = 0
    loc_names: set[str] = set()
    loc_files: set[str] = set()
    depth = max_depth = 0
    funcs: list[str] = []
    tf32 = False

    for lineno, line in enumerate(lines, start=1):
        if line.startswith("#loc"):
            loc_defs += 1
            m = _LOC_NAME.match(line)
            if m:
                loc_names.add(m.group(1))
            m = _LOC_FILE.match(line)
            if m:
                loc_files.add(m.group(1))
            continue
        loc_uses += len(_LOC_USE.findall(line))
        if "tf32" in line:
            tf32 = True
        m = _DEF.match(line)
        if m:
            defs.append(m.group(1))
            if _MULTI_RESULT.match(line):
                multi_result.append(f"line{lineno}:{m.group(1)}")
        m = _OP.match(line)
        if m:
            op = m.group(1)
            ops[op] = ops.get(op, 0) + 1
            dialect = op.split(".", 1)[0]
            dialects[dialect] = dialects.get(dialect, 0) + 1
            if op == "tt.func":
                fn = re.search(r"@([A-Za-z_][A-Za-z0-9_]*)", line)
                if fn:
                    funcs.append(fn.group(1))
        depth += line.count("{") - line.count("}")
        max_depth = max(max_depth, depth)

    return {
        "sha256": sha256(text),
        "bytes": len(text.encode("utf-8")),
        "lines": len(lines),
        "ops_total": sum(ops.values()),
        "op_vocab": len(ops),
        "op_hist": dict(sorted(ops.items())),
        "dialects": dict(sorted(dialects.items())),
        "ssa_defs": len(defs),
        "ssa_unique": len(set(defs)),
        "multi_result_ops": len(multi_result),
        "multi_result_examples": multi_result,
        "ops": {k: ops.get(k, 0) for k in OPS_OF_INTEREST},
        "loc_defs": loc_defs,
        "loc_uses": loc_uses,
        "loc_files": sorted(loc_files),
        "loc_names": sorted(loc_names),
        "max_region_depth": max_depth,
        "has_module_wrapper": any(line_str.strip() == "module {" for line_str in lines),
        "funcs": funcs,
        "has_tf32": tf32,
    }


def load_golden() -> dict:
    return json.loads(GOLDEN.read_text())


def fixture_names() -> list[str]:
    return sorted(p.stem for p in FIXTURES.glob("*.ttir"))


def read_fixture(name: str) -> str:
    return (FIXTURES / f"{name}.ttir").read_text()
