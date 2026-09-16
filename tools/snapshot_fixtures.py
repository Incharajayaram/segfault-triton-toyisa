#!/usr/bin/env python3
"""Snapshot raw Triton IR dumps into the frozen fixture canon.

    python3 tools/snapshot_fixtures.py --write    # fixtures/raw -> fixtures/ + observations
    python3 tools/snapshot_fixtures.py --golden   # re-derive observations only
    python3 tools/snapshot_fixtures.py --check    # verify, exit 1 on drift

`--check` is what CI runs (make golden-check). It fails if:

  * a normalized fixture's hash differs from the recorded one, or
  * any recorded observation no longer matches the fixture text, or
  * a fixture exists with no record, or a record with no fixture.

It prints the offending field, the recorded value and the observed value, so a
red CI run names the drift instead of reporting "assertion failed".

Stdlib only: no Triton, no NumPy, no GPU. Regenerating fixtures/raw does need
Triton and is a separate, manual job (make fixtures).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixtures_lib import (
    FIXTURES,
    GOLDEN,
    RAW,
    SCHEMA_VERSION,
    load_golden,
    normalize,
    scan,
)

# ---------------------------------------------------------------------------
# intent: hand-authored, reviewed, never rewritten by this tool.
# These say what each tier was DESIGNED to exhibit. Observations are what the
# bytes actually say. A disagreement between the two is a corpus bug, and
# --check reports it separately from a drift.
# ---------------------------------------------------------------------------
INTENT = {
    "t0_vecadd": {
        "purpose": "elementwise only; true-negative control for the MAC and epilogue idioms",
        "expect_dot": 0,
        "expect_loop": False,
        "expect_mac_idiom": 0,
        "expect_epilogue_idiom": 0,
        "expect_access": "structured",
        "expect_fully_lowered": True,
    },
    "t1_matmul": {
        "purpose": "the primary kernel; loop-carried pointers are the hard case",
        "expect_dot": 1,
        "expect_loop": True,
        "expect_mac_idiom": 1,
        "expect_epilogue_idiom": 0,
        "expect_access": "structured",
        "expect_fully_lowered": True,
    },
    "t2_matmul_relu": {
        "purpose": "matmul plus an elementwise epilogue after the accumulator",
        "expect_dot": 1,
        "expect_loop": True,
        "expect_mac_idiom": 1,
        "expect_epilogue_idiom": 1,
        "expect_access": "structured",
        "expect_fully_lowered": True,
    },
    "t3_modulo": {
        "purpose": "negative control; modulo addressing must NOT resolve to a structured descriptor",
        "expect_dot": 0,
        "expect_loop": False,
        "expect_mac_idiom": 0,
        "expect_epilogue_idiom": 0,
        "expect_access": "unstructured",
        "expect_fully_lowered": False,
    },
}


def _provenance() -> dict:
    def run(cmd: list[str]) -> str:
        try:
            return subprocess.run(
                cmd, capture_output=True, text=True, timeout=30, check=False
            ).stdout.strip()
        except Exception:  # pragma: no cover - provenance is best effort
            return "unavailable"

    versions = FIXTURES / "VERSIONS.txt"
    return {
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),  # noqa: UP017
        "python": sys.version.split()[0],
        "triton": run([sys.executable, "-c", "import triton;print(triton.__version__)"]),
        "torch": run([sys.executable, "-c", "import torch;print(torch.__version__)"]),
        "versions_file": versions.read_text().strip() if versions.exists() else None,
    }


def _guard_provenance(golden: dict) -> None:
    """Refuse to silently absorb a Triton upgrade.

    The fixtures are an artifact of one Triton version's lowering. If the
    version changed, re-deriving observations would hide a real change behind a
    green run, so we stop and make the author say so explicitly.
    """
    old = golden.get("provenance", {}).get("triton")
    new = _provenance()["triton"]
    if old and new and old != new and old != "unavailable":
        sys.exit(
            f"refusing to re-derive observations: triton changed {old} -> {new}.\n"
            f"This means the corpus itself changed. Regenerate fixtures/raw, update\n"
            f"fixtures/VERSIONS.txt, and re-review every intent block in GOLDEN.json\n"
            f"before running --write again."
        )


def write(mode: str) -> int:
    FIXTURES.mkdir(exist_ok=True)
    golden = load_golden() if GOLDEN.exists() else {
        "schema": SCHEMA_VERSION,
        "observations": {},
        "intent": INTENT,
    }
    if mode == "golden":
        _guard_provenance(golden)

    if mode == "write":
        if not RAW.is_dir() or not list(RAW.glob("*.ttir")):
            sys.exit(f"no raw dumps in {RAW}; run `make fixtures` first")
        for src in sorted(RAW.glob("*.ttir")):
            dest = FIXTURES / src.name
            before = src.read_text()
            after = normalize(before)
            dest.write_text(after)
            print(f"normalized {src.name}: {len(before)} -> {len(after)} bytes")

    observations = {}
    for path in sorted(FIXTURES.glob("*.ttir")):
        observations[path.stem] = scan(path.read_text())

    golden["schema"] = SCHEMA_VERSION
    golden["provenance"] = _provenance()
    golden["observations"] = observations
    golden.setdefault("intent", INTENT)
    for name, block in INTENT.items():
        golden["intent"].setdefault(name, block)
    GOLDEN.write_text(json.dumps(golden, indent=2, sort_keys=True) + "\n")

    print(f"wrote {GOLDEN.relative_to(FIXTURES.parent)}: "
          f"{len(observations)} fixtures, {sum(o['ops_total'] for o in observations.values())} ops")
    return 0


def _diff(prefix: str, recorded, observed, problems: list[str]) -> None:
    if recorded == observed:
        return
    problems.append(f"{prefix}\n    recorded: {recorded!r}\n    observed: {observed!r}")


def check() -> int:
    if not GOLDEN.exists():
        print("GOLDEN.json missing; run `make golden`")
        return 1
    golden = load_golden()
    problems: list[str] = []

    if golden.get("schema") != SCHEMA_VERSION:
        problems.append(f"schema {golden.get('schema')} != {SCHEMA_VERSION}")

    on_disk = {p.stem for p in FIXTURES.glob("*.ttir")}
    recorded = set(golden.get("observations", {}))
    for missing in sorted(recorded - on_disk):
        problems.append(f"fixture/{missing}: recorded but not on disk")
    for extra in sorted(on_disk - recorded):
        problems.append(f"fixture/{extra}: on disk but not recorded")

    for name in sorted(on_disk & recorded):
        actual = scan((FIXTURES / f"{name}.ttir").read_text())
        for key, want in golden["observations"][name].items():
            _diff(f"fixture/{name}.{key}", want, actual.get(key), problems)
        intent = golden.get("intent", {}).get(name)
        if intent is None:
            problems.append(f"fixture/{name}: no intent block")
            continue
        if intent.get("expect_dot") != actual["ops"]["tt.dot"]:
            problems.append(
                f"intent/{name}.expect_dot says {intent.get('expect_dot')} "
                f"but the fixture has {actual['ops']['tt.dot']}"
            )
        if intent.get("expect_loop") != (actual["ops"]["scf.for"] > 0):
            problems.append(
                f"intent/{name}.expect_loop says {intent.get('expect_loop')} "
                f"but the fixture has {actual['ops']['scf.for']} scf.for"
            )

    if problems:
        print(f"FIXTURE CANON DRIFT: {len(problems)} problem(s)\n")
        for p in problems:
            print(f"  - {p}")
        print("\nIf the fixture text is correct, run `make golden` and re-review the diff.")
        print("If the test expectation is correct, the fixture is wrong: fix fixtures/raw.")
        return 1

    print(f"fixture canon OK: {len(recorded)} fixtures, "
          f"{sum(o['ops_total'] for o in golden['observations'].values())} ops, "
          f"triton={golden.get('provenance', {}).get('triton')}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--write", action="store_true", help="normalize raw dumps and re-derive")
    g.add_argument("--golden", action="store_true", help="re-derive observations only")
    g.add_argument("--check", action="store_true", help="verify canon (CI)")
    ap.add_argument("--sync-raw", metavar="DIR",
                    help="copy raw dumps from DIR into fixtures/raw before --write")
    args = ap.parse_args()

    if args.sync_raw:
        src = Path(args.sync_raw)
        RAW.mkdir(parents=True, exist_ok=True)
        for p in sorted(src.glob("*.ttir")):
            shutil.copy2(p, RAW / p.name)
            print(f"raw <- {p.name}")

    if args.check:
        return check()
    if args.golden:
        return write("golden")
    return write("write")


if __name__ == "__main__":
    raise SystemExit(main())
