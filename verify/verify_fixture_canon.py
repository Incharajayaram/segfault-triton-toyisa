#!/usr/bin/env python3
"""verify_fixture_canon.py — the fixture corpus vs its frozen canon.

Converted from tests/contract/test_fixtures.py (v1, the intent tests v2
dropped) + tests/contract/test_fixtures.py (v2, hash/observation checks).
Every expected value comes from fixtures/GOLDEN.json (the reviewed canon),
NOT from running pipeline code.

Checks:
  F1  recorded sha256 == file sha256, per tier
  F2  every recorded observation matches the file (canon agrees both ways)
  F3  no absolute paths in any fixture
  F4  normalization is idempotent and total
  F5  intent: t0 is the true-negative control (tt.dot == 0)
  F6  intent: t1 is the primary kernel (exactly 1 tt.dot, 1 scf.for)
  F7  intent: t1 loop recurrence shape (iter_args advanced by constant)
  F8  intent: t1 declares tf32 (tolerance must be derived, not chosen)
  F9  intent: t2 adds an epilogue (one extra tt.load vs t1)
  F10 intent: t3 is unstructured and must not resolve
  F11 intent: region nesting depths (t0/t3=2, t1/t2=3)
  F12 intent: loc table carries the python variable names
  F13 intent: the corpus is a graduated ladder (4 distinct files, >=3 op vocab)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from fixtures_lib import FIXTURES, load_golden, normalize, read_fixture, scan  # noqa: E402

FAILURES: list[str] = []
ABS_PATH = re.compile(r'loc\("(/[^"]*)"')


def check(name, actual, expected, context=""):
    if actual == expected:
        print(f"  PASS {name}: {actual!r}")
    else:
        FAILURES.append(name)
        print(f"  FAIL {name}: actual={actual!r} expected={expected!r}"
              + (f"  ({context})" if context else ""))


def main() -> int:
    golden = load_golden()
    TIERS = sorted(golden["intent"])
    obs = {n: scan(read_fixture(n)) for n in TIERS}

    print("F1: recorded sha256 == file sha256")
    for n in TIERS:
        check(f"{n}: sha256", obs[n]["sha256"], golden["observations"][n]["sha256"])

    print("F2: every recorded observation matches the file")
    for n in TIERS:
        for key, want in golden["observations"][n].items():
            if key in ("intent",):
                continue
            check(f"{n}.{key}", obs[n].get(key), want)

    print("F3: no absolute paths in any fixture")
    for n in TIERS:
        check(f"{n}: no abs path", bool(ABS_PATH.search(read_fixture(n))), False)

    print("F4: normalization is idempotent and total")
    for n in TIERS:
        text = read_fixture(n)
        once = normalize(text)
        check(f"{n}: normalize idempotent", normalize(once), once)

    print("F5: t0 is the true-negative control")
    check("t0.tt.dot == 0", obs["t0_vecadd"]["ops"].get("tt.dot", 0), 0)

    print("F6: t1 is the primary kernel")
    check("t1.tt.dot == 1", obs["t1_matmul"]["ops"].get("tt.dot", 0), 1)
    check("t1.scf.for == 1", obs["t1_matmul"]["ops"].get("scf.for", 0), 1)

    print("F7: t1 loop recurrence shape (pointers are iter_args advanced by a constant)")
    t1_text = read_fixture("t1_matmul")
    loop_line = next(line for line in t1_text.splitlines() if "scf.for" in line)
    check("t1: loop carries iter_args", "iter_args" in loop_line, True)
    check("t1: pointers advanced via tt.addptr on iter_args",
          "%a_ptrs" in t1_text and "%b_ptrs" in t1_text, True)

    print("F8: t1 declares tf32 so tolerance must be derived, not chosen")
    check("t1.has_tf32", obs["t1_matmul"].get("has_tf32"), True)

    print("F9: t2 adds an epilogue after the accumulator")
    check("t2.tt.dot == t1.tt.dot", obs["t2_matmul_relu"]["ops"]["tt.dot"],
          obs["t1_matmul"]["ops"]["tt.dot"])
    check("t2.tt.load == t1.tt.load + 1",
          obs["t2_matmul_relu"]["ops"]["tt.load"],
          obs["t1_matmul"]["ops"]["tt.load"] + 1)

    print("F10: t3 is unstructured and must not resolve")
    check("t3.arith.remsi >= 1", obs["t3_modulo"]["ops"].get("arith.remsi", 0) >= 1, True)
    check("t3.tt.dot == 0", obs["t3_modulo"]["ops"].get("tt.dot", 0), 0)

    print("F11: region nesting is exercised by t1/t2")
    depths = {n: obs[n]["max_region_depth"] for n in TIERS}
    check("t0/t3 depth 2", (depths["t0_vecadd"], depths["t3_modulo"]), (2, 2))
    check("t1/t2 depth 3", (depths["t1_matmul"], depths["t2_matmul_relu"]), (3, 3))

    print("F12: loc table carries python variable names (the debugging story)")
    names = set(obs["t1_matmul"]["loc_names"])
    check("t1 loc names superset",
          {"a_ptrs", "b_ptrs", "c_ptrs", "acc", "a_ptr", "b_ptr", "c_ptr"} <= names, True)

    print("F13: the corpus is a graduated ladder, not four copies")
    check("4 distinct files", len({obs[n]["sha256"] for n in TIERS}), 4)
    check(">=3 distinct op vocab sizes",
          len({obs[n]["op_vocab"] for n in TIERS}) >= 3, True)

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
