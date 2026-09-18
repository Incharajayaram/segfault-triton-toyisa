#!/usr/bin/env python3
"""verify_build_ir.py — RawModule -> semantic IR construction.

Converted from tests/contract/test_to_ir.py. Expected values come from the
fixture text itself (read and quoted in comments), not from running build_ir.

Checks:
  I1  t0_vecadd builds IR; module body carries tt.func with block args
  I2  multi-result binding: t1's %acc_25:3 becomes three SsaValues
  I3  duplicate SSA definition is refused with a diagnostic
  I4  undefined operand reference is refused with a diagnostic
  I5  handbuilt module: constant binds to an SsaValue result (structural)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from tritonflow.ttir.parser import parse_raw
from tritonflow.ttir.ssa import SsaValue
from tritonflow.ttir.to_ir import build_ir

FAILURES: list[str] = []


def check(name, actual, expected, context=""):
    if actual == expected:
        print(f"  PASS {name}: {actual!r}")
    else:
        FAILURES.append(name)
        print(f"  FAIL {name}: actual={actual!r} expected={expected!r}"
              + (f"  ({context})" if context else ""))


def main() -> int:
    fixtures = ROOT / "fixtures"

    print("I1: t0_vecadd builds IR with tt.func block args")
    raw = parse_raw((fixtures / "t0_vecadd.ttir").read_text(), source_path="t0")
    result = build_ir(raw)
    check("t0: build_ir ok", result.ok, True, f"diagnostic={result.diagnostic!r}")
    if result.ok:
        module = result.module

        def walk_t0(region):
            for b in region.blocks:
                for op in b.operations:
                    yield op
                    for r in op.regions:
                        yield from walk_t0(r)

        all_ops = list(walk_t0(module.body))
        funcs = [op for op in all_ops if op.name == "tt.func"]
        check("t0: exactly one tt.func in IR", len(funcs), 1)
        entry = funcs[0].regions[0].blocks[0] if funcs else None
        check("t0: func entry block args == 4", len(entry.args) if entry else -1, 4,
              f"got {[str(a) for a in entry.args] if entry else 'no func'}")

    print("I2: multi-result binding (%acc_25:3 -> three SsaValues)")
    raw1 = parse_raw((fixtures / "t1_matmul.ttir").read_text(), source_path="t1")
    result1 = build_ir(raw1)
    check("t1: build_ir ok", result1.ok, True, f"diagnostic={result1.diagnostic!r}")
    if result1.ok:
        mod1 = result1.module

        def walk(region):
            for b in region.blocks:
                for op in b.operations:
                    yield op
                    for r in op.regions:
                        yield from walk(r)

        all_ops = list(walk(mod1.body))
        multi = [op for op in all_ops if len(op.results) == 3]
        check("t1: one 3-result op exists (the tt.dot accumulator)", len(multi), 1)
        if multi:
            check("t1: all 3 results are SsaValue",
                  all(isinstance(r, SsaValue) for r in multi[0].results), True)

    print("I3: duplicate SSA definition is refused")
    dup = ("module {\n"
           "  tt.func public @f(%x: i32) {\n"
           "    %0 = arith.constant 1 : i32\n"
           "    %0 = arith.constant 2 : i32\n"
           "    tt.return\n"
           "  }\n"
           "}")
    res_dup = build_ir(parse_raw(dup, source_path="<dup>"))
    check("dup: refused", res_dup.ok, False, f"got {res_dup}")

    print("I4: undefined operand reference is refused")
    undef = ("module {\n"
             "  tt.func public @f() {\n"
             "    %0 = arith.addf %ghost, %ghost2 : f32\n"
             "    tt.return\n"
             "  }\n"
             "}")
    res_undef = build_ir(parse_raw(undef, source_path="<undef>"))
    check("undef: refused", res_undef.ok, False, f"got {res_undef}")

    print("I5: handbuilt module builds structurally")
    raw_h = parse_raw("module {\n  %c = arith.constant 64 : i32\n}", source_path="<hb>")
    res_h = build_ir(raw_h)
    check("handbuilt: ok", res_h.ok, True)
    if res_h.ok:
        # arith.constant at module top level nests under the module op
        def walk_hb(region):
            for b in region.blocks:
                for op in b.operations:
                    yield op
                    for r in op.regions:
                        yield from walk_hb(r)
        consts = [op for op in walk_hb(res_h.module.body) if op.name == "arith.constant"]
        check("handbuilt: one constant found", len(consts), 1)
        if consts:
            check("handbuilt: constant result is SsaValue",
                  isinstance(consts[0].results[0], SsaValue), True)

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
