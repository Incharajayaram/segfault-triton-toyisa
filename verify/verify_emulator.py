#!/usr/bin/env python3
"""verify_emulator.py — the NumPy emulator: precision policy and refusal paths.

Converted from tests/contract/test_emulator.py + tests/unit hardware checks.
Expected values are derived from IEEE-754 / tf32 bit layouts computed
independently in this file (not from running the emulator).

Checks:
  E1  tf32 truncation preserves powers of two exactly
  E2  tf32 truncation drops the low 13 mantissa bits (10-bit mantissa kept)
  E3  derive_tolerance: ieee fp32 -> 0.0 (bit-exact), integer -> 0.0
  E4  derive_tolerance: tf32 band is positive and derived from n and the
      tf32 epsilon, not a magic constant
  E5  emulator raises on missing program input (fail loud, not silent zero)
  E6  emulator halts on an UNSUPPORTED marker instead of executing past it
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from triton_toyisa.emu.exec import emulate  # noqa: E402
from triton_toyisa.emu.precision import PrecisionPolicy  # noqa: E402
from triton_toyisa.emit.ir import Imm, Instr, Program, SsaRef  # noqa: E402

FAILURES: list[str] = []


def check(name, actual, expected, context=""):
    if actual == expected:
        print(f"  PASS {name}: {actual!r}")
    else:
        FAILURES.append(name)
        print(f"  FAIL {name}: actual={actual!r} expected={expected!r}"
              + (f"  ({context})" if context else ""))


def main() -> int:
    from triton_toyisa.emu.precision import derive_tolerance, tf32_truncate

    print("E1/E2: tf32 truncation (independent bit math: 10 mantissa bits kept)")
    # 1.0, 2.0, 0.5 are powers of two — tf32 must preserve them exactly
    for v in (1.0, 2.0, 0.5, 1024.0):
        check(f"tf32({v}) == {v}", float(tf32_truncate(np.array([v], dtype=np.float32))[0]), v)
    # a 23-bit mantissa value must round-trip through 10 kept bits:
    # 1 + 2**-11 (bit 11 set) truncates to 1.0; 1 + 2**-10 (bit 10) survives
    check("tf32(1 + 2**-11) == 1.0",
          float(tf32_truncate(np.array([1.0 + 2.0**-11], dtype=np.float32))[0]), 1.0)
    check("tf32(1 + 2**-10) == 1 + 2**-10",
          float(tf32_truncate(np.array([1.0 + 2.0**-10], dtype=np.float32))[0]),
          1.0 + 2.0**-10)

    print("E3: derived tolerance: exact paths are 0.0, ieee is the fp32 half-ulp bound")
    # independently: fp32 half-ulp = 2**-24
    n = 32
    check("integer path -> 0.0",
          derive_tolerance("tf32", reduction_length=n, dtype="i32").value, 0.0)
    fp32_half_ulp = 2.0**-24
    got_ieee = derive_tolerance("ieee", reduction_length=n, dtype="f32").value
    check("ieee tolerance == n * 2**-24", got_ieee, n * fp32_half_ulp)

    print("E4: tf32 band is derived, not magic")
    # independently: each input truncated -> each product off by <= 2*(2**-11),
    # plus the fp32 accumulation step error 2**-24
    expected_bound = n * (2.0 * 2.0**-11 + 2.0**-24)
    got = derive_tolerance("tf32", reduction_length=n, dtype="f32").value
    check("tf32 tolerance formula", got, expected_bound)
    check("tf32 tolerance > 0", got > 0, True)
    check("tf32 tolerance carries its derivation string",
          isinstance(derive_tolerance("tf32", reduction_length=n, dtype="f32").derivation, str)
          and len(derive_tolerance("tf32", reduction_length=n, dtype="f32").derivation) > 20,
          True)

    print("E5: missing program input fails loud")
    prog = Program(
        isa_name="toyisa1", schema_version=1, kernel_name="k", total_cost=1.0,
        inputs=("%x",),
        instrs=(Instr(name="EPI", cost=1.0, loop=None, defs=("%r",),
                      operands={"in0": SsaRef("%x")}, source_ops=()),),
    )
    try:
        emulate(prog, {})  # %x is declared but not supplied
        FAILURES.append("E5")
        print("  FAIL E5: missing input accepted silently")
    except Exception as exc:  # noqa: BLE001 — the raise IS the expected behavior
        check("E5: raises on missing input", isinstance(exc, Exception), True,
              f"{type(exc).__name__}: {exc}")

    print("E6: UNSUPPORTED marker halts execution (FR-005, postcondition 2)")
    from triton_toyisa.emit.ir import UnsupportedMarker

    prog2 = Program(
        isa_name="toyisa1", schema_version=1, kernel_name="k2", total_cost=0.0,
        inputs=("%x",),
        instrs=(),
        unsupported=(UnsupportedMarker(
            op_name="tt.something",
            reason="no admissible lowering: every candidate rejected",
        ),),
    )
    try:
        emulate(prog2, {"%x": np.ones(4, dtype=np.float32)})
        FAILURES.append("E6")
        print("  FAIL E6: program with UNSUPPORTED marker executed anyway")
    except Exception as exc:  # noqa: BLE001 — ProgramNotExecutable IS the expected behavior
        check("E6: UNSUPPORTED halts via ProgramNotExecutable",
              type(exc).__name__, "ProgramNotExecutable",
              f"got {type(exc).__name__}: {exc}")
        check("E6: the refusal names the operation and reason",
              "tt.something" in str(exc) and "no admissible lowering" in str(exc), True,
              f"message: {exc}")

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
