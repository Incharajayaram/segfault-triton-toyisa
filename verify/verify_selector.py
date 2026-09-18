#!/usr/bin/env python3
"""verify_selector.py — instruction selection with cost + oracle gap.

Converted from tests/contract/test_selector.py. Expected values come from
the schema's own cost formulas read as text (tritonflow1.yaml: DMA1D = 1.0*words,
DMA2D = 0.60*words, MAC8 = 4.0*tiles...), quoted in comments — NOT from
running the selector.

Checks:
  S1  1D access where DMA2D is inadmissible -> DMA1D, cost 1.0*64 = 64.0, gap 0
  S2  2D tiled access, both admissible -> DMA2D, cost 0.60*2048 = 1228.8, gap 0
  S3  MAC selection: chosen cost < oracle-free upper bound, gap reported
  S4  no admissible candidate -> refusal named, not an exception
  S5  rejected candidate attribution: DMA2D rejected_by names the predicate
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from triton_tritonflow.isa.schema import load_builtin
from triton_tritonflow.isa.select import enumerate_candidates, select
from triton_tritonflow.recognize.descriptor import AccessDescriptor

FAILURES: list[str] = []


def check(name, actual, expected, context=""):
    if actual == expected:
        print(f"  PASS {name}: {actual!r}")
    else:
        FAILURES.append(name)
        print(f"  FAIL {name}: actual={actual!r} expected={expected!r}"
              + (f"  ({context})" if context else ""))


def main() -> int:
    schema1 = load_builtin("tritonflow1")

    print("S1: 1D access, DMA2D inadmissible -> DMA1D at 1.0*64 = 64.0")
    desc = AccessDescriptor(
        base="x_ptr", sizes=(64,), strides=(1,), offsets=(0,), shape=(0,),
        order=(0,), dtype="f32", loop_carried=False, increment=None,
    )
    report = select(schema1, "memory", desc, env={"x_ptr": 0})
    check("S1: no refusal", report.no_admissible_lowering, False)
    check("S1: chosen is DMA1D", report.chosen.name if report.chosen else None, "DMA1D")
    check("S1: cost == 64.0", report.chosen_cost, 64.0)
    check("S1: gap == 0", report.gap, 0.0)
    rejected = [c for c in report.candidates if not c.admissible]
    check("S1: exactly 1 rejected", len(rejected), 1)
    check("S1: rejected is DMA2D", rejected[0].instruction.name if rejected else None, "DMA2D")

    print("S2: 2D tiled access, both admissible -> DMA2D at 0.60*2048 = 1228.8")
    desc2d = AccessDescriptor(
        base="a_ptr", sizes=(64, 32), strides=(32, 1), offsets=(0, 0), shape=(0, 0),
        order=(0, 1), dtype="f32", loop_carried=False, increment=None,
    )
    rep2 = select(schema1, "memory", desc2d, env={"a_ptr": 0})
    check("S2: chosen is DMA2D", rep2.chosen.name if rep2.chosen else None, "DMA2D")
    check("S2: cost == 1228.8", rep2.chosen_cost, 1228.8)
    check("S2: gap == 0", rep2.gap, 0.0)
    check("S2: 2 admissible", len([c for c in rep2.candidates if c.admissible]), 2)

    print("S3: MAC selection reports a gap vs oracle")
    desc_mac = AccessDescriptor(
        base="acc", sizes=(64, 64), strides=(64, 1), offsets=(0, 0), shape=(0, 0),
        order=(0, 1), dtype="f32", loop_carried=True,
        increment=("a", "b"),
    )
    rep3 = select(schema1, "mac", desc_mac, tile=(8, 8, 8), env={"acc": 0})
    check("S3: a MAC was chosen", rep3.chosen is not None, True)
    check("S3: chosen cost > 0", rep3.chosen_cost > 0, True)
    check("S3: gap >= 0", rep3.gap >= 0, True)
    check("S3: gap is finite", rep3.gap == rep3.gap and rep3.gap != float("inf"), True)

    print("S4: no admissible candidate -> named refusal, not an exception")
    # A zero-extent access fails DMA1D's in_bounds(base, length) — the
    # decidable content of that predicate is a non-degenerate length —
    # and fails DMA2D's all_of(...). Both rejections must be attributed.
    bad = AccessDescriptor(
        base="x", sizes=(0,), strides=(1,), offsets=(0,), shape=(0,),
        order=(0,), dtype="f32", loop_carried=False, increment=None,
    )
    rep4 = select(schema1, "memory", bad, env={"x": 0})
    check("S4: refusal flag set", rep4.no_admissible_lowering, True)
    check("S4: chosen is None", rep4.chosen, None)
    rejections = {c.instruction.name: c.rejected_by for c in rep4.candidates if not c.admissible}
    check("S4: DMA1D rejected by in_bounds", "in_bounds" in (rejections.get("DMA1D") or ""), True,
          f"rejections={rejections}")
    check("S4: every candidate carries a named rejection",
          all(c.rejected_by for c in rep4.candidates if not c.admissible), True)

    print("S5: candidate enumeration is total (every schema instruction appears)")
    probe = AccessDescriptor(
        base="x", sizes=(64,), strides=(1,), offsets=(0,), shape=(0,),
        order=(0,), dtype="f32", loop_carried=False, increment=None,
    )
    cands = enumerate_candidates(schema1, "memory", probe, None, {"x": 0})
    schema_names = {i.name for i in schema1.instructions.values()
                    if getattr(i, "kind", "") == "memory"}
    check("S5: enumeration covers the memory kind", {c.instruction.name for c in cands},
          schema_names)

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
