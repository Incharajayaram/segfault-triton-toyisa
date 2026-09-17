#!/usr/bin/env python3
"""verify_hardware_units.py — the coalescing and bank-conflict models.

Converted from tests/unit/test_hardware_models.py. Expected values are
computed independently from the definitions (32-byte lines, N banks,
interleave 4B), shown inline.

Checks:
  H1  contiguous stride-1 warp: 1 transaction, efficiency 1.0
  H2  stride-2 warp: efficiency degrades in a computable way
  H3  fully scattered large stride: efficiency drops
  H4  misaligned base costs extra transactions
  H5  bank conflicts: 2-way on stride 8 with 4 banks (0,8,16,24 -> bank 0 twice)
  H6  bank conflicts: 16 lanes all on one bank -> 16-way
  H7  bank conflicts: unit stride across banks -> zero conflicts
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from triton_toyisa.emu.hardware import BankConflictUnit, CoalescingUnit  # noqa: E402

FAILURES: list[str] = []


def check(name, actual, expected, context=""):
    if actual == expected:
        print(f"  PASS {name}: {actual!r}")
    else:
        FAILURES.append(name)
        print(f"  FAIL {name}: actual={actual!r} expected={expected!r}"
              + (f"  ({context})" if context else ""))


def main() -> int:
    cu = CoalescingUnit(cache_line_bytes=32, warp_size=32)

    print("H1: contiguous stride-1 f32 warp (32 lanes x 4B = 128B = 4 lines... or 1 line per 8 lanes)")
    rep1 = cu.analyze(base_address=0, stride_elements=1, element_bytes=4)
    check("H1: fully coalesced", rep1.is_fully_coalesced, True)
    check("H1: efficiency 1.0", rep1.coalescing_efficiency, 1.0)
    check("H1: transacted == requested",
          rep1.transacted_bytes, rep1.requested_bytes)

    print("H2: stride-2 warp (every other f32 -> half the requested bytes per line)")
    rep2 = cu.analyze(base_address=0, stride_elements=2, element_bytes=4)
    check("H2: efficiency < 1.0", rep2.coalescing_efficiency < 1.0, True,
          f"eff={rep2.coalescing_efficiency}")
    check("H2: efficiency computable from the definition (requested/transacted)",
          rep2.coalescing_efficiency,
          rep2.requested_bytes / rep2.transacted_bytes)
    check("H2: more transactions than contiguous",
          rep2.num_transactions > rep1.num_transactions, True)

    print("H3: fully scattered (stride 17) — efficiency strictly worse than stride-2")
    rep3 = cu.analyze(base_address=0, stride_elements=17, element_bytes=4)
    check("H3: efficiency < stride-2 efficiency",
          rep3.coalescing_efficiency < rep2.coalescing_efficiency, True,
          f"stride17={rep3.coalescing_efficiency} stride2={rep2.coalescing_efficiency}")

    print("H4: misaligned base (offset 16B into a 32B line)")
    rep4 = cu.analyze(base_address=16, stride_elements=1, element_bytes=4)
    check("H4: misaligned requests >= contiguous transactions",
          rep4.num_transactions >= rep1.num_transactions, True,
          f"misaligned={rep4.num_transactions} contig={rep1.num_transactions}")

    bu = BankConflictUnit(num_banks=16, bank_width_bytes=4)

    print("H5: 16 lanes, stride 8 words (32B): lanes hit banks 0,8,0,8,... -> 8 accesses per bank across 2 banks")
    rep5 = bu.analyze(addresses=[i * 32 for i in range(16)])
    check("H5: 2-bank conflict detected (7 stalls per bank = 14 total)", rep5.total_conflicts, 14,
          f"conflicts={rep5.total_conflicts} stalls={rep5.stall_cycles}")

    print("H6a: all 16 lanes on same address -> broadcast (0 conflicts)")
    rep6a = bu.analyze(addresses=[0] * 16)
    check("H6a: broadcast has zero conflict", rep6a.total_conflicts, 0,
          f"conflicts={rep6a.total_conflicts}")

    print("H6b: 16 distinct addresses all mapping to bank 0 -> 16-way conflict (15 stalls)")
    rep6b = bu.analyze(addresses=[i * 64 for i in range(16)])
    check("H6b: maximal conflict", rep6b.total_conflicts, 15,
          f"conflicts={rep6b.total_conflicts}")

    print("H7: unit stride across distinct banks -> zero conflicts")
    rep7 = bu.analyze(addresses=[i * 4 for i in range(16)])
    check("H7: zero conflicts", rep7.total_conflicts, 0,
          f"conflicts={rep7.total_conflicts}")

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
