"""Unit tests for GPU hardware modeling components: memory coalescing and bank conflicts.

Tests warp-level memory request coalescing (128-byte/32-byte cache lines) and
shared memory/scratchpad multi-bank arbitration (16/32 banks).
"""

from __future__ import annotations

import unittest

from tritonflow.emu.hardware import BankConflictUnit, CoalescingUnit


class TestHardwareModels(unittest.TestCase):
    """Unit tests for CoalescingUnit and BankConflictUnit."""

    def test_coalescing_unit_contiguous_stride_1(self) -> None:
        """Stride 1 warp access (32 threads x 4 bytes = 128 bytes) forms exactly 1 transaction."""
        coalescer = CoalescingUnit(cache_line_bytes=128, warp_size=32)
        report = coalescer.analyze(base_address=0, stride_elements=1, element_bytes=4)
        self.assertEqual(report.active_threads, 32)
        self.assertEqual(report.num_transactions, 1)
        self.assertEqual(report.requested_bytes, 128)
        self.assertEqual(report.transacted_bytes, 128)
        self.assertEqual(report.coalescing_efficiency, 1.0)
        self.assertTrue(report.is_fully_coalesced)

    def test_coalescing_unit_stride_2(self) -> None:
        """Stride 2 warp access (32 threads x 8 bytes stride = 256 bytes) spans 2 cache lines."""
        coalescer = CoalescingUnit(cache_line_bytes=128, warp_size=32)
        report = coalescer.analyze(base_address=0, stride_elements=2, element_bytes=4)
        self.assertEqual(report.active_threads, 32)
        self.assertEqual(report.num_transactions, 2)
        self.assertEqual(report.requested_bytes, 128)
        self.assertEqual(report.transacted_bytes, 256)
        self.assertEqual(report.coalescing_efficiency, 0.5)

    def test_coalescing_unit_uncoalesced_large_stride(self) -> None:
        """Large stride where each thread hits a separate cache line -> 32 transactions."""
        coalescer = CoalescingUnit(cache_line_bytes=128, warp_size=32)
        report = coalescer.analyze(base_address=0, stride_elements=32, element_bytes=4)
        self.assertEqual(report.num_transactions, 32)
        self.assertEqual(report.requested_bytes, 128)
        self.assertEqual(report.transacted_bytes, 32 * 128)
        self.assertAlmostEqual(report.coalescing_efficiency, 128 / (32 * 128))
        self.assertFalse(report.is_fully_coalesced)

    def test_coalescing_misaligned_base(self) -> None:
        """Base offset misaligned by 64 bytes inside 128-byte line splits into 2 cache lines."""
        coalescer = CoalescingUnit(cache_line_bytes=128, warp_size=32)
        report = coalescer.analyze(base_address=64, stride_elements=1, element_bytes=4)
        # 64 to 64 + 128 = 192, spanning lines [0..128) and [128..256)
        self.assertEqual(report.num_transactions, 2)
        self.assertEqual(report.coalescing_efficiency, 0.5)

    def test_bank_conflicts_zero_conflicts(self) -> None:
        """Sequential word access across 16 banks -> 16 distinct banks, 0 conflicts, 0 stalls."""
        bank_unit = BankConflictUnit(num_banks=16, bank_width_bytes=4)
        addresses = [i * 4 for i in range(16)]
        report = bank_unit.analyze(addresses)
        self.assertEqual(report.total_conflicts, 0)
        self.assertEqual(report.max_conflict_degree, 1)
        self.assertEqual(report.stall_cycles, 0)
        self.assertEqual(len(report.bank_access_counts), 16)
        self.assertTrue(all(c == 1 for c in report.bank_access_counts.values()))

    def test_bank_conflicts_2_way(self) -> None:
        """Stride 2 across 16 banks -> 8 banks hit twice each -> 2-way conflict, 1 stall cycle."""
        bank_unit = BankConflictUnit(num_banks=16, bank_width_bytes=4)
        # 16 requests with stride 2: 0, 8, 16, 24, ...
        addresses = [i * 8 for i in range(16)]
        report = bank_unit.analyze(addresses)
        self.assertEqual(report.max_conflict_degree, 2)
        self.assertEqual(report.stall_cycles, 1)
        self.assertEqual(report.total_conflicts, 8)

    def test_bank_conflicts_16_way_all_hit_bank_0(self) -> None:
        """All 16 threads hit different words in bank 0 -> 16-way conflict, 15 stall cycles."""
        bank_unit = BankConflictUnit(num_banks=16, bank_width_bytes=4)
        # Bank 0 is hit for byte addresses 0, 64, 128, 192... (multiples of num_banks * bank_width = 64)
        addresses = [i * 64 for i in range(16)]
        report = bank_unit.analyze(addresses)
        self.assertEqual(report.max_conflict_degree, 16)
        self.assertEqual(report.stall_cycles, 15)
        self.assertEqual(report.total_conflicts, 15)


if __name__ == "__main__":
    unittest.main()
