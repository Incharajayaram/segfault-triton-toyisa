"""Contract tests for isa.select using standard library unittest.

Validates candidate enumeration, predicate rejection attribution,
cost minimization, and optimality gap reporting against oracle_min.
"""

from __future__ import annotations

import unittest

from triton_toyisa.isa.schema import load_builtin
from triton_toyisa.isa.select import select
from triton_toyisa.recognize.descriptor import AccessDescriptor


class TestSelectorContract(unittest.TestCase):
    """Real contract tests for instruction selection."""

    def setUp(self) -> None:
        self.schema1 = load_builtin("toyisa1")
        self.schema2 = load_builtin("toyisa2")

    def test_dma1d_chosen_when_dma2d_inadmissible(self) -> None:
        """1D access where 2D is inadmissible: DMA1D chosen, DMA2D rejected with reason."""
        desc = AccessDescriptor(
            base="x_ptr",
            sizes=(64,),
            strides=(1,),
            offsets=(0,),
            shape=(0,),
            order=(0,),
            dtype="f32",
            loop_carried=False,
            increment=None,
        )
        report = select(self.schema1, "memory", desc, env={"x_ptr": 0})
        self.assertFalse(report.no_admissible_lowering)
        self.assertIsNotNone(report.chosen)
        self.assertEqual(report.chosen.name, "DMA1D")
        self.assertEqual(report.chosen_cost, 64.0)
        self.assertEqual(report.gap, 0.0)

        # Check candidate attribution
        rejected = [c for c in report.candidates if not c.admissible]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0].instruction.name, "DMA2D")
        self.assertIn("all_of", rejected[0].rejected_by)

    def test_dma2d_chosen_when_both_admissible(self) -> None:
        """2D tiled access: both DMA1D and DMA2D are admissible, cheaper DMA2D is chosen."""
        desc2d = AccessDescriptor(
            base="a_ptr",
            sizes=(64, 32),
            strides=(32, 1),
            offsets=(0, 0),
            shape=(0, 0),
            order=(0, 1),
            dtype="f32",
            loop_carried=False,
            increment=None,
        )
        report = select(self.schema1, "memory", desc2d, env={"a_ptr": 0})
        self.assertIsNotNone(report.chosen)
        self.assertEqual(report.chosen.name, "DMA2D")
        # 64*32 words = 2048 words. DMA1D is 2048.0, DMA2D is 0.60 * 2048 = 1228.8
        self.assertAlmostEqual(report.chosen_cost, 1228.8)
        self.assertAlmostEqual(report.gap, 0.0)

        # Both candidates should be admissible
        admissible = [c for c in report.candidates if c.admissible]
        self.assertEqual(len(admissible), 2)
        names = {c.instruction.name for c in admissible}
        self.assertEqual(names, {"DMA1D", "DMA2D"})

    def test_mac_selection_and_gap(self) -> None:
        """Test MAC candidate selection and oracle comparison."""
        tile = (64, 64, 32)
        env = {"a_base": 16, "b_base": 16, "%sam": 32, "%sak": 1}
        report = select(self.schema1, "mac", None, tile=tile, env=env)
        self.assertIsNotNone(report.chosen)
        self.assertEqual(report.chosen.name, "MAC8")
        self.assertEqual(report.chosen_cost, 2048.0)
        self.assertEqual(report.gap, 0.0)

    def test_no_admissible_lowering(self) -> None:
        """When constraints fail for all candidates, no_admissible_lowering is True."""
        desc = AccessDescriptor(
            base="unaligned_ptr",
            sizes=(64,),
            strides=(1,),
            offsets=(0,),
            shape=(0,),
            order=(0,),
            dtype="f32",
            loop_carried=False,
            increment=None,
        )
        # OPU instructions in toyisa2 require alignment 8
        report = select(self.schema2, "mac", desc, tile=(7, 7, 7), env={"a_base": 3, "b_base": 3})
        self.assertTrue(report.no_admissible_lowering)
        self.assertIsNone(report.chosen)


if __name__ == "__main__":
    unittest.main()
