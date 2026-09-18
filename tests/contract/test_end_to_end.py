"""Contract tests for end-to-end compiler lowering pipeline and numerical parity.

Validates that all four frozen fixtures (t0_vecadd, t1_matmul, t2_matmul_relu, t3_modulo)
lower cleanly to the Vortex RISC-V GPGPU ISA, execute in the bitwise-accurate emulator,
and produce outputs within derived numerical parity bounds.
"""

from __future__ import annotations

import unittest

import numpy as np

from tritonflow.isa.schema import load_builtin
from tritonflow.lower import lower_fixture


class TestEndToEndContract(unittest.TestCase):
    """End-to-end lowering and execution contract tests."""

    def test_vortex_rvgpu_schema_loads(self) -> None:
        """The extended Vortex RISC-V GPGPU ISA schema loads and validates."""
        schema = load_builtin("vortex_rvgpu")
        self.assertEqual(schema.name, "vortex_rvgpu")
        self.assertEqual(schema.schema_version, 1)
        expected_instrs = {
            "LDG", "STG", "LDS", "STS",
            "TCU_MMA16", "TCU_MMA32",
            "VADD", "VMUL", "VSUB", "VMOD", "VRELU", "VCLAMP",
            "BARRIER",
        }
        self.assertTrue(expected_instrs.issubset(set(schema.instructions.keys())))

    def test_t0_vecadd_end_to_end(self) -> None:
        """t0_vecadd lowers, executes, and yields exact 0.0 relative error."""
        ctx = lower_fixture("t0_vecadd", isa_name="vortex_rvgpu")
        self.assertEqual(ctx.unsupported, [])
        self.assertGreater(ctx.emitted_instructions, 0)
        self.assertEqual(ctx.parity_max_rel_err, 0.0)
        self.assertIsNotNone(ctx.hardware_stats)
        self.assertEqual(ctx.hardware_stats.coalescing_efficiency, 1.0)
        self.assertEqual(ctx.hardware_stats.total_bank_conflicts, 0)

    def test_t1_matmul_end_to_end(self) -> None:
        """t1_matmul lowers, executes with TCU_MMA16, and meets TF32 derived tolerance."""
        ctx = lower_fixture("t1_matmul", isa_name="vortex_rvgpu")
        self.assertEqual(ctx.unsupported, [])
        self.assertGreater(ctx.emitted_instructions, 0)
        self.assertIsNotNone(ctx.parity_max_rel_err)
        self.assertLessEqual(ctx.parity_max_rel_err, ctx.tolerance)
        self.assertIsNotNone(ctx.hardware_stats)
        self.assertGreater(ctx.hardware_stats.dram_bytes_transacted, 0)

    def test_t2_matmul_relu_end_to_end(self) -> None:
        """t2_matmul_relu lowers with TCU_MMA16 + VRELU and meets tolerance."""
        ctx = lower_fixture("t2_matmul_relu", isa_name="vortex_rvgpu")
        self.assertEqual(ctx.unsupported, [])
        self.assertGreater(ctx.emitted_instructions, 0)
        self.assertIsNotNone(ctx.parity_max_rel_err)
        self.assertLessEqual(ctx.parity_max_rel_err, ctx.tolerance)
        # Verify non-negativity from ReLU
        out = ctx.emu_outputs["out"]
        self.assertTrue(np.all(out >= 0.0))

    def test_t3_modulo_end_to_end(self) -> None:
        """t3_modulo lowers with VMOD and yields exact parity."""
        ctx = lower_fixture("t3_modulo", isa_name="vortex_rvgpu")
        self.assertEqual(ctx.unsupported, [])
        self.assertGreater(ctx.emitted_instructions, 0)
        self.assertEqual(ctx.parity_max_rel_err, 0.0)
        self.assertIsNotNone(ctx.hardware_stats)


if __name__ == "__main__":
    unittest.main()
