"""Contract tests for emu.exec and emu.precision using standard library unittest.

Validates tolerance derivation formulas, TF32 truncation bit-arithmetic,
and emulator execution semantics.
"""

from __future__ import annotations

import unittest

import numpy as np

from tritonflow.emit.ir import (
    Instr,
    MemRef,
    Program,
    SourceRef,
    UnsupportedMarker,
)
from tritonflow.emu.exec import (
    MissingInput,
    ProgramNotExecutable,
    emulate,
)
from tritonflow.emu.precision import (
    derive_tolerance,
    tf32_truncate,
)


class TestEmulatorContract(unittest.TestCase):
    """Real contract tests for emulator and precision policy."""

    def test_derive_tolerance_ieee_fp32(self) -> None:
        """IEEE FP32 tolerance derivation must match reduction_length * 2**-24."""
        tol = derive_tolerance("ieee", 32, "f32")
        expected = 32 * (2.0 ** -24)
        self.assertAlmostEqual(tol.value, expected)
        self.assertFalse(tol.absolute)
        self.assertIn("ieee fp32", tol.derivation)

    def test_derive_tolerance_tf32(self) -> None:
        """TF32 tolerance derivation accounts for 10-bit mantissa truncation + fp32 accumulation."""
        tol = derive_tolerance("tf32", 32, "f32")
        truncation = 2.0 * (2.0 ** -11)
        accumulation = 2.0 ** -24
        expected = 32 * (truncation + accumulation)
        self.assertAlmostEqual(tol.value, expected)
        self.assertFalse(tol.absolute)
        self.assertIn("tf32 inputs", tol.derivation)

    def test_derive_tolerance_integer_exactness(self) -> None:
        """Integer reductions are exact by contract: tolerance is 0, absolute."""
        tol = derive_tolerance("ieee", 64, "i32")
        self.assertEqual(tol.value, 0.0)
        self.assertTrue(tol.absolute)

    def test_derive_tolerance_invalid_arguments(self) -> None:
        with self.assertRaises(ValueError):
            derive_tolerance("unknown_precision", 32, "f32")  # type: ignore
        with self.assertRaises(ValueError):
            derive_tolerance("ieee", -1, "f32")

    def test_tf32_truncate_preserves_powers_of_two(self) -> None:
        """Exact powers of two have zero mantissa fraction and must survive truncation unchanged."""
        exact = np.array([0.0, 1.0, 2.0, 4.0, 0.5, 0.25, -8.0], dtype=np.float32)
        truncated = tf32_truncate(exact)
        np.testing.assert_array_equal(exact, truncated)

    def test_tf32_truncate_rounds_low_bits(self) -> None:
        """Values with precision beyond 10 mantissa bits must be rounded."""
        # 1.0 + 2**-23 has 1 bit at bit 0 of mantissa.
        # In TF32 (10 bits), this gets truncated to 1.0.
        val = np.array([1.0 + 2.0**-20], dtype=np.float32)
        truncated = tf32_truncate(val)
        self.assertEqual(truncated[0], 1.0)

    def test_emulate_halts_on_unsupported_marker(self) -> None:
        """Program carrying UNSUPPORTED marker raises ProgramNotExecutable."""
        p = Program(
            isa_name="tritonflow1",
            schema_version=1,
            kernel_name="unsupported_op",
            total_cost=0.0,
            inputs=("%x",),
            instrs=(),
            unsupported=(
                UnsupportedMarker(
                    op_name="tt.gather",
                    kind="UNSUPPORTED",
                    reason="non-affine access",
                    loc_name="scatter_idx",
                ),
            ),
        )
        with self.assertRaises(ProgramNotExecutable):
            emulate(p, {"%x": np.zeros(4, dtype=np.float32)})

    def test_emulate_raises_on_missing_input(self) -> None:
        """Program addressing an undeclared input raises MissingInput."""
        p = Program(
            isa_name="tritonflow1",
            schema_version=1,
            kernel_name="missing_in",
            total_cost=4.0,
            inputs=("%x",),
            instrs=(
                Instr(
                    name="DMA1D",
                    cost=4.0,
                    loop=None,
                    defs=("%val",),
                    operands={"src": MemRef.of("global", "%x")},
                    constraint=None,
                    constrained_on=None,
                    source_ops=(SourceRef("tt.load", 1, 1, "x"),),
                ),
            ),
        )
        # We omit %x from inputs dict
        with self.assertRaises(MissingInput):
            emulate(p, {})


if __name__ == "__main__":
    unittest.main()
