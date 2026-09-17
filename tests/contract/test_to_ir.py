"""Contract tests for ttir.to_ir using standard library unittest.

Validates the transformation of RawModule into semantic Module,
enforcing SSA invariants (definition before use, uniqueness, multi-result binding).
"""

from __future__ import annotations

import unittest

from triton_toyisa.ttir.parser import RawLoc, RawModule, RawOp
from triton_toyisa.ttir.ssa import Module, SsaValue
from triton_toyisa.ttir.to_ir import build_ir


class TestToIrContract(unittest.TestCase):
    """Real contract tests for semantic IR construction."""

    def test_build_ir_basic_structure(self) -> None:
        """build_ir converts RawOp into a typed Operation with bound SsaValues."""
        dummy_raw = RawModule(
            ops=[
                RawOp(
                    name="arith.constant",
                    results=["%c64_i32"],
                    operands=[],
                    attrs={"value": "64"},
                    result_types=["i32"],
                    operand_types=[],
                    regions=[],
                    loc=RawLoc(name="#loc1"),
                    line=10,
                    col=4,
                )
            ],
            loc_table={"#loc1": RawLoc(name="#loc1", line=10, col=4)},
            source_path="<dummy>",
            triton_version="3.7.1",
            diagnostics=[],
        )
        result = build_ir(dummy_raw)
        self.assertTrue(result.ok, f"handbuilt input refused: {result.diagnostic}")
        module = result.unwrap()
        self.assertIsInstance(module, Module)
        (op,) = module.body.blocks[0].operations
        self.assertEqual(op.name, "arith.constant")
        self.assertEqual(len(op.results), 1)
        self.assertIsInstance(op.results[0], SsaValue)
        self.assertEqual(op.results[0].type.dtype, "i32")

    def test_build_ir_multi_result_binding(self) -> None:
        """Every result of a multi-result operation must be bound as an SsaValue."""
        raw = RawModule(
            ops=[
                RawOp(
                    name="arith.constant",
                    results=["%c0"],
                    operands=[],
                    attrs={"value": "0"},
                    result_types=["i32"],
                    operand_types=[],
                    regions=[],
                    loc=RawLoc(name="#loc1"),
                    line=10,
                    col=4,
                ),
                RawOp(
                    name="scf.for",
                    results=["%acc_0", "%acc_1", "%acc_2"],
                    operands=["%c0"],
                    attrs={},
                    result_types=[
                        "tensor<64x64xf32>",
                        "tensor<64x64xf32>",
                        "tensor<64x64xf32>",
                    ],
                    operand_types=["i32"],
                    regions=[],
                    loc=RawLoc(name="#loc1"),
                    line=12,
                    col=4,
                ),
            ],
            loc_table={"#loc1": RawLoc(name="#loc1", line=10, col=4)},
            source_path="<test>",
            triton_version="3.7.1",
            diagnostics=[],
        )
        result = build_ir(raw)
        self.assertTrue(result.ok)
        module = result.unwrap()
        for_op = module.body.blocks[0].operations[1]
        self.assertEqual(for_op.name, "scf.for")
        self.assertEqual(len(for_op.results), 3)
        self.assertEqual([v.name for v in for_op.results], ["%acc_0", "%acc_1", "%acc_2"])
        for v in for_op.results:
            self.assertEqual(v.type.kind, "tensor")
            self.assertEqual(v.type.shape, (64, 64))

    def test_build_ir_refuses_undefined_operand(self) -> None:
        """An operand referencing an undefined SSA name must be rejected by IR layer."""
        raw = RawModule(
            ops=[
                RawOp(
                    name="arith.addi",
                    results=["%res"],
                    operands=["%undefined_var"],
                    attrs={},
                    result_types=["i32"],
                    operand_types=["i32"],
                    regions=[],
                    loc=RawLoc(name="#loc1"),
                    line=10,
                    col=4,
                )
            ],
            loc_table={"#loc1": RawLoc(name="#loc1", line=10, col=4)},
            source_path="<test>",
            triton_version="3.7.1",
            diagnostics=[],
        )
        result = build_ir(raw)
        self.assertFalse(result.ok)
        self.assertEqual(result.diagnostic.layer, "ir")
        self.assertEqual(result.diagnostic.found, "%undefined_var")

    def test_build_ir_refuses_duplicate_ssa_definition(self) -> None:
        """Defining the same SSA name twice is invalid IR (EC-026)."""
        raw = RawModule(
            ops=[
                RawOp(
                    name="arith.constant",
                    results=["%dup"],
                    operands=[],
                    attrs={"value": "1"},
                    result_types=["i32"],
                    operand_types=[],
                    regions=[],
                    loc=RawLoc(name="#loc1"),
                    line=10,
                    col=4,
                ),
                RawOp(
                    name="arith.constant",
                    results=["%dup"],
                    operands=[],
                    attrs={"value": "2"},
                    result_types=["i32"],
                    operand_types=[],
                    regions=[],
                    loc=RawLoc(name="#loc1"),
                    line=11,
                    col=4,
                ),
            ],
            loc_table={"#loc1": RawLoc(name="#loc1", line=10, col=4)},
            source_path="<test>",
            triton_version="3.7.1",
            diagnostics=[],
        )
        result = build_ir(raw)
        self.assertFalse(result.ok)
        self.assertEqual(result.diagnostic.layer, "ir")


if __name__ == "__main__":
    unittest.main()
