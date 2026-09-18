"""Contract tests for isa.schema using standard library unittest.

Validates declarative ISA schemas (tritonflow1, tritonflow2), predicate logic,
cost expressions, and fail-closed admissibility.
"""

from __future__ import annotations

import unittest

from triton_tritonflow.isa.schema import (
    IsaSchema,
    cost_of,
    evaluate,
    load_builtin,
    validate_schema,
)
from triton_tritonflow.recognize.descriptor import AccessDescriptor
from triton_tritonflow.recognize.walk import SymExpr


class TestIsaSchemaContract(unittest.TestCase):
    """Real contract tests for the declarative ISA schema specification."""

    def test_load_builtin_tritonflow1(self) -> None:
        schema = load_builtin("tritonflow1")
        self.assertIsInstance(schema, IsaSchema)
        self.assertEqual(schema.name, "tritonflow1")
        self.assertEqual(schema.schema_version, 1)

        # Invariants: must have memory spaces and instructions
        self.assertGreater(len(schema.data_model.memory_spaces), 0)
        self.assertEqual(schema.data_model.memory_spaces[0].name, "global")
        self.assertEqual(schema.data_model.memory_spaces[0].alignment_words, 4)

        # Expected instructions in tritonflow1
        inst_names = list(schema.instructions.keys())
        for expected in ("DMA1D", "DMA2D", "MAC8", "MAC16", "EPI"):
            self.assertIn(expected, inst_names)

        # Validate schema passes without violation
        violations = validate_schema(schema)
        self.assertEqual(len(violations), 0, f"Schema validation violations: {violations}")

    def test_load_builtin_tritonflow2(self) -> None:
        schema = load_builtin("tritonflow2")
        self.assertIsInstance(schema, IsaSchema)
        self.assertEqual(schema.name, "tritonflow2")

        # tritonflow2 has banked scratchpad & accumulator
        spaces = {space.name: space for space in schema.data_model.memory_spaces}
        self.assertIn("global", spaces)
        self.assertIn("scratch", spaces)
        self.assertIn("accum", spaces)

        inst_names = list(schema.instructions.keys())
        for expected in ("LDG", "LDS2D", "OPU32", "OPU8", "VPU", "CLAMP"):
            self.assertIn(expected, inst_names)

        violations = validate_schema(schema)
        self.assertEqual(len(violations), 0, f"tritonflow2 schema violations: {violations}")

    def test_predicate_evaluation_admissibility(self) -> None:
        """Test predicate evaluation for aligned, in_bounds, and arithmetic."""
        desc = AccessDescriptor(
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
        env = {"a_ptr": 128, "%sam": 32, "%sak": 1}

        # aligned check
        self.assertTrue(evaluate("aligned(base, 4)", desc, env=env))
        self.assertTrue(evaluate("stride[1] == 1", desc, env=env))
        self.assertTrue(evaluate("all_of(stride[1] == 1, aligned(base, 4))", desc, env=env))

        # False condition
        self.assertFalse(evaluate("stride[1] == 4", desc, env=env))
        self.assertFalse(evaluate("aligned(base, 256)", desc, env=env))

    def test_fail_closed_on_unknown(self) -> None:
        """Unknown facts must evaluate to 'unknown' rather than passing."""
        desc = AccessDescriptor(
            base="unresolved_ptr",
            sizes=(64, 32),
            strides=(SymExpr.symbol("%sam"), 1),
            offsets=(0, 0),
            shape=(0, 0),
            order=(0, 1),
            dtype="f32",
            loop_carried=False,
            increment=None,
        )
        # Without env, stride[0] is symbolic, so stride[0] % 4 is undecidable
        verdict = evaluate("stride[0] % 4 == 0", desc, env={})
        self.assertEqual(verdict, "unknown")

    def test_cost_evaluation(self) -> None:
        schema = load_builtin("tritonflow1")
        dma1d = schema.instruction("DMA1D")
        self.assertIsNotNone(dma1d)

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
        # 64 words * 1.00 = 64.0
        cost = cost_of(dma1d, desc, tile=None, env={"x_ptr": 0})
        self.assertAlmostEqual(cost, 64.0)


if __name__ == "__main__":
    unittest.main()
