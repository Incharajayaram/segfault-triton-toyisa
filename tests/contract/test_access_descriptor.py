"""Contract tests for recognize.descriptor and recognize.walk using standard library unittest.

Validates the symbolic expression algebra (SymExpr), AccessDescriptor properties,
canonical descriptor keys, and bounded traversal accounting.
"""

from __future__ import annotations

import unittest

from triton_toyisa.recognize.descriptor import (
    AccessDescriptor,
    BudgetExhausted,
    Ok,
    Unstructured,
)
from triton_toyisa.recognize.walk import (
    MAX_HOPS,
    BoundedWalker,
    BudgetReached,
    SymExpr,
)


class TestAccessDescriptorContract(unittest.TestCase):
    """Real contract tests for access descriptors and symbolic expressions."""

    def test_sym_expr_algebra_and_normalization(self) -> None:
        """SymExpr must normalize terms on construction so equal expressions compare equal."""
        a = SymExpr.symbol("sam")
        two_a = a + a
        scaled_a = SymExpr.const(2) * a
        self.assertEqual(two_a, scaled_a)
        self.assertFalse(a.is_constant)
        self.assertIsNone(a.as_int())

        c4 = SymExpr.const(4)
        c8 = SymExpr.const(8)
        self.assertTrue((c4 + c8).is_constant)
        self.assertEqual((c4 + c8).as_int(), 12)

    def test_sym_expr_multivariate_product(self) -> None:
        """Product of two symbols must combine symbol names in sorted order."""
        sam = SymExpr.symbol("sam")
        sak = SymExpr.symbol("sak")
        prod = sam * sak
        self.assertEqual(prod.symbols(), frozenset(["sam", "sak"]))
        # Commutativity
        self.assertEqual(sam * sak, sak * sam)

    def test_descriptor_is_contiguous_predicate(self) -> None:
        """Unit stride on the last dimension with zero wraparound is contiguous."""
        contiguous = AccessDescriptor(
            base="x_ptr",
            sizes=(1024,),
            strides=(1,),
            offsets=(0,),
            shape=(0,),
            order=(0,),
            dtype="f32",
            loop_carried=False,
            increment=None,
        )
        self.assertTrue(contiguous.is_contiguous)

        # Non-unit stride
        strided = AccessDescriptor(
            base="x_ptr",
            sizes=(1024,),
            strides=(4,),
            offsets=(0,),
            shape=(0,),
            order=(0,),
            dtype="f32",
            loop_carried=False,
            increment=None,
        )
        self.assertFalse(strided.is_contiguous)

        # Modulo wraparound (shape != 0)
        wrapped = AccessDescriptor(
            base="x_ptr",
            sizes=(1024,),
            strides=(1,),
            offsets=(0,),
            shape=(64,),
            order=(0,),
            dtype="f32",
            loop_carried=False,
            increment=None,
        )
        self.assertFalse(wrapped.is_contiguous)

    def test_descriptor_key_stability(self) -> None:
        """descriptor_key() must produce byte-stable canonical text across runs."""
        desc = AccessDescriptor(
            base="a_ptr",
            sizes=(64, 32),
            strides=(32, 1),
            offsets=(0, 0),
            shape=(0, 0),
            order=(0, 1),
            dtype="f32",
            loop_carried=True,
            increment=32,
        )
        key = desc.descriptor_key()
        self.assertIn("base=a_ptr", key)
        self.assertIn("sizes=[64, 32]", key)
        self.assertIn("strides=[32, 1]", key)
        self.assertIn("loop_carried=True", key)
        self.assertIn("increment=32", key)

    def test_descriptor_result_union(self) -> None:
        """DescriptorResult is a closed union of Ok, Unstructured, BudgetExhausted."""
        desc = AccessDescriptor(
            base="x",
            sizes=(64,),
            strides=(1,),
            offsets=(0,),
            shape=(0,),
            order=(0,),
            dtype="f32",
            loop_carried=False,
            increment=None,
        )
        ok_res = Ok(desc)
        self.assertEqual(ok_res.descriptor, desc)

        unstructured_res = Unstructured(reason="non-affine index: tt.gather", ops=("tt.gather",))
        self.assertIn("non-affine", unstructured_res.reason)

        budget_res = BudgetExhausted(hops=MAX_HOPS, limit=MAX_HOPS)
        self.assertEqual(budget_res.hops, 32)

    def test_bounded_walker_budget(self) -> None:
        """BoundedWalker must raise BudgetReached when limit is exhausted."""
        from triton_toyisa.ttir.ssa import Operation
        walker = BoundedWalker(limit=5)
        for i in range(5):
            walker.visit(Operation(name=f"op_{i}"))
        self.assertEqual(walker.hops, 5)
        with self.assertRaises(BudgetReached):
            walker.visit(Operation(name="op_overflow"))


if __name__ == "__main__":
    unittest.main()
