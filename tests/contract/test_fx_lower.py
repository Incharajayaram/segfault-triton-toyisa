"""Contract tests for the live FX-graph lowering proof of concept.

The property under test is not "it produces instructions" but "the *schema*
produced them": the same graph lowered against three ISAs must select three
different instruction sets at three different costs, with no branch on the ISA
name anywhere in `fx_lower`.
"""

from __future__ import annotations

import unittest

try:
    import torch
    from torch.fx import symbolic_trace

    HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is an optional extra
    HAS_TORCH = False

if HAS_TORCH:
    from tritonflow.torch_backend.fx_lower import (
        lower_fx_graph,
        supported_targets,
        try_lower_and_run,
    )

ISAS = ("tritonflow1", "tritonflow2", "vortex_rvgpu")


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class TestFxLowering(unittest.TestCase):
    def _trace(self, fn, shape=(8, 8)):
        return symbolic_trace(fn), [torch.randn(*shape), torch.randn(*shape)]

    def test_elementwise_matches_eager_on_every_isa(self) -> None:
        def fn(a, b):
            return torch.relu(a + b)

        gm, xs = self._trace(fn)
        for isa in ISAS:
            with self.subTest(isa=isa):
                result, lowering = try_lower_and_run(gm, xs, xs, isa_name=isa)
                self.assertIsNotNone(result, f"{isa} did not lower")
                self.assertTrue(lowering.fully_lowered)
                expected = fn(*xs).numpy()
                self.assertAlmostEqual(
                    float(abs(result - expected).max()), 0.0, places=6
                )

    def test_matmul_lowers_and_matches(self) -> None:
        def fn(a, b):
            return torch.mm(a, b)

        gm, xs = self._trace(fn, (16, 16))
        result, lowering = try_lower_and_run(gm, xs, xs, isa_name="tritonflow1")
        self.assertIsNotNone(result)
        self.assertIn("MAC", " ".join(i.name for i in lowering.program.instructions()))
        self.assertLess(float(abs(result - fn(*xs).numpy()).max()), 1e-4)

    def test_selection_differs_across_isas(self) -> None:
        """Transfer is a re-selection, not a rename.

        If two ISAs produced the same instruction names at the same cost, the
        schema would not be driving anything.
        """

        def fn(a, b):
            return torch.relu(a + b)

        gm, xs = self._trace(fn)
        seen = set()
        for isa in ISAS:
            lowering = lower_fx_graph(gm, xs, isa_name=isa)
            self.assertIsNotNone(lowering, isa)
            seen.add(
                (
                    frozenset(i.name for i in lowering.program.instructions()),
                    round(lowering.program.total_cost, 3),
                )
            )
        self.assertEqual(len(seen), len(ISAS), seen)

    def test_relu_does_not_lower_to_an_adder(self) -> None:
        """Vortex's elementwise units share a rule and a cost.

        Before the schema declared `op:` on each, minimum-cost selection broke
        the tie by declaration order and chose VADD for relu; the emulator
        dispatches those by instruction name, so it computed an addition and
        returned a wrong answer with no diagnostic.
        """

        def fn(a, b):
            return torch.relu(a + b)

        gm, xs = self._trace(fn)
        lowering = lower_fx_graph(gm, xs, isa_name="vortex_rvgpu")
        names = [i.name for i in lowering.program.instructions()]
        self.assertIn("VRELU", names)
        self.assertEqual(names.count("VADD"), 1, names)

    def test_unsupported_op_refuses_with_a_reason(self) -> None:
        def fn(a, b):
            return torch.sigmoid(a + b)

        gm, xs = self._trace(fn)
        reasons: list[str] = []
        self.assertIsNone(lower_fx_graph(gm, xs, isa_name="tritonflow1", report=reasons))
        self.assertTrue(reasons)
        self.assertIn("sigmoid", reasons[0])

    def test_op_the_isa_does_not_declare_refuses(self) -> None:
        """tritonflow1's EPI declares `op: [add, relu]`, so a multiply has no lowering."""

        def fn(a, b):
            return a * b

        gm, xs = self._trace(fn)
        reasons: list[str] = []
        self.assertIsNone(lower_fx_graph(gm, xs, isa_name="tritonflow1", report=reasons))
        self.assertIn("no admissible", reasons[0])

    def test_supported_targets_is_declared(self) -> None:
        self.assertIn("relu", supported_targets())
        self.assertIn("mm", supported_targets())


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class TestBackendIntegration(unittest.TestCase):
    def test_torch_compile_uses_the_fx_path(self) -> None:
        from tritonflow.torch_backend.compiler import tritonflow_backend

        def fn(a, b):
            return torch.relu(a + b)

        gm = symbolic_trace(fn)
        xs = [torch.randn(8, 8), torch.randn(8, 8)]
        run = tritonflow_backend(gm, xs)
        self.assertTrue(hasattr(run, "tritonflow_fx"), "the FX path was not taken")
        out = run(*xs)
        out = out[0] if isinstance(out, (list, tuple)) else out
        self.assertLess(float((torch.as_tensor(out) - fn(*xs)).abs().max()), 1e-6)

    def test_unsupported_graph_falls_back_and_records_why(self) -> None:
        from tritonflow.torch_backend.compiler import tritonflow_backend

        def fn(a, b):
            return torch.sigmoid(a + b)

        gm = symbolic_trace(fn)
        xs = [torch.randn(8, 8), torch.randn(8, 8)]
        run = tritonflow_backend(gm, xs)
        self.assertFalse(hasattr(run, "tritonflow_fx"))
        self.assertTrue(run.tritonflow_plan.fallbacks)
        self.assertIn("sigmoid", run.tritonflow_plan.fallbacks[-1].reason)


if __name__ == "__main__":
    unittest.main()
