"""Contract tests for report.coverage and report.transfer using standard library unittest.

Validates the reporting contract specified in coverage-report.md:
boolean fully_lowered, upper-bound labelled fractions, and unsupported op inventory.
"""

from __future__ import annotations

import dataclasses
import unittest

from tritonflow.report.coverage import (
    CoverageReport,
    TierCoverage,
    UnsupportedOp,
    coverage_report,
    render_markdown,
)
from tritonflow.report.transfer import (
    Edit,
    TransferReport,
    limitations_document,
    render_transfer_markdown,
    transfer_report,
)


class TestCoverageReportContract(unittest.TestCase):
    """Contract tests for coverage and transfer reports (FR-027..FR-031)."""

    def test_report_module_contract(self) -> None:
        """Report package must export coverage_report and transfer_report."""
        self.assertTrue(callable(coverage_report))
        self.assertTrue(callable(transfer_report))

    def test_boolean_comes_first(self) -> None:
        """FR-027 postcondition 1: fully_lowered boolean appears before fractions."""
        fields = [f.name for f in dataclasses.fields(TierCoverage)]
        fully_lowered_idx = fields.index("fully_lowered")
        fraction_idx = fields.index("annotated_node_fraction")
        self.assertLess(
            fully_lowered_idx,
            fraction_idx,
            "fully_lowered must appear before annotated_node_fraction in TierCoverage definition",
        )

    def test_fraction_labelled_upper_bound(self) -> None:
        """FR-027 postcondition 2: fraction is explicitly labelled as upper bound."""
        report = CoverageReport()
        markdown = render_markdown(report)
        self.assertTrue(
            "upper bound" in markdown.lower() or "≤" in markdown,
            "Rendered markdown must explicitly label annotated fraction as upper bound",
        )

    def test_unsupported_inventory_actionable(self) -> None:
        """FR-028: unsupported operations carry op_name, reason, and loc."""
        unsup = UnsupportedOp(op_name="arith.remsi", reason="modulo wraparound unmodeled", loc="x_ptr")
        self.assertEqual(unsup.op_name, "arith.remsi")
        self.assertEqual(unsup.reason, "modulo wraparound unmodeled")
        self.assertEqual(unsup.loc, "x_ptr")
        self.assertIn("arith.remsi at x_ptr", str(unsup))

    def test_per_tier_never_aggregated(self) -> None:
        """FR-027 postcondition 4: reports are strictly per-ISA and per-tier."""
        fields = {f.name for f in dataclasses.fields(TierCoverage)}
        self.assertIn("isa_name", fields)
        self.assertIn("tier_name", fields)

    def test_transfer_report_contract(self) -> None:
        """FR-030: transfer report tracks stages and edits outside schema and rules."""
        runs = [
            {"isa_name": "tritonflow1", "tier_name": "t0_vecadd", "total_cost": 9217.5},
            {"isa_name": "tritonflow2", "tier_name": "t0_vecadd", "total_cost": 7374.0},
        ]
        edits = [Edit(path="isa/schemas/tritonflow2.yaml", lines_changed=96, reason="New target ISA")]
        treport = transfer_report(runs, edits=edits, baseline_isa="tritonflow1", target_isa="tritonflow2")
        self.assertIsInstance(treport, TransferReport)
        self.assertTrue(len(treport.stages) > 0)
        self.assertEqual(treport.cost_comparison["t0_vecadd"], (9217.5, 7374.0))
        md = render_transfer_markdown(treport)
        self.assertIn("tritonflow1", md)
        self.assertIn("tritonflow2", md)

    def test_limitations_document(self) -> None:
        """FR-031: non-goals carry general solutions/citations or explicit no known implementation."""
        non_goals = [
            {
                "goal": "Non-affine modulo wraparound",
                "reference": "Polyhedral model (ISL / Presburger arithmetic)",
                "state": "Honest refusal; delegated to PyTorch eager fallback",
            },
            {
                "goal": "Arbitrary indirect branch targets",
                "reference": "no known implementation",
                "state": "Excluded by structured CFG contract",
            },
        ]
        doc = limitations_document(non_goals)
        self.assertIn("Non-affine modulo wraparound", doc)
        self.assertIn("no known implementation", doc)


if __name__ == "__main__":
    unittest.main()
