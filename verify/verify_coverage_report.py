#!/usr/bin/env python3
"""Verification script for report/coverage.py and report/transfer.py.

Asserts:
1. Coverage report computes boolean-first fully_lowered correctly.
2. Labelled upper bound is present in text output.
3. Unsupported operations inventory carries op_name, reason, and loc.
4. Transfer report generates cross-ISA comparison and stage breakdown.
5. Limitations document generates literature cross-references.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from triton_tritonflow.diagnostics import (
    diagnose_unsupported_op,
)
from triton_tritonflow.report.coverage import (
    CoverageReport,
    UnsupportedOp,
    coverage_report,
    render_markdown,
)
from triton_tritonflow.report.transfer import (
    Edit,
    TransferReport,
    limitations_document,
    render_transfer_markdown,
    transfer_report,
)


def test_coverage_reporting() -> None:
    # Build sample pipeline runs
    runs = [
        {
            "isa_name": "tritonflow1",
            "tier_name": "t0_vecadd",
            "annotated_ops": {"tt.load", "tt.store", "arith.addf"},
            "total_ops": 18,
            "latency_ms": 1.2,
            "unsupported": [],
            "selection_decisions": [
                {
                    "binding_id": "%offs",
                    "chosen": "EPI",
                    "chosen_cost": 0.5,
                    "oracle_min": 0.5,
                    "gap": 0.0,
                    "rejected": [],
                }
            ],
        },
        {
            "isa_name": "tritonflow1",
            "tier_name": "t3_modulo",
            "annotated_ops": {"tt.load", "tt.store"},
            "total_ops": 25,
            "latency_ms": 1.8,
            "unsupported": [
                UnsupportedOp(
                    op_name="arith.remsi",
                    reason="modulo indexing unmodeled (FR-025)",
                    loc="rm_3",
                )
            ],
        },
    ]

    report = coverage_report(runs)
    assert isinstance(report, CoverageReport), "Must return CoverageReport instance"
    assert len(report.tiers) == 2, f"Expected 2 tiers, got {len(report.tiers)}"

    t0 = report.get_tier("tritonflow1", "t0_vecadd")
    assert t0 is not None, "Tier t0_vecadd missing"
    assert t0.fully_lowered is True, "t0_vecadd must be fully_lowered"
    assert t0.annotated_node_fraction > 0, "t0_vecadd fraction must be > 0"
    assert len(t0.selection_quality) == 1, "t0 selection quality missing"

    t3 = report.get_tier("tritonflow1", "t3_modulo")
    assert t3 is not None, "Tier t3_modulo missing"
    assert t3.fully_lowered is False, "t3_modulo must be fully_lowered=False"
    assert len(t3.unsupported) == 1, "t3_modulo must have 1 unsupported op"
    assert t3.unsupported[0].loc == "rm_3", f"Expected loc rm_3, got {t3.unsupported[0].loc}"

    md = render_markdown(report)
    assert "| tritonflow1 | t0_vecadd | yes |" in md, "Expected t0_vecadd yes row in markdown"
    assert "| tritonflow1 | t3_modulo | NO |" in md, "Expected t3_modulo NO row in markdown"
    assert "upper bound" in md.lower() or "≤" in md, "Must label annotated as upper bound"
    print("  ✔ Coverage report data structures and markdown verified")


def test_transfer_reporting() -> None:
    runs = [
        {"isa_name": "tritonflow1", "tier_name": "t0_vecadd", "total_cost": 9217.5},
        {"isa_name": "tritonflow2", "tier_name": "t0_vecadd", "total_cost": 7374.0},
    ]
    edits = [
        Edit(path="src/triton_tritonflow/isa/schemas/tritonflow2.yaml", lines_changed=96, reason="Target schema definition"),
        Edit(path="src/triton_tritonflow/isa/rules/tritonflow2.py", lines_changed=48, reason="Target rules definition"),
    ]
    treport = transfer_report(runs, edits=edits, baseline_isa="tritonflow1", target_isa="tritonflow2")
    assert isinstance(treport, TransferReport), "Must return TransferReport"
    assert len(treport.stages) == 7, f"Expected 7 pipeline stages, got {len(treport.stages)}"
    assert treport.cost_comparison["t0_vecadd"] == (9217.5, 7374.0)

    tmd = render_transfer_markdown(treport)
    assert "Cross-ISA Transfer Report" in tmd
    assert "tritonflow1" in tmd
    assert "tritonflow2" in tmd
    print("  ✔ Transfer report stages, cost deltas, and markdown verified")


def test_limitations_and_diagnostics() -> None:
    non_goals = [
        {
            "goal": "Modulo wraparound indexing",
            "reference": "Presburger / polyhedral integer set library",
            "state": "Delegated to eager fallback per FR-025",
        }
    ]
    doc = limitations_document(non_goals)
    assert "Modulo wraparound" in doc
    assert "Presburger" in doc

    diag = diagnose_unsupported_op("arith.remsi", loc="offs", line=10)
    rendered = diag.render(use_color=False)
    assert "arith.remsi" in rendered
    assert "loc(offs)" in rendered
    assert "hint:" in rendered
    print("  ✔ Limitations cross-referencing and diagnostics verified")


def main() -> int:
    print("=== Verifying Report Generation (Coverage, Transfer, Diagnostics) ===")
    test_coverage_reporting()
    test_transfer_reporting()
    test_limitations_and_diagnostics()
    print("All report and diagnostics checks passed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
