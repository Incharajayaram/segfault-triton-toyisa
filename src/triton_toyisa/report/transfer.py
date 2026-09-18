"""Cross-ISA transfer reporting and limitations documentation.

Implements contracts/coverage-report.md postconditions 6 and 8 (FR-030, FR-031).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

__all__ = [
    "Edit",
    "StageTransfer",
    "TransferReport",
    "transfer_report",
    "render_transfer_markdown",
    "limitations_document",
]

# Standard compiler pipeline stages declared in plan.md
PIPELINE_STAGES = (
    "frontend_parser",
    "def_use_graph",
    "access_descriptor",
    "hardware_models",
    "instruction_selector",
    "code_emitter",
    "machine_emulator",
)


@dataclass(frozen=True)
class Edit:
    """Code change required to enable target ISA."""

    path: str
    lines_changed: int
    reason: str


@dataclass(frozen=True)
class StageTransfer:
    """Status of an individual pipeline stage for a target ISA."""

    stage_name: str
    isa_name: str
    transferred: bool
    edit_location: str | None = None
    notes: str = ""


@dataclass
class TransferReport:
    """The cross-ISA transferability evaluation report (FR-030)."""

    target_isa: str
    baseline_isa: str = "toyisa1"
    stages: list[StageTransfer] = field(default_factory=list)
    edits_outside_schema_and_rules: list[Edit] = field(default_factory=list)
    cost_comparison: dict[str, tuple[float, float]] = field(default_factory=dict)
    generation_timestamp: str = ""
    zero_edit_transfer: bool = False

    def is_fully_transferred(self) -> bool:
        """True if every declared pipeline stage transferred successfully."""
        return bool(self.stages) and all(s.transferred for s in self.stages)


def transfer_report(
    runs: list[Any],
    edits: list[Edit] | None = None,
    baseline_isa: str = "toyisa1",
    target_isa: str = "toyisa2",
) -> TransferReport:
    """Generate transfer report for an independently-authored ISA (FR-030).

    Args:
        runs: Execution runs containing both baseline_isa and target_isa runs.
        edits: Any edits made outside isa/schemas/ and isa/rules/.
        baseline_isa: Initial ISA name (default "toyisa1").
        target_isa: Transferred ISA name (e.g. "toyisa2" or "vortex_rvgpu").

    Returns:
        TransferReport with per-stage transfer outcomes and cost comparisons.
    """
    edits = edits or []
    stages: list[StageTransfer] = []

    # Map available runs by (isa, tier)
    run_map: dict[tuple[str, str], Any] = {}
    for r in runs:
        if isinstance(r, dict):
            run_map[(r.get("isa_name", ""), r.get("tier_name", ""))] = r
        else:
            run_map[(getattr(r, "schema", getattr(r, "isa_name", "")), getattr(r, "tier", getattr(r, "tier_name", "")))] = r

    # Determine stage transferability
    for stage in PIPELINE_STAGES:
        # Check if edits affected this stage
        relevant_edits = [e for e in edits if stage in e.path or stage in e.reason.lower()]
        transferred = len(relevant_edits) == 0
        edit_loc = relevant_edits[0].path if relevant_edits else None
        notes = relevant_edits[0].reason if relevant_edits else "Zero modifications required; schema-driven."

        stages.append(
            StageTransfer(
                stage_name=stage,
                isa_name=target_isa,
                transferred=transferred,
                edit_location=edit_loc,
                notes=notes,
            )
        )

    # Cost comparison across common tiers
    cost_comparison: dict[str, tuple[float, float]] = {}
    common_tiers = {"t0_vecadd", "t1_matmul", "t2_matmul_relu", "t3_modulo"}

    for tier in sorted(common_tiers):
        base_run = run_map.get((baseline_isa, tier))
        target_run = run_map.get((target_isa, tier))

        def _cost(r: Any) -> float:
            if r is None:
                return 0.0
            if isinstance(r, dict):
                return float(r.get("total_cost", 0.0))
            return float(getattr(r, "total_cost", 0.0))

        if base_run is not None or target_run is not None:
            cost_comparison[tier] = (_cost(base_run), _cost(target_run))

    zero_edit = len(edits) == 0

    return TransferReport(
        target_isa=target_isa,
        baseline_isa=baseline_isa,
        stages=stages,
        edits_outside_schema_and_rules=edits,
        cost_comparison=cost_comparison,
        generation_timestamp=datetime.now().isoformat(),
        zero_edit_transfer=zero_edit,
    )


def render_transfer_markdown(report: TransferReport) -> str:
    """Render transfer report as markdown (FR-030)."""
    lines = [
        f"# Cross-ISA Transfer Report: `{report.baseline_isa}` → `{report.target_isa}`",
        "",
        f"**Generated**: {report.generation_timestamp or datetime.now().isoformat()}  ",
        f"**Zero-Edit Status**: {'✅ Verified (0 edits outside schema/rules)' if report.zero_edit_transfer else '⚠️ Requires External Edits'}",
        "",
        "## Pipeline Stage Transferability",
        "",
        "| Pipeline Stage | Target ISA | Transferred | Edit Location | Notes |",
        "|---|---|---|---|---|",
    ]

    for stage in report.stages:
        trans_str = "✅ YES" if stage.transferred else "❌ NO"
        loc_str = f"`{stage.edit_location}`" if stage.edit_location else "–"
        lines.append(
            f"| `{stage.stage_name}` | `{stage.isa_name}` | {trans_str} | {loc_str} | {stage.notes} |"
        )

    lines.extend([
        "",
        "## Cost Comparison vs Baseline",
        "",
        f"| Tier | `{report.baseline_isa}` Cost | `{report.target_isa}` Cost | Delta (%) |",
        "|---|---|---|---|",
    ])

    for tier, (base_cost, target_cost) in report.cost_comparison.items():
        if base_cost > 0:
            delta_pct = ((target_cost - base_cost) / base_cost) * 100
            delta_str = f"{delta_pct:+.1f}%"
        else:
            delta_str = "N/A"
        lines.append(
            f"| `{tier}` | {base_cost:.1f} | {target_cost:.1f} | {delta_str} |"
        )

    if report.edits_outside_schema_and_rules:
        lines.extend([
            "",
            "## Edits Outside Schemas and Rules",
            "",
        ])
        for edit in report.edits_outside_schema_and_rules:
            lines.append(f"- `{edit.path}` ({edit.lines_changed} lines): {edit.reason}")

    return "\n".join(lines)


def limitations_document(non_goals: list[dict[str, str]]) -> str:
    """Generate structured limitations document cross-referencing literature (FR-031).

    Every non-goal appears with the source work that solves it at full generality,
    or an explicit 'no known implementation' — never a bare 'future work'.
    """
    lines = [
        "# Architectural Limitations & Research Non-Goals",
        "",
        "| Non-Goal / Limitation | General Solution / Literature Reference | State in this Prototype |",
        "|---|---|---|",
    ]

    for item in non_goals:
        goal = item.get("goal", "Unspecified limitation")
        ref = item.get("reference", "no known implementation")
        state = item.get("state", "Excluded by contract")
        lines.append(f"| {goal} | {ref} | {state} |")

    lines.extend([
        "",
        "> [!NOTE]",
        "> Per project constitution (FR-031), limitations are acknowledged with citations",
        "> rather than deferred as ungrounded 'future work'.",
    ])

    return "\n".join(lines)
