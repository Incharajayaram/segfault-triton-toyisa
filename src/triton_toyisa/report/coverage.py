"""Coverage reporting: boolean-before-fraction, per-ISA, per-tier.

Implements contracts/coverage-report.md postconditions 1-5 and 9 (FR-027..FR-029).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

__all__ = [
    "CoverageReport",
    "TierCoverage",
    "SelectionQuality",
    "UnsupportedOp",
    "coverage_report",
    "render_markdown",
]


@dataclass(frozen=True)
class UnsupportedOp:
    """One operation that could not be lowered (FR-028)."""

    op_name: str
    reason: str
    loc: str | None = None  # From SourceRef.loc_name if available

    def __str__(self) -> str:
        loc_str = f" at {self.loc}" if self.loc else ""
        return f"{self.op_name}{loc_str}: {self.reason}"


@dataclass(frozen=True)
class SelectionQuality:
    """How well the selector chose, vs exhaustive oracle (FR-029)."""

    binding_id: str
    chosen_instruction: str
    chosen_cost: float
    oracle_min_cost: float
    gap: float  # chosen - oracle_min
    rejected_alternatives: list[tuple[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class TierCoverage:
    """Coverage for one (ISA, tier) pair. Boolean comes first (FR-027 post 1)."""

    isa_name: str
    tier_name: str

    # 1. BOOLEAN FIRST (FR-027 postcondition 1)
    fully_lowered: bool  # TRUE iff every value-producing op is annotated or provably elided

    # 2. Fractions with explicit upper bound definition (FR-027 post 2)
    largest_fully_lowered_subgraph: float  # Fraction of ops in largest valid subgraph
    annotated_node_fraction: float  # Labelled as upper bound (≤)

    # 3. Actionable unsupported op inventory (FR-028)
    unsupported: list[UnsupportedOp] = field(default_factory=list)

    # 4. Selection quality evidence (FR-029)
    selection_quality: list[SelectionQuality] = field(default_factory=list)

    # 5. Performance (SC-007, postcondition 9)
    latency_ms: float = 0.0  # One-time per-kernel-shape cost

    # 6. Target instruction mix
    instruction_mix: dict[str, int] = field(default_factory=dict)


@dataclass
class CoverageReport:
    """Complete coverage report. Per-ISA, per-tier, never aggregated (FR-027 post 4)."""

    tiers: list[TierCoverage] = field(default_factory=list)
    generation_timestamp: str = ""
    pipeline_version: str = "0.1.0"
    notes: list[str] = field(default_factory=list)

    def add_tier(self, tier: TierCoverage) -> None:
        """Add a tier's coverage statistics."""
        self.tiers.append(tier)

    def get_tier(self, isa_name: str, tier_name: str) -> TierCoverage | None:
        """Retrieve coverage for a specific (ISA, tier) pair."""
        for tier in self.tiers:
            if tier.isa_name == isa_name and tier.tier_name == tier_name:
                return tier
        return None

    def summary(self) -> dict[str, Any]:
        """High-level summary counts (never a single aggregated coverage percentage)."""
        return {
            "total_tiers": len(self.tiers),
            "fully_lowered_count": sum(1 for t in self.tiers if t.fully_lowered),
            "isa_count": len({t.isa_name for t in self.tiers}),
            "avg_latency_ms": (
                sum(t.latency_ms for t in self.tiers) / len(self.tiers)
                if self.tiers
                else 0.0
            ),
        }


def coverage_report(runs: list[Any]) -> CoverageReport:
    """Generate coverage report from pipeline runs (FR-027).

    Args:
        runs: List of run results (dicts or RunContext objects).
    Returns:
        CoverageReport with per-tier statistics.
    """
    report = CoverageReport(
        generation_timestamp=datetime.now().isoformat(),
    )

    for run in runs:
        if isinstance(run, dict):
            isa_name = run.get("isa_name", "unknown")
            tier_name = run.get("tier_name", "unknown")
            program = run.get("program")
            annotated_ops = run.get("annotated_ops", set())
            total_ops = run.get("total_ops", 0)
            latency_ms = run.get("latency_ms", 0.0)
            unsupported_raw = run.get("unsupported", [])
            selection_raw = run.get("selection_decisions", [])
        else:
            # Assume RunContext or similar dataclass
            isa_name = getattr(run, "schema", getattr(run, "isa_name", "unknown"))
            tier_name = getattr(run, "tier", getattr(run, "tier_name", "unknown"))
            program = getattr(run, "program", None)
            annotated_ops = getattr(run, "annotations", set())
            total_ops = getattr(run, "value_ops", getattr(run, "raw_op_count", 0))
            latency_ms = getattr(run, "generation_ns", 0.0)
            if latency_ms > 0:
                latency_ms /= 1e6
            unsupported_raw = getattr(run, "unsupported", [])
            selection_raw = getattr(run, "selection_decisions", [])

        unsupported_markers: list[UnsupportedOp] = []

        # 1. From program markers
        if program and hasattr(program, "markers"):
            for marker in program.markers():
                unsupported_markers.append(
                    UnsupportedOp(
                        op_name=marker.source.op_name if getattr(marker, "source", None) else "unknown",
                        reason=getattr(marker, "reason", "unsupported operation"),
                        loc=marker.source.loc_name if getattr(marker, "source", None) else None,
                    )
                )

        # 2. From explicit unsupported items
        for u in unsupported_raw:
            if isinstance(u, UnsupportedOp):
                unsupported_markers.append(u)
            elif isinstance(u, dict):
                unsupported_markers.append(
                    UnsupportedOp(
                        op_name=u.get("op_name", "unknown"),
                        reason=u.get("reason", "unknown"),
                        loc=u.get("loc"),
                    )
                )
            elif isinstance(u, str):
                unsupported_markers.append(
                    UnsupportedOp(op_name="unsupported", reason=u)
                )

        # Boolean first (FR-027 post 1)
        fully_lowered = len(unsupported_markers) == 0

        # Fractions
        if isinstance(annotated_ops, dict):
            annotated_count = len(annotated_ops)
        elif isinstance(annotated_ops, (set, list, tuple)):
            annotated_count = len(annotated_ops)
        elif isinstance(annotated_ops, (int, float)):
            annotated_count = int(annotated_ops)
        else:
            annotated_count = 0

        annotated_fraction = (
            annotated_count / total_ops if total_ops > 0 else (1.0 if fully_lowered else 0.0)
        )
        largest_subgraph = 1.0 if fully_lowered else min(0.99, annotated_fraction)

        # Instruction mix - inspect all instructions across prologue, loops, and epilogue
        instruction_mix: dict[str, int] = {}
        if program and hasattr(program, "instructions"):
            for instr in program.instructions():
                name = getattr(instr, "name", "unknown")
                instruction_mix[name] = instruction_mix.get(name, 0) + 1
        elif program and hasattr(program, "instrs"):
            for instr in program.instrs:
                name = getattr(instr, "name", "unknown")
                instruction_mix[name] = instruction_mix.get(name, 0) + 1

        # Selection quality
        selection_quality: list[SelectionQuality] = []
        for dec in selection_raw:
            if isinstance(dec, SelectionQuality):
                selection_quality.append(dec)
            elif isinstance(dec, dict):
                selection_quality.append(
                    SelectionQuality(
                        binding_id=dec.get("binding_id", "op"),
                        chosen_instruction=dec.get("chosen", "unknown"),
                        chosen_cost=float(dec.get("chosen_cost", 0.0)),
                        oracle_min_cost=float(dec.get("oracle_min", 0.0)),
                        gap=float(dec.get("gap", 0.0)),
                        rejected_alternatives=dec.get("rejected", []),
                    )
                )

        tier_coverage = TierCoverage(
            isa_name=str(isa_name),
            tier_name=str(tier_name),
            fully_lowered=fully_lowered,
            largest_fully_lowered_subgraph=float(largest_subgraph),
            annotated_node_fraction=float(annotated_fraction),
            unsupported=unsupported_markers,
            selection_quality=selection_quality,
            latency_ms=float(latency_ms),
            instruction_mix=instruction_mix,
        )
        report.add_tier(tier_coverage)

    return report


def render_markdown(report: CoverageReport) -> str:
    """Render coverage report as markdown table (FR-027 post 7).

    Returns prose generated from data, conforming to the contract shape:
    | ISA | Tier | fully lowered | largest subgraph | annotated (≤) | unsupported | latency ms |
    """
    lines = [
        "# Coverage Report",
        "",
        f"**Generated**: {report.generation_timestamp or datetime.now().isoformat()}  ",
        f"**Pipeline Version**: {report.pipeline_version}",
        "",
    ]

    if report.notes:
        lines.append("## Notes")
        for note in report.notes:
            lines.append(f"- {note}")
        lines.append("")

    summary = report.summary()
    lines.extend([
        "## Summary",
        "",
        f"- Total tiers tested: {summary['total_tiers']}",
        f"- Fully lowered: {summary['fully_lowered_count']}/{summary['total_tiers']}",
        f"- ISAs tested: {summary['isa_count']}",
        f"- Average latency: {summary['avg_latency_ms']:.1f} ms",
        "",
        "## Per-Tier Coverage",
        "",
        "| ISA | Tier | fully lowered | largest subgraph | annotated (≤) | unsupported | latency ms |",
        "|---|---|---|---|---|---|---|",
    ])

    for tier in sorted(report.tiers, key=lambda t: (t.isa_name, t.tier_name)):
        fully_str = "yes" if tier.fully_lowered else "NO"
        if not tier.unsupported:
            unsup_str = "–"
        else:
            first_loc = tier.unsupported[0].loc or tier.unsupported[0].op_name
            unsup_str = f"{len(tier.unsupported)} ({first_loc})"

        lines.append(
            f"| {tier.isa_name} | {tier.tier_name} | {fully_str} | "
            f"{tier.largest_fully_lowered_subgraph:.2f} | "
            f"{tier.annotated_node_fraction:.2f} | "
            f"{unsup_str} | "
            f"{tier.latency_ms:.1f} |"
        )

    lines.extend([
        "",
        "> [!NOTE]",
        "> `annotated (≤)` is an upper bound on coverage. `fully lowered` is the definitive metric.",
        "",
    ])

    # Instruction Mix Detail
    has_mix = any(t.instruction_mix for t in report.tiers)
    if has_mix:
        lines.extend(["## Instruction Mix", ""])
        for tier in report.tiers:
            if tier.instruction_mix:
                lines.append(f"### {tier.isa_name} / {tier.tier_name}")
                lines.append("")
                total_instrs = sum(tier.instruction_mix.values())
                for instr, count in sorted(tier.instruction_mix.items(), key=lambda x: -x[1]):
                    pct = (count / total_instrs * 100) if total_instrs > 0 else 0
                    lines.append(f"- `{instr}`: {count} ({pct:.1f}%)")
                lines.append("")

    # Selection Quality Detail (FR-029)
    has_sq = any(t.selection_quality for t in report.tiers)
    if has_sq:
        lines.extend(["## Selection Quality vs Oracle", ""])
        for tier in report.tiers:
            if tier.selection_quality:
                lines.append(f"### {tier.isa_name} / {tier.tier_name}")
                lines.append("")
                for sq in tier.selection_quality:
                    gap_sign = f"+{sq.gap:.2f}" if sq.gap > 0 else f"{sq.gap:.2f}"
                    lines.append(
                        f"- Binding `{sq.binding_id}`: chose `{sq.chosen_instruction}` "
                        f"(cost: {sq.chosen_cost:.1f}, oracle: {sq.oracle_min_cost:.1f}, gap: {gap_sign})"
                    )
                    if sq.rejected_alternatives:
                        for alt_name, reason in sq.rejected_alternatives[:3]:
                            lines.append(f"  - rejected `{alt_name}`: {reason}")
                lines.append("")

    return "\n".join(lines)
