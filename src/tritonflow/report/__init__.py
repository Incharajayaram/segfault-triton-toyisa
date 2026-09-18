"""report: the artifact.

Coverage (boolean before fraction), selection quality against the oracle, and
the cross-ISA transfer table. Every number here is generated from data
(FR-027..FR-031).
"""

from .coverage import (
    CoverageReport,
    SelectionQuality,
    TierCoverage,
    UnsupportedOp,
    coverage_report,
    render_markdown,
)
from .transfer import (
    Edit,
    StageTransfer,
    TransferReport,
    limitations_document,
    render_transfer_markdown,
    transfer_report,
)

__all__ = [
    "CoverageReport",
    "Edit",
    "SelectionQuality",
    "StageTransfer",
    "TierCoverage",
    "TransferReport",
    "UnsupportedOp",
    "coverage_report",
    "limitations_document",
    "render_markdown",
    "render_transfer_markdown",
    "transfer_report",
]
