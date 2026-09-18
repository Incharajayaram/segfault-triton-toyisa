"""The one place bench/ touches the pipeline.

`bench/run.py` never imports project modules directly: it asks this adapter for
a `RunContext`. That keeps the protocol in one file and lets every track land
its part without editing the runner.

Track D owns this file. Until the pipeline exists, `lower_fixture` returns
`None`, every row is recorded as `unavailable`, and `bench/results.json` is
still a valid artifact with all rows present and a reason for each.

DETERMINISM CONTRACT (see docs/team/testing-ci.md S7):
  * inputs come from `make_inputs`, never from a bare `np.random` call
  * the seed is the one in bench/cases.yaml
  * no GPU, no network, no environment-dependent behaviour
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

SEED = 24173


@dataclass
class RunContext:
    """Everything a bench metric may read. Filled by the pipeline, not by bench."""

    tier: str
    module: Any = None          # ttir.ssa.Module
    program: Any = None         # emit.ir.Program
    annotations: dict = field(default_factory=dict)
    unsupported: list = field(default_factory=list)
    total_cost: float | None = None
    reference_cost: float | None = None
    oracle_cost: float | None = None
    value_ops: int = 0
    annotated_value_ops: int = 0
    largest_subgraph_ops: int = 0
    emitted_instructions: int = 0
    raw_op_count: int = 0
    emu_outputs: dict | None = None
    reference_outputs: dict | None = None
    tolerance: float | None = None
    schema: str | None = None


def make_inputs(tier: str, seed: int = SEED) -> dict[str, np.ndarray]:
    """Deterministic inputs per tier. Shapes match the frozen fixtures' blocks."""
    rng = np.random.default_rng(seed)
    shapes = {
        "t0_vecadd": {"x": (1024,), "y": (1024,)},
        "t1_matmul": {"a": (64, 32), "b": (32, 64)},
        "t2_matmul_relu": {"a": (64, 32), "b": (32, 64)},
        "t3_modulo": {"x": (1024,), "y": (1024,)},
    }[tier]
    return {k: rng.standard_normal(s, dtype=np.float32) for k, s in shapes.items()}


def lower_fixture(tier: str) -> RunContext | None:
    """Run the real pipeline on one frozen fixture."""
    try:
        from triton_tritonflow.lower import lower_fixture as _lower
        return _lower(tier)
    except Exception:
        return None
