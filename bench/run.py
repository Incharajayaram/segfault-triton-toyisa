#!/usr/bin/env python3
"""T-L5 runner: emit bench/results.json for every (tier, metric) row.

Never gates a merge. Never omits a row. A metric that cannot be computed is
recorded as `unavailable` with a reason, so a missing number is visible in the
published table instead of being silently absent.

    python3 bench/run.py --out bench/results.json

Stdlib + numpy + pyyaml. No Triton, no torch, no GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(ROOT / "tools"))

from adapter import RunContext, lower_fixture  # noqa: E402
from fixtures_lib import load_golden  # noqa: E402

SCHEMA = 1


def provenance(cfg: dict) -> dict:
    return {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": __import__("os").cpu_count(),
        "seed": cfg.get("seed"),
        "gpu": False,
        "corpus_hashes": {
            k: v["sha256"][:16] for k, v in load_golden()["observations"].items()
        },
    }


def _ratio(num, den):
    if num is None or den in (None, 0):
        return None
    return num / den


METRIC_FIELDS = {
    # metric id -> (source, kind)
    "fully_lowered": "fully_lowered",
    "node_fraction": "node_fraction",
    "largest_subgraph": "largest_subgraph",
    "cost_ratio_vs_reference": "cost_ratio_vs_reference",
    "cost_ratio_vs_oracle": "cost_ratio_vs_oracle",
    "instruction_count": "instruction_count",
    "generation_ns": "generation_ns",
    "parity_max_rel_err": "parity_max_rel_err",
}


def measure(metric: str, ctx: RunContext | None, cfg: dict) -> tuple[str, object, str | None]:
    """Return (status, value, reason). Status is ok | unavailable | error."""
    if ctx is None:
        return "unavailable", None, "pipeline not implemented (see bench/adapter.py)"

    try:
        if metric == "fully_lowered":
            if ctx.value_ops and not ctx.unsupported and ctx.annotated_value_ops == ctx.value_ops:
                return "ok", True, None
            return "ok", False, None
        if metric == "node_fraction":
            return "ok", _ratio(ctx.annotated_value_ops, ctx.value_ops), None
        if metric == "largest_subgraph":
            return "ok", _ratio(ctx.largest_subgraph_ops, ctx.value_ops), None
        if metric == "cost_ratio_vs_reference":
            return "ok", _ratio(ctx.total_cost, ctx.reference_cost), None
        if metric == "cost_ratio_vs_oracle":
            return "ok", _ratio(ctx.total_cost, ctx.oracle_cost), None
        if metric == "instruction_count":
            return "ok", {"emitted": ctx.emitted_instructions, "raw_ops": ctx.raw_op_count}, None
        if metric == "generation_ns":
            return "ok", ctx.__dict__.get("generation_ns"), None
        if metric == "parity_max_rel_err":
            return "ok", ctx.__dict__.get("parity_max_rel_err"), None
    except Exception as exc:  # a bench row must never kill the run
        return "error", None, f"{type(exc).__name__}: {exc}"
    return "unavailable", None, f"unknown metric {metric}"


def timed_lower(tier: str, cfg: dict) -> tuple[RunContext | None, float | None]:
    """Run the pipeline under the fixed timing protocol."""
    warmup, repeats = int(cfg.get("warmup", 1)), int(cfg.get("repeats", 5))
    for _ in range(warmup):
        lower_fixture(tier)
    samples: list[int] = []
    ctx = None
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        ctx = lower_fixture(tier)
        samples.append(time.perf_counter_ns() - t0)
    if ctx is None:
        return None, None
    return ctx, float(statistics.median(samples))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "bench" / "results.json"))
    ap.add_argument("--cases", default=str(ROOT / "bench" / "cases.yaml"))
    ap.add_argument("--only", default=None, help="substring filter on case id")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.cases).read_text())
    if cfg.get("allow_gpu"):
        sys.exit("bench must not run on a GPU (docs/team/testing-ci.md S11)")

    gated = (cfg.get("seed"), cfg.get("repeats"), cfg.get("warmup"))
    expected = (24173, 5, 1)
    if gated != expected:
        sys.exit(f"protocol drift: (seed, repeats, warmup)={gated}, expected {expected}")

    results = []
    contexts: dict[str, object] = {}

    for tier in cfg["tiers"]:
        if tier not in contexts:
            ctx = None
            gen_ns = None
            if any(m["id"] == "generation_ns" for m in cfg["metrics"]):
                ctx, gen_ns = timed_lower(tier, cfg)
            else:  # pragma: no cover - only if the metric list is edited
                ctx = lower_fixture(tier)
            if ctx is not None:
                ctx.__dict__["generation_ns"] = gen_ns
            contexts[tier] = ctx

        for metric in cfg["metrics"]:
            cid = f"{tier}.{metric['id']}"
            if args.only and args.only not in cid:
                continue
            status, value, reason = measure(metric["id"], contexts[tier], cfg)
            results.append(
                {
                    "id": cid,
                    "tier": tier,
                    "metric": metric["id"],
                    "unit": metric.get("unit"),
                    "headline": bool(metric.get("headline")),
                    "upper_bound": bool(metric.get("upper_bound")),
                    "formula": metric.get("formula"),
                    "status": status,
                    "value": value,
                    "reason": reason,
                }
            )

    payload = {
        "schema": SCHEMA,
        "provenance": provenance(cfg),
        "protocol": {
            "seed": cfg["seed"],
            "repeats": cfg["repeats"],
            "warmup": cfg["warmup"],
            "clock": cfg.get("clock"),
            "statistic": cfg.get("statistic"),
            "gpu": False,
            "config_sha256": hashlib.sha256(Path(args.cases).read_bytes()).hexdigest()[:16],
        },
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"bench: {len(results)} rows -> {out}  ({ok} ok, {len(results) - ok} unavailable/error)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
