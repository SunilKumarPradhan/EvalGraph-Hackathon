"""Per-attachment aggregation: raw sample values → one value, a status from the catalog thresholds, worst samples."""

from __future__ import annotations

import math
import statistics

from ..core.schemas import EvaluatorSpec, NodeEvalResult, SampleResult, Status


def p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def status_of(spec: EvaluatorSpec, value: float | None) -> Status:
    if value is None:
        return "na"
    t = spec.thresholds
    if spec.higher_is_better:
        return "green" if value >= t.green else "amber" if value >= t.amber else "red"
    return "green" if value <= t.green else "amber" if value <= t.amber else "red"


def aggregate(spec: EvaluatorSpec, node_id: str, results: list[SampleResult], na_reason: str = "") -> NodeEvalResult:
    scored = [r for r in results if r.value is not None and r.error is None]
    errors = [r for r in results if r.error]
    if not scored:
        status: Status = "error" if errors else "na"
        reason = (f"all {len(errors)} sample(s) failed: {errors[0].error}" if errors else na_reason or "no applicable samples")
        return NodeEvalResult(node_id=node_id, evaluator_id=spec.id, value=None, unit=spec.unit, status=status,
                              n=0, reason=reason[:300])
    values = [r.value for r in scored]
    value = p95(values) if spec.unit == "ms" else statistics.fmean(values)
    worst_first = sorted(scored, key=lambda r: r.value, reverse=not spec.higher_is_better)
    good = spec.thresholds.green
    worst = [r.sample_id for r in worst_first
             if (r.value < good if spec.higher_is_better else r.value > good)][:3]
    reason = f"{len(errors)} sample(s) failed and are excluded" if errors else ""
    return NodeEvalResult(node_id=node_id, evaluator_id=spec.id, value=value, unit=spec.unit,
                          status=status_of(spec, value), n=len(scored), worst_samples=worst, reason=reason)
