"""Compare two runs of the same analysis attachment by attachment (deltas only; no single verdict, conf_07 Q-33)."""

from __future__ import annotations

from ..core.schemas import RunComparison, RunResult
from ..evaluators.catalog import BY_ID


def compare(base: RunResult, other: RunResult) -> RunComparison:
    other_by_key = {(r.node_id, r.evaluator_id): r for r in other.results}
    rows = []
    keys = [(r.node_id, r.evaluator_id) for r in base.results]
    keys += [k for k in other_by_key if k not in set(keys)]
    base_by_key = {(r.node_id, r.evaluator_id): r for r in base.results}
    for key in keys:
        a, b = base_by_key.get(key), other_by_key.get(key)
        spec = BY_ID[key[1]]
        va = a.value if a else None
        vb = b.value if b else None
        delta = vb - va if va is not None and vb is not None else None
        better = None if delta is None or delta == 0 else (delta > 0) == spec.higher_is_better
        rows.append({"node_id": key[0], "evaluator_id": key[1], "unit": spec.unit, "base": va, "other": vb,
                     "delta": delta, "better": better,
                     "base_status": a.status if a else "na", "other_status": b.status if b else "na"})
    return RunComparison(base_run_id=base.id, other_run_id=other.id, rows=rows)
