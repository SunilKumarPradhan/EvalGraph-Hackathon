"""Red/amber results → bottlenecks: contributors from static analysis, an explanation, a suggested experiment.

The explanation may cite only what it is given: the node's hotspot evidence, its config, and the worst samples'
reasons. Offline/mock it's a template built from the same material; with DIAL an LLM writes it from that material.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel

from ..core.schemas import (
    AnalysisResult,
    Bottleneck,
    Evidence,
    Hotspot,
    NodeEvalResult,
    SampleResult,
)
from ..evaluators.base import fmt_value
from ..evaluators.catalog import BY_ID
from ..llm import LLMError, complete_json
from ..settings import get_settings

logger = logging.getLogger(__name__)
SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}

EXPLAIN_SYSTEM = (
    "You explain why a component of an LLM application scored badly in an evaluation. Use ONLY the facts given: "
    "the static-analysis findings, the component's configuration and the worst samples' judge reasons. Do not invent "
    "causes, numbers or code. 2-3 sentences, plain language, name the most likely cause first. If the facts don't "
    "explain the score, say so.")


class Explanation(BaseModel):
    explanation: str


def _relevant_hotspots(analysis: AnalysisResult, result: NodeEvalResult) -> list[Hotspot]:
    """Findings that predict this metric: on this node first, else anywhere upstream in the graph (e.g. a disabled
    guardrail explains a failing injection score measured at the entrypoint). Unrelated findings are never cited."""
    targeted = [h for h in analysis.hotspots if result.evaluator_id in h.recommended_evaluators]
    on_node = [h for h in targeted if h.node_id == result.node_id]
    return sorted(on_node or targeted, key=lambda h: SEVERITY_RANK[h.severity])


def _contributors(analysis: AnalysisResult, result: NodeEvalResult, hotspots: list[Hotspot]) -> list[Evidence]:
    out: list[Evidence] = []
    seen: set[tuple] = set()
    for ev in (e for h in hotspots for e in h.evidence):
        key = (ev.file, ev.line, ev.fact)
        if key not in seen:
            seen.add(key)
            out.append(ev)
    node = next((n for n in analysis.graph.nodes if n.id == result.node_id), None)
    if node is not None:
        cited = " ".join(e.fact for e in out)
        out += [Evidence(file=node.file or "", fact=f"config {k} = {v!r}")
                for k, v in list(node.config.items())[:4] if k not in cited]  # a hotspot already cites it
    return out[:8]


def _titled(h: Hotspot, result: NodeEvalResult, labels: dict[str, str]) -> str:
    return h.title if h.node_id == result.node_id else f"{h.title} (on {labels.get(h.node_id, h.node_id)})"


def _template(label: str, result: NodeEvalResult, hotspots: list[Hotspot], contributors: list[Evidence],
              worst: list[SampleResult], labels: dict[str, str]) -> str:
    spec = BY_ID[result.evaluator_id]
    op = "≥" if spec.higher_is_better else "≤"
    parts = [(f"{spec.name} on {label} is {fmt_value(result.value, spec.unit)} ({result.status}; green needs "
              f"{op} {fmt_value(spec.thresholds.green, spec.unit)}) over {result.n} sample(s).")]
    if hotspots:
        parts.append("Static analysis flagged: " + "; ".join(_titled(h, result, labels) for h in hotspots[:3]) + ".")
    if contributors:
        parts.append("Evidence: " + "; ".join(c.fact for c in contributors[:3]) + ".")
    if worst:
        parts.append("Worst samples: " + "; ".join(f"{w.sample_id}: {w.reason[:90]}" for w in worst[:3]) + ".")
    if not hotspots:
        parts.append("No static finding predicts this metric; inspect the worst samples.")
    return " ".join(parts)


def _llm_explain(label: str, result: NodeEvalResult, hotspots: list[Hotspot], contributors: list[Evidence],
                 worst: list[SampleResult], labels: dict[str, str]) -> str:
    spec = BY_ID[result.evaluator_id]
    user = (f"Component: {label} ({result.node_id})\nMetric: {spec.name} = {fmt_value(result.value, spec.unit)} "
            f"({result.status}), {result.n} samples\n\nStatic-analysis findings:\n"
            + "\n".join(f"- {_titled(h, result, labels)}: {h.why}" for h in hotspots[:4])
            + "\n\nEvidence:\n" + "\n".join(f"- {c.file}:{c.line or ''} {c.fact}" for c in contributors)
            + "\n\nWorst samples:\n" + "\n".join(f"- {w.sample_id}: {w.reason}" for w in worst[:3]))
    out, _ = complete_json("bottleneck_explain", EXPLAIN_SYSTEM, user, Explanation, temperature=0.0, max_tokens=400)
    return out.explanation.strip()


def diagnose(analysis: AnalysisResult, results: list[NodeEvalResult], samples: list[SampleResult]) -> list[Bottleneck]:
    labels = {n.id: n.label for n in analysis.graph.nodes}
    bad = [r for r in results if r.status in ("red", "amber")]

    def badness(r: NodeEvalResult) -> tuple:
        spec = BY_ID[r.evaluator_id]
        gap = (spec.thresholds.green - r.value) if spec.higher_is_better else (r.value - spec.thresholds.green)
        return (r.status != "red", -gap / (abs(spec.thresholds.green) or 1.0))

    out: list[Bottleneck] = []
    for r in sorted(bad, key=badness):
        hotspots = _relevant_hotspots(analysis, r)
        contributors = _contributors(analysis, r, hotspots)
        by_id = {s.sample_id: s for s in samples if s.node_id == r.node_id and s.evaluator_id == r.evaluator_id}
        worst = [by_id[sid] for sid in r.worst_samples if sid in by_id]
        label = labels.get(r.node_id, r.node_id)
        explanation = ""
        if not get_settings().mock_llm:
            try:
                explanation = _llm_explain(label, r, hotspots, contributors, worst, labels)
            except LLMError as exc:
                logger.warning("bottleneck explanation fell back to the template: %s", exc)
        explanation = explanation or _template(label, r, hotspots, contributors, worst, labels)
        variant = next((h.suggested_variant for h in hotspots
                        if h.suggested_variant and h.suggested_variant in analysis.summary.variants), None)
        fix = next((h for h in hotspots if h.suggested_variant == variant), hotspots[0] if hotspots else None)
        experiment = (f"Re-run with variant '{variant}' (targets: {fix.title})" if variant and fix
                      else f"Address '{fix.title}' and re-run" if fix else None)
        patchable = [h.id for h in hotspots if h.patch] if analysis.summary.adapter == "sandbox" else []
        if not variant and patchable:
            first = next(h for h in hotspots if h.id == patchable[0])
            experiment = f"Re-run with the rule-based patch ({'; '.join(e.summary for e in first.patch)})"
        out.append(Bottleneck(node_id=r.node_id, evaluator_id=r.evaluator_id, value=r.value, status=r.status,
                              contributors=contributors, explanation=explanation, suggested_experiment=experiment,
                              suggested_variant=variant, suggested_patch=patchable[:1]))
    return out
