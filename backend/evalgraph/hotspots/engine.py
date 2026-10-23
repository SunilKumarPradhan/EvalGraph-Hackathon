"""Evaluator recommendations from hotspots and component types."""

from __future__ import annotations

from ..core.schemas import ApplicationGraph, Hotspot, Recommendation, Severity
from ..evaluators.catalog import BY_ID, applies

SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}
# Golden-set fields a generated (synthetic) dataset never has. It does carry expected_behavior="refuse" probes.
LABEL_FIELDS = {"relevant_doc_ids", "reference_answer", "expected_numeric", "expected_route", "expected_tool",
                "expected_findings", "history"}
# Without labels, measure the same concern reference-free where a catalog evaluator exists; otherwise drop it.
REFERENCE_FREE = {"answer_relevance": "task_completion", "context_precision": "context_relevance",
                  "context_recall": "context_relevance", "numeric_accuracy": "faithfulness"}


def primary_entrypoint(graph: ApplicationGraph) -> str | None:
    """The entrypoint a request most likely comes through: write routes (POST, websocket, CLI) before GET routes,
    non-streaming before streaming, then graph order. Request-level evaluators go there, once."""
    eps = [n for n in graph.nodes if n.type == "entrypoint"]
    if not eps:
        return None

    def rank(item):
        i, n = item
        facts = " ".join(e.fact for e in n.evidence).lower()
        return ("@app.get" in facts or ".get " in facts, "stream" in (n.symbol or n.id).lower(), i)

    return min(enumerate(eps), key=rank)[1].id


def needs_labels(evaluator_id: str) -> bool:
    return bool(set(BY_ID[evaluator_id].needs) & LABEL_FIELDS)


def recommend(graph: ApplicationGraph, hotspots: list[Hotspot], labelled: bool = True, flow=None,
              proposals=()) -> list[Recommendation]:
    """labelled=False: the repo will be run on a generated dataset (no golden labels), so only recommend evaluators
    that can actually score there (conf_05/conf_06 Q-24).
    flow: understanding.flow.FlowResult (what each prompt is built from, where model output goes);
    proposals: the investigator's validated EvalProposals."""
    types = {n.id: n.type for n in graph.nodes}
    recs: dict[tuple[str, str], Recommendation] = {}

    def add(node_id: str, evaluator_id: str, priority: Severity, rationale: str) -> None:
        if node_id not in types or not applies(evaluator_id, types[node_id]):
            return
        key = (node_id, evaluator_id)
        if not labelled and needs_labels(evaluator_id):
            swap = REFERENCE_FREE.get(evaluator_id)
            if swap and applies(swap, types[node_id]):
                add(node_id, swap, priority, rationale)
            return
        if key not in recs or SEVERITY_RANK[priority] < SEVERITY_RANK[recs[key].priority]:
            recs[key] = Recommendation(node_id=node_id, evaluator_id=evaluator_id, priority=priority, rationale=rationale)

    for h in hotspots:
        for e in h.recommended_evaluators:
            add(h.node_id, e, h.severity, h.title)
    fed_by_retrieval = {e.target for e in graph.edges if types.get(e.source) in ("retriever", "reranker")}
    primary = primary_entrypoint(graph)
    for n in graph.nodes:
        if n.type == "llm" and n.id in fed_by_retrieval:
            add(n.id, "faithfulness", "medium", "Generation grounded in retrieved context")
        if n.id == primary:
            add(n.id, "answer_relevance", "medium", "End-to-end answer quality")
            if not labelled:  # every generated dataset carries prompt-injection probes and is reference-free
                add(n.id, "injection_resistance", "medium", "Generated injection probes test the whole request")
                add(n.id, "safety", "low", "Reference-free safety check of every answer")
            add(n.id, "latency", "low", "End-to-end latency")
            add(n.id, "cost", "low", "End-to-end cost")
        if n.type == "tool":
            add(n.id, "tool_correctness", "medium", "Every tool gets a correctness check")
        if n.type == "router":
            add(n.id, "routing_accuracy", "medium", "Routing decides everything downstream")
        if n.type in ("retriever",) and not any(r.node_id == n.id for r in recs.values()):
            add(n.id, "context_precision", "medium", "Retrieval quality")
        if n.type == "reranker":  # the final context: shows whether reranking actually sharpens what the LLM sees
            add(n.id, "context_precision", "medium", "Precision of the final, reranked context")
            add(n.id, "context_relevance", "low", "Relevance of the final, reranked context")
    if flow is not None:  # from data flow: what the model actually sees, and where its output goes
        for r in flow.recipes:
            where = f"{r.path}:{r.line}"
            if "retrieved" in r.parts:
                add(r.comp, "faithfulness", "medium", f"Prompt is built from retrieved context ({where})")
            if "tool_output" in r.parts or "external" in r.parts:
                add(r.comp, "faithfulness", "medium", f"Answer is built from tool / API results ({where})")
                if r.has_tools:
                    add(r.comp, "injection_resistance", "high", f"Tool output re-enters a tool-calling model ({where})")
            if "user_input" in r.system:
                add(r.comp, "injection_resistance", "high", f"User input reaches the system prompt ({where})")
        for s in flow.sinks:
            if s.kind == "parse" and s.comp and s.taints("llm_output"):
                add(s.comp, "parse_success", "medium", f"Model output is parsed at {s.path}:{s.line}")
    for p in proposals:
        add(p.node_id, p.evaluator_id, "medium", f"Investigator: {p.rationale.strip()[:160] or 'proposed evaluation'}")
    return sorted(recs.values(), key=lambda r: (SEVERITY_RANK[r.priority], r.node_id, r.evaluator_id))
