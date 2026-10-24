"""Hotspot rules (docs/02-CONTRACTS.md §10).

Rules read generic static signals: flag-constant names + literal values, missing HTTP timeouts,
prompt-string content, and the presence/absence of component types. Nothing here names a
specific repository. Each rule returns (node_id, evidence) pairs; evidence always comes from facts.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from ..analyzer.facts import ConstFact, RepoFacts
from ..core.schemas import ApplicationGraph, Evidence, GraphNode, Severity

Finding = tuple[str, list[Evidence]]


def _fmt(value) -> str:
    return f'"{value}"' if isinstance(value, str) else repr(value)


class Ctx:
    def __init__(self, graph: ApplicationGraph, facts: RepoFacts):
        self.graph = graph
        self.facts = facts
        self.consts = facts.all_consts()

    # -- nodes
    def nodes(self, *types: str) -> list[GraphNode]:
        return sorted((n for n in self.graph.nodes if n.type in types), key=lambda n: n.id)

    def target(self, node_type: str, file: str | None) -> GraphNode | None:
        """Node of a type in the same file, else the first node of that type."""
        candidates = self.nodes(node_type)
        same = [n for n in candidates if n.file == file]
        return (same or candidates or [None])[0]

    def in_file(self, node_type: str, file: str) -> list[GraphNode]:
        return [n for n in self.nodes(node_type) if n.file == file]

    def fed_by(self, node_id: str, *types: str) -> bool:
        types_by_id = {n.id: n.type for n in self.graph.nodes}
        return any(e.target == node_id and types_by_id.get(e.source) in types for e in self.graph.edges)

    # -- constants
    def find(self, name_pattern: str, pred: Callable[[object], bool]) -> list[tuple[str, ConstFact]]:
        rx = re.compile(name_pattern)
        return [(p, c) for p, c in self.consts if rx.search(c.name) and _safe(pred, c.value)]

    @staticmethod
    def ev(path: str, c: ConstFact) -> Evidence:
        return Evidence(file=path, line=c.line, snippet=c.snippet, fact=f"{c.name} = {_fmt(c.value)}")


def _safe(pred: Callable[[object], bool], value: object) -> bool:
    try:
        return bool(pred(value))
    except TypeError:
        return False


is_false = lambda v: v is False
is_true = lambda v: v is True
is_none = lambda v: v is None
is_num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)


# ---------------------------------------------------------------- rules


def rag_topk_no_rerank(x: Ctx) -> list[Finding]:
    topk = x.find(r"(^|_)(TOP_K|K)$|TOP_K", lambda v: is_num(v) and v >= 8)
    if not topk:
        return []
    rerank_flags = x.find(r"USE_RERANK|RERANK_ENABLED|ENABLE_RERANK", is_false)
    if x.nodes("reranker") and not rerank_flags:
        return []
    out = []
    for path, c in topk:
        node = x.target("retriever", path)
        if node:
            extra = [x.ev(p, f) for p, f in rerank_flags] or [
                Evidence(file=node.file or path, line=None, fact="no reranker component detected")]
            out.append((node.id, [x.ev(path, c), *extra]))
    return out


def rag_big_chunks(x: Ctx) -> list[Finding]:
    hits = x.find(r"CHUNK_SIZE", lambda v: is_num(v) and v >= 800) + x.find(r"CHUNK_OVERLAP", lambda v: v == 0)
    node = x.target("retriever", hits[0][0]) if hits else None
    return [(node.id, [x.ev(p, c) for p, c in hits])] if node else []


def _answer_llms(x: Ctx, path: str) -> list[GraphNode]:
    llms = x.in_file("llm", path)
    fed = [n for n in llms if x.fed_by(n.id, "retriever", "reranker")]
    return fed or llms


def llm_high_temp(x: Ctx) -> list[Finding]:
    out = []
    for path, c in x.find(r"TEMPERATURE", lambda v: is_num(v) and v > 0.7):
        out += [(n.id, [x.ev(path, c)]) for n in x.in_file("llm", path)]
    for f in x.facts.files:  # literal temperature=… kwargs inside llm components
        for d in f.defs:
            node = next((n for n in x.nodes("llm") if n.file == f.path and n.symbol == d.name), None)
            for call in d.calls if node else []:
                t = call.kwargs.get("temperature")
                if is_num(t) and t > 0.7:
                    out.append((node.id, [Evidence(file=f.path, line=call.line, snippet=call.snippet,
                                                   fact=f"temperature={t}")]))
    return out


def llm_ungrounded(x: Ctx) -> list[Finding]:
    out = []
    for path, c in x.find(r"GROUND|STRICT_(PROMPT|ANSWER|GROUNDING)|^STRICT$|CITATION", is_false):
        out += [(n.id, [x.ev(path, c)]) for n in _answer_llms(x, path)]
    return out


def llm_unbounded_context(x: Ctx) -> list[Finding]:
    out = []
    for path, c in x.find(r"MAX_CONTEXT|CONTEXT_BUDGET|CONTEXT_LIMIT|MAX_CHUNKS", is_none):
        out += [(n.id, [x.ev(path, c)]) for n in _answer_llms(x, path)]
    return out


def extra_llm_hop(x: Ctx) -> list[Finding]:
    out = []
    for path, c in x.find(r"^ENABLE_", is_true):
        for n in x.in_file("llm", path):
            if re.search(r"rewrite|condense|rephrase|expand|hyde|reformulate", (n.symbol or "").lower()):
                out.append((n.id, [x.ev(path, c)]))
    return out


BIG_MODEL = re.compile(r"^large$|gpt-4o(?!-mini)|gpt-4\.1(?!-mini|-nano)|opus|sonnet|^gpt-5(?!-mini|-nano)")


def big_model_for_classify(x: Ctx) -> list[Finding]:
    out = []
    for path, c in x.find(r"(ROUTER|ROUTING|CLASSIF|INTENT).*MODEL", lambda v: isinstance(v, str) and BIG_MODEL.search(v)):
        node = x.target("router", path)
        if node:
            out.append((node.id, [x.ev(path, c)]))
    return out


def router_no_schema(x: Ctx) -> list[Finding]:
    out = []
    for path, c in x.find(r"FEW_SHOT|EXAMPLES|ARG_SCHEMA|TOOL_SCHEMA", is_false):
        node = x.target("router", path)
        if node:
            out.append((node.id, [x.ev(path, c)]))
    return out


def parser_naive_json(x: Ctx) -> list[Finding]:
    out = []
    for path, c in x.find(r"ROBUST|STRICT_JSON|SAFE_PARSE|REPAIR_JSON", is_false):
        node = x.target("parser", path) or x.target("router", path)
        if node:
            out.append((node.id, [x.ev(path, c)]))
    return out


def tool_brittle_lookup(x: Ctx) -> list[Finding]:
    return [(n.id, [x.ev(path, c)]) for path, c in x.find(r"FUZZY|NORMALI[SZ]E", is_false)
            for n in x.in_file("tool", path)]


def tool_unvalidated(x: Ctx) -> list[Finding]:
    retries = [x.ev(p, c) for p, c in x.find(r"REPAIR|RETRY|RETRIES", lambda v: v is False or v == 0)]
    out = [(n.id, [x.ev(path, c), *retries]) for path, c in x.find(r"VALIDATE", is_false)
           for n in x.in_file("tool", path)]
    if not out and retries and x.nodes("tool"):
        out.append((x.nodes("tool")[0].id, retries))
    return out


HTTP = re.compile(r"^(requests|httpx)\.(get|post|put|delete|request)$")


def ext_no_timeout(x: Ctx) -> list[Finding]:
    out = []
    for n in x.nodes("external_api", "tool"):
        f = x.facts.file(n.file or "")
        d = next((d for d in f.defs if d.name == n.symbol), None) if f else None
        evs = []
        for call in d.calls if d else []:
            if HTTP.match(call.callee) and ("timeout" not in call.kwargs or call.kwargs["timeout"] is None):
                evs.append(Evidence(file=n.file, line=call.line, snippet=call.snippet, fact=f"{call.callee} without timeout"))
        if n.type == "external_api":
            evs += [x.ev(p, c) for p, c in x.find(r"TIMEOUT", is_none) if p == n.file]
        if evs:
            evs += [x.ev(p, c) for p, c in x.find(r"CACHE", is_false) if p == n.file]
            out.append((n.id, evs))
    return out


def agent_ungrounded(x: Ctx) -> list[Finding]:
    out = []
    for path, c in x.find(r"^USE_", is_false):
        for n in x.in_file("agent", path):
            if n.provider not in ("langgraph", "langchain"):
                out.append((n.id, [x.ev(path, c)]))
    return out


def no_guardrail(x: Ctx) -> list[Finding]:
    guards = x.nodes("guardrail")
    if not guards:
        entry = (x.nodes("entrypoint") or x.nodes("agent") or [None])[0]
        return [(entry.id, [Evidence(file=entry.file or "", line=None, fact="no guardrail component detected")])] if entry else []
    return [(x.target("guardrail", path).id, [x.ev(path, c)]) for path, c in x.find(r"GUARDRAIL|MODERATION|SAFETY", is_false)]


def prompt_secret(x: Ctx) -> list[Finding]:
    evs = [Evidence(file=f.path, line=p.line, snippet="", fact=f"prompt {p.name} contains internal notes")
           for f in x.facts.files for p in f.prompts if p.has_internal_notes]
    node = (x.nodes("guardrail") or x.nodes("entrypoint") or x.nodes("llm") or [None])[0]
    return [(node.id, evs)] if evs and node else []


def memory_unbounded(x: Ctx) -> list[Finding]:
    out = []
    for path, c in x.find(r"WINDOW|MAX_TURNS|MAX_HISTORY|HISTORY_LIMIT", is_none):
        node = x.target("memory", path)
        if node:
            out.append((node.id, [x.ev(path, c)]))
    return out


def no_eval_suite(x: Ctx) -> list[Finding]:
    if x.facts.has_eval_suite:
        return []
    node = (x.nodes("entrypoint") or x.graph.nodes or [None])[0]
    return [(node.id, [Evidence(file=node.file or "", line=None, fact="no eval/golden dataset found in the repo")])] if node else []


@dataclass(frozen=True)
class Rule:
    id: str
    severity: Severity
    title: str
    impacts: tuple[str, ...]
    evaluators: tuple[str, ...]
    fix_variant: str | None
    find: Callable[[Ctx], list[Finding]]


RULES: list[Rule] = [
    Rule("RAG_TOPK_NO_RERANK", "high", "Context relevance likely weak", ("irrelevant context", "higher token cost"),
         ("context_relevance", "context_precision"), "fix-retrieval", rag_topk_no_rerank),
    Rule("RAG_BIG_CHUNKS", "medium", "Large chunks mix several topics", ("irrelevant context", "lower precision"),
         ("context_precision",), "fix-retrieval", rag_big_chunks),
    Rule("LLM_HIGH_TEMP", "medium", "High sampling temperature on a factual path", ("hallucination", "inconsistent numbers"),
         ("faithfulness", "numeric_accuracy"), "fix-generation", llm_high_temp),
    Rule("LLM_UNGROUNDED", "high", "Answers are not constrained to the context", ("hallucination", "no citations"),
         ("faithfulness",), "fix-generation", llm_ungrounded),
    Rule("LLM_UNBOUNDED_CONTEXT", "medium", "All retrieved context is stuffed into the prompt", ("token cost", "latency"),
         ("cost", "latency"), "fix-generation", llm_unbounded_context),
    Rule("EXTRA_LLM_HOP", "medium", "Extra LLM round trip on every request", ("latency", "cost"),
         ("latency", "cost"), "fix-cost", extra_llm_hop),
    Rule("BIG_MODEL_FOR_CLASSIFY", "low", "Classification runs on the large model", ("cost",),
         ("cost",), "fix-cost", big_model_for_classify),
    Rule("ROUTER_NO_SCHEMA", "high", "Router prompt has no argument schema or examples", ("misroutes", "tool argument errors"),
         ("routing_accuracy", "tool_correctness"), "fix-routing", router_no_schema),
    Rule("PARSER_NAIVE_JSON", "medium", "LLM output parsed with bare json.loads", ("parse failures", "fallback routes"),
         ("parse_success", "routing_accuracy"), "fix-routing", parser_naive_json),
    Rule("TOOL_BRITTLE_LOOKUP", "medium", "Exact-match lookup on user-provided names", ("tool failures",),
         ("tool_correctness", "numeric_accuracy"), "fix-tools", tool_brittle_lookup),
    Rule("TOOL_UNVALIDATED", "medium", "Tool arguments are not validated or repaired", ("bad records", "failed calls"),
         ("tool_correctness",), "fix-tools", tool_unvalidated),
    Rule("EXT_NO_TIMEOUT", "medium", "External call without timeout or cache", ("latency", "hang / outage risk"),
         ("latency",), "fix-tools", ext_no_timeout),
    Rule("AGENT_UNGROUNDED", "medium", "Agent works without the analyzer evidence available to it", ("missed issues", "false positives"),
         ("review_accuracy",), "fix-review", agent_ungrounded),
    Rule("NO_GUARDRAIL", "high", "No active input guardrail", ("prompt injection", "off-topic answers"),
         ("injection_resistance",), "fix-safety", no_guardrail),
    Rule("PROMPT_SECRET", "high", "Internal notes embedded in a system prompt", ("secret leakage via injection",),
         ("injection_resistance",), "fix-safety", prompt_secret),
    Rule("MEMORY_UNBOUNDED", "medium", "Whole conversation replayed on every turn", ("token growth", "latency"),
         ("memory_footprint", "cost"), "fix-memory", memory_unbounded),
    Rule("NO_EVAL_SUITE", "low", "No evaluation dataset in the repo", ("regressions go unnoticed",),
         ("answer_relevance",), None, no_eval_suite),
]
