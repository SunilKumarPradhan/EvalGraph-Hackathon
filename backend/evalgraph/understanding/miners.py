"""RECOMMEND (static half): generic evidence miners -> candidate hotspots.

Miners read call sites (kwargs, positional numbers, loop/try context), constants, config files and
prompt strings. They never refer to a specific repo. Values written as `cfg.top_k` / `settings.TOP_K`
are resolved through constants and config files in the buffer (config resolution).
"""

from __future__ import annotations

import json
import re
import tomllib
from collections import defaultdict
from dataclasses import dataclass, field

import yaml

from ..analyzer.facts import CallFact, FileFacts, RepoFacts
from ..core.schemas import ApplicationGraph, ConfigValue, Evidence, GraphNode
from ..hotspots.rules import RULES as FLAG_RULES, Ctx as FlagCtx
from .buffer import RepoBuffer
from .components import Component, SymbolGraph
from .flow import SYSTEM_CHANNELS, FlowResult
from .signals import INDEX_BUILD, RETRIEVE_CALLS, SymbolInfo

HTTP_RX = re.compile(r"^(requests|httpx|axios)\.(get|post|put|patch|delete|request)$|^fetch$|^axios$")
KEY_RX = re.compile(r"""["'](sk-[A-Za-z0-9_\-]{20,}|sk-ant-[A-Za-z0-9_\-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{30,}|AIza[0-9A-Za-z_\-]{30,})["']""")
TOPK_KEYS = {"k", "top_k", "topk", "topK", "n_results", "similarity_top_k", "limit", "top_n", "fetch_k"}
FILTER_KEYS = {"filter", "filters", "where", "metadata_filter", "query_filter", "filter_dict", "where_document"}
BIG_MODEL = re.compile(r"^large$|gpt-4o(?!-mini)|gpt-4\.1(?!-mini|-nano)|gpt-4(?!o|\.1|-turbo)|opus|sonnet|gpt-5(?!-mini|-nano)")


@dataclass
class Finding:
    rule_id: str
    node_id: str
    evidence: list[Evidence]
    params: dict[str, ConfigValue] = field(default_factory=dict)


# metadata for rules not already defined by the flag-constant family
TAXONOMY_EXTRA = {
    "RAG_NO_METADATA_FILTER": ("low", "Vector search without metadata filtering", ("off-topic context",),
                               ("context_precision", "context_relevance"), None),
    "INDEX_REBUILT_PER_REQUEST": ("high", "Index / embeddings rebuilt on every request", ("latency", "embedding cost"),
                                  ("latency", "cost"), None),
    "LLM_NO_MAX_TOKENS": ("low", "LLM call without an output token limit", ("cost", "latency spikes"),
                          ("cost", "latency"), None),
    "LLM_IN_LOOP": ("medium", "LLM called inside a loop", ("latency", "cost multiplies with input size"),
                    ("latency", "cost"), None),
    "LLM_NO_RETRY": ("low", "No retry / timeout around LLM calls", ("failed requests", "hangs"),
                     ("latency", "task_completion"), None),
    "PROMPT_INJECTION_SURFACE": ("high", "User input interpolated into a system prompt", ("prompt injection",),
                                 ("injection_resistance", "safety"), "fix-safety"),
    "HARDCODED_SECRET": ("high", "API key hardcoded in source", ("credential leak",), ("safety",), None),
    # data-flow rules (understanding/flow.py): where model output and untrusted text actually go
    "LLM_OUTPUT_TO_EXEC": ("high", "Model output executed as code / shell / SQL", ("remote code execution", "data loss"),
                           ("safety", "injection_resistance"), None),
    "LLM_OUTPUT_UNVALIDATED": ("medium", "Model-chosen tool arguments used without validation", ("tool errors", "unsafe calls"),
                               ("tool_correctness", "task_completion"), None),
    "INDIRECT_INJECTION": ("high", "Tool / web output fed back to a tool-calling model", ("indirect prompt injection",),
                           ("injection_resistance", "safety", "task_completion"), None),
}


class ConfigIndex:
    """name -> (value, file, line) from constants and YAML/JSON/TOML config files."""

    def __init__(self, buffer: RepoBuffer, files: list[FileFacts]):
        self.values: dict[str, tuple[ConfigValue, str, int | None]] = {}
        ranked = sorted(buffer.files, key=lambda p: (0 if re.search(r"(default|base|baseline|config|settings)\.", p) else 1, p))
        for path in ranked:
            if path.endswith((".yaml", ".yml", ".json", ".toml")) and not re.search(r"package(-lock)?\.json$|tsconfig", path):
                for key, value in self._flatten(self._load(path, buffer.files[path])):
                    line = self._line_of(buffer.files[path], key)
                    self.values.setdefault(key.lower(), (value, path, line))
        for f in files:
            for c in f.consts:
                self.values.setdefault(c.name.lower(), (c.value, f.path, c.line))

    @staticmethod
    def _load(path: str, text: str):
        try:
            if path.endswith((".yaml", ".yml")):
                return yaml.safe_load(text)
            if path.endswith(".json"):
                return json.loads(text)
            return tomllib.loads(text)
        except Exception:
            return None

    def _flatten(self, data, prefix: str = ""):
        if isinstance(data, dict):
            for k, v in data.items():
                yield from self._flatten(v, str(k))
        elif isinstance(data, (int, float, bool, str)) or data is None:
            if prefix:
                yield prefix, data

    @staticmethod
    def _line_of(text: str, key: str) -> int | None:
        for i, line in enumerate(text.splitlines(), 1):
            if re.match(rf"\s*[\"']?{re.escape(key)}[\"']?\s*[:=]", line):
                return i
        return None

    def resolve(self, expr: ConfigValue) -> tuple[ConfigValue, str | None, int | None] | None:
        """`cfg.top_k` -> value of top_k / DEFAULT_TOP_K / TOP_K from config files or constants."""
        if not isinstance(expr, str) or not re.fullmatch(r"[A-Za-z_][\w.]*", expr):
            return None
        name = expr.split(".")[-1].lower()
        for cand in (name, f"default_{name}", name.upper().lower()):
            if cand in self.values:
                v, path, line = self.values[cand]
                return v, path, line
        for key, (v, path, line) in self.values.items():
            if key.endswith("_" + name):
                return v, path, line
        return None


class Miner:
    def __init__(self, buffer: RepoBuffer, files: list[FileFacts], symbols: dict[str, SymbolInfo],
                 comps: list[Component], g: SymbolGraph, has_eval_suite: bool, scope: set[str] | None = None,
                 flow: FlowResult | None = None):
        self.buffer, self.files, self.comps, self.g, self.flow = buffer, files, comps, g, flow
        # only code on the application's request paths is mined (not tests, scripts or client plumbing)
        self.symbols = {k: v for k, v in symbols.items() if scope is None or k in scope}
        self.project_fns = {info.d.name for info in symbols.values()}
        self.has_eval_suite = has_eval_suite
        self.config = ConfigIndex(buffer, files)
        self.by_key: dict[str, list[Component]] = defaultdict(list)
        for c in comps:
            self.by_key[c.key].append(c)
        self.findings: list[Finding] = []

    # ---------------------------------------------------------------- helpers
    def types(self, *types: str) -> list[Component]:
        return [c for c in self.comps if c.type in types]

    def target(self, key: str, *types: str) -> Component | None:
        """Component of the given types at this symbol, else at a caller (2 hops), else same file, else any."""
        frontier, seen = [key], {key}
        for _ in range(3):
            for k in frontier:
                hit = next((c for c in self.by_key.get(k, []) if c.type in types), None)
                if hit:
                    return hit
            frontier = [p for k in frontier for p in self.g.inc.get(k, ()) if p not in seen]
            seen.update(frontier)
        path = key.split("::")[0]
        same = [c for c in self.types(*types) if c.path == path]
        return same[0] if same else None

    def ev(self, path: str, call: CallFact, fact: str) -> Evidence:
        return Evidence(file=path, line=call.line, snippet=call.snippet, fact=fact)

    def value(self, raw: ConfigValue) -> tuple[ConfigValue, Evidence | None]:
        """Literal or config-resolved value (+ evidence for the resolved source)."""
        if isinstance(raw, (int, float, bool)) or raw is None:
            return raw, None
        hit = self.config.resolve(raw)
        if hit is None:
            return raw, None
        v, path, line = hit
        return v, Evidence(file=path or "", line=line, snippet=self.buffer.line(path or "", line), fact=f"{raw} resolves to {v!r}")

    def add(self, rule: str, comp: Component | None, evidence: list[Evidence], **params) -> None:
        if comp is not None and evidence:
            self.findings.append(Finding(rule, comp.id, evidence, params))

    def calls(self):
        for key, info in self.symbols.items():
            for call in info.d.calls:
                yield key, info, call

    # ---------------------------------------------------------------- miners
    def retrieval(self) -> None:
        has_reranker = bool(self.types("reranker"))
        for key, info, call in self.calls():
            last = call.callee.replace("()", "").split(".")[-1]
            if "." not in call.callee.replace("()", "") and last in self.project_fns:
                continue  # a call into the project's own function, not a library search
            is_retrieval = last in RETRIEVE_CALLS or any(s.facet == "retriever" and s.line == call.line and not s.weak for s in info.signals)
            if not is_retrieval:
                continue
            comp = self.target(key, "retriever")
            k_val, k_ev = None, []
            for name, raw in call.kwargs.items():
                if name in TOPK_KEYS:
                    v, extra = self.value(raw)
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        k_val, k_ev = v, [self.ev(info.path, call, f"{name}={raw}")] + ([extra] if extra else [])
                elif name in ("search_kwargs", "searchKwargs") and isinstance(raw, str):
                    m = re.search(r"['\"]?k['\"]?\s*:\s*(\d+)", raw)
                    if m:
                        k_val, k_ev = int(m.group(1)), [self.ev(info.path, call, f"{name} k={m.group(1)}")]
            if k_val is None and last.startswith(("similarity", "similaritySearch", "max_marginal")) and call.num_args:
                k_val, k_ev = call.num_args[0], [self.ev(info.path, call, f"k={int(call.num_args[0])} (positional)")]
            if k_val is not None and k_val >= 8 and not has_reranker:
                self.add("RAG_TOPK_NO_RERANK", comp, k_ev + [Evidence(file=info.path, line=None, fact="no reranker component detected")])
            if not FILTER_KEYS & set(call.kwargs) and last not in ("as_retriever", "asRetriever", "as_query_engine"):
                self.add("RAG_NO_METADATA_FILTER", comp, [self.ev(info.path, call, f"{call.callee} has no metadata filter")])

    def chunking(self) -> None:
        for key, info, call in self.calls():
            last = call.callee.replace("()", "").split(".")[-1]
            if not re.search(r"Splitter$|SentenceSplitter|SemanticChunker", last):
                continue
            ev = []
            for name, raw in call.kwargs.items():
                v, extra = self.value(raw)
                if name in ("chunk_size", "chunkSize") and isinstance(v, (int, float)) and v >= 1000:
                    ev += [self.ev(info.path, call, f"{name}={v}")] + ([extra] if extra else [])
                if name in ("chunk_overlap", "chunkOverlap") and v == 0:
                    ev += [self.ev(info.path, call, f"{name}=0")]
            self.add("RAG_BIG_CHUNKS", self.target(key, "retriever", "dataset"), ev)

    def index_per_request(self) -> None:
        entries = [c.key for c in self.types("entrypoint")]
        request_path = set().union(*(self.g.reach(e) for e in entries)) if entries else set()
        for key, info, call in self.calls():
            last = call.callee.replace("()", "").split(".")[-1]
            if last not in INDEX_BUILD or key not in request_path:
                continue
            cached = any(s.facet == "cache" for s in info.signals) or any(n.startswith("_") and n.isidentifier() for n in info.d.names_used)
            if not cached:
                self.add("INDEX_REBUILT_PER_REQUEST", self.target(key, "vector_db", "retriever", "dataset"),
                         [self.ev(info.path, call, f"{call.callee} runs on the request path without caching")])

    def llm_calls(self) -> None:
        has_retry = any(s.facet == "retry" for info in self.symbols.values() for s in info.signals)
        llm_keys = {c.key for c in self.types("llm", "router", "agent")}
        for key, info, call in self.calls():
            sigs = [s for s in info.signals if s.line == call.line and s.facet in ("llm", "client") and not s.weak]
            if not sigs:
                continue
            comp = self.target(key, "llm", "router", "agent", "guardrail")
            for name in ("temperature",):
                v, extra = self.value(call.kwargs.get(name)) if name in call.kwargs else (None, None)
                if isinstance(v, (int, float)) and v > 0.7:
                    self.add("LLM_HIGH_TEMP", comp, [self.ev(info.path, call, f"temperature={v}")] + ([extra] if extra else []))
            direct_api = any(s.fact.startswith("calls ") for s in sigs)
            if direct_api and not {"max_tokens", "max_output_tokens", "maxTokens", "max_completion_tokens"} & set(call.kwargs):
                self.add("LLM_NO_MAX_TOKENS", comp, [self.ev(info.path, call, f"{call.callee} without max_tokens")])
            if call.in_loop:
                self.add("LLM_IN_LOOP", comp, [self.ev(info.path, call, f"{call.callee} inside a loop")])
            model, extra = self.value(call.kwargs.get("model")) if "model" in call.kwargs else (None, None)
            if isinstance(model, str) and BIG_MODEL.search(model) and comp is not None and comp.type == "router":
                self.add("BIG_MODEL_FOR_CLASSIFY", comp, [self.ev(info.path, call, f"model={model}")])
            sdk_retries = any((s.provider or "") in ("openai", "anthropic", "dial", "azure-openai") for s in sigs)
            if not has_retry and direct_api and not sdk_retries and not {"max_retries", "maxRetries", "timeout"} & set(call.kwargs):
                self.add("LLM_NO_RETRY", comp, [self.ev(info.path, call, f"{call.callee} with no retry/timeout")])
        # calls into LLM components from inside loops
        for key, info, call in self.calls():
            if call.in_loop:
                name = call.callee.replace("()", "").split(".")[-1]
                for c in self.comps:
                    if c.key in llm_keys and c.symbol == name and c.key != key:
                        self.add("LLM_IN_LOOP", c, [self.ev(info.path, call, f"{name}() called inside a loop")])

    def parsing(self) -> None:
        for key, info, call in self.calls():
            if call.callee not in ("json.loads", "JSON.parse") or call.in_try:
                continue
            llm_facing = any(s.facet in ("llm", "parser") for s in info.signals)
            structured = any(s.facet == "structured" for s in info.signals)
            if llm_facing and not structured:
                self.add("PARSER_NAIVE_JSON", self.target(key, "parser", "router", "llm"),
                         [self.ev(info.path, call, f"{call.callee} on model output outside try")])

    def http(self) -> None:
        for key, info, call in self.calls():
            if not HTTP_RX.match(call.callee.replace("()", "")):
                continue
            raw = call.kwargs.get("timeout", "<missing>")
            v, extra = self.value(raw) if raw != "<missing>" else ("<missing>", None)
            if v in ("<missing>", None):
                fact = f"{call.callee} without timeout" if v == "<missing>" else f"{call.callee} timeout={raw} (None)"
                self.add("EXT_NO_TIMEOUT", self.target(key, "external_api", "tool"),
                         [self.ev(info.path, call, fact)] + ([extra] if extra else []))

    def safety(self) -> None:
        guards = self.types("guardrail")
        anchor = (guards or self.types("entrypoint") or self.types("agent") or self.types("llm") or [None])[0]
        if not guards and anchor is not None:
            self.add("NO_GUARDRAIL", anchor, [Evidence(file=anchor.path, line=None, fact="no guardrail / moderation component detected")])
        for f in self.files:
            for p in f.prompts:
                if p.has_internal_notes:
                    self.add("PROMPT_SECRET", anchor, [Evidence(file=f.path, line=p.line, snippet=self.buffer.line(f.path, p.line)[:200],
                                                                fact=f"prompt {p.name} contains internal notes")])
                if not guards and re.search(r"\{[a-z_]*(question|query|input|message|user)[a-z_]*\}", p.preview, re.I) \
                        and re.search(r"you are|system", p.preview, re.I):
                    llm = next((c for c in self.types("llm") if p.name in self.symbols[c.key].d.names_used), None)
                    self.add("PROMPT_INJECTION_SURFACE", llm or anchor,
                             [Evidence(file=f.path, line=p.line, snippet=self.buffer.line(f.path, p.line)[:200], fact=f"prompt {p.name} embeds user input")])
        for path, text in self.buffer.code_files().items():
            for i, line in enumerate(text.splitlines(), 1):
                if KEY_RX.search(line) and anchor is not None:
                    self.add("HARDCODED_SECRET", anchor, [Evidence(file=path, line=i, snippet="<redacted>", fact="hardcoded API key")])

    def memory(self) -> None:
        bounded = re.compile(r"\[-\s*\w+\s*:\]|\[-\d+:\]|\.pop\(|del\s+\w+\[|trim|truncat|window|summar|max_turns|max_history", re.I)
        for key, info, call in self.calls():
            parts = call.callee.replace("()", "").split(".")
            if parts[-1] in ("append", "extend", "push") and len(parts) >= 2 and re.search(r"history|messages|conversation|turns|chat", parts[-2], re.I):
                src = "\n".join(self.buffer.files.get(info.path, "").splitlines()[info.d.line - 1:info.d.end_line])
                var = parts[-2]
                builds_prompt = re.search(rf"\breturn\s+{re.escape(var)}\b", src)  # local list returned = prompt building
                if not bounded.search(src) and not builds_prompt:
                    comp = self.target(key, "memory", "llm", "agent", "entrypoint")
                    self.add("MEMORY_UNBOUNDED", comp, [self.ev(info.path, call, f"{call.callee}: history grows without truncation")])
        for key, info, call in self.calls():
            if any(s.facet == "memory_unbounded" and s.line == call.line for s in info.signals):
                self.add("MEMORY_UNBOUNDED", self.target(key, "memory"), [self.ev(info.path, call, f"{call.callee} keeps the whole history")])

    def flags(self) -> None:
        """The PHASE-03 flag-constant rules, run as one generic miner family."""
        nodes = [GraphNode(id=c.id, type=c.type, label=c.label, file=c.path, symbol=c.symbol, provider=c.provider)
                 for c in self.comps]
        graph = ApplicationGraph(nodes=nodes, edges=[])
        facts = RepoFacts(name="", root="", files=self.files, has_eval_suite=self.has_eval_suite)
        ctx = FlagCtx(graph, facts)
        for rule in FLAG_RULES:
            if rule.id in ("NO_GUARDRAIL", "PROMPT_SECRET") :
                if rule.id == "NO_GUARDRAIL":
                    for node_id, evs in rule.find(ctx):
                        if any("GUARDRAIL" in e.fact or "MODERATION" in e.fact or "SAFETY" in e.fact for e in evs):
                            self.findings.append(Finding(rule.id, node_id, evs))
                continue
            for node_id, evs in rule.find(ctx):
                self.findings.append(Finding(rule.id, node_id, evs))

    def flows(self) -> None:
        """Data-flow rules: what the model's output reaches, and what untrusted text reaches the model."""
        if self.flow is None:
            return
        by_id = {c.id: c for c in self.comps}
        guards = self.types("guardrail")
        flagged = {f.rule_id for f in self.findings}

        def origin(taints) -> list[Evidence]:
            return [Evidence(file=t.path, line=t.line, snippet=self.buffer.line(t.path, t.line)[:200],
                             fact=f"{t.label.replace('_', ' ')} from {t.name}") for t in taints[:2] if t.line]

        for s in self.flow.sinks:
            here = by_id.get(s.comp) or self.target(s.key, "tool", "agent", "llm", "router", "parser", "external_api")
            model = s.taints("llm_output")
            from_llm = [t for t in model if t.comp]
            if s.kind == "exec" and model:
                self.add("LLM_OUTPUT_TO_EXEC", here, [s.evidence(f"{s.snippet[:60]} runs model-generated input")] + origin(model))
            elif s.kind == "tool" and from_llm and not s.validated and "TOOL_UNVALIDATED" not in flagged:
                self.add("LLM_OUTPUT_UNVALIDATED", here,
                         [s.evidence("model-chosen arguments go straight into the tool call")] + origin(from_llm))
            elif s.kind == "llm" and s.has_tools and not guards:
                untrusted = s.taints("external", "tool_output")
                if untrusted:
                    self.add("INDIRECT_INJECTION", here,
                             [s.evidence("tool / web output re-enters a model that can call tools")] + origin(untrusted))
            if s.kind == "llm" and not guards:
                system = sorted({t for k, ts in s.channels.items() if k in SYSTEM_CHANNELS for t in ts if t.label == "user_input"},
                                key=lambda t: (t.path, t.line or 0))
                if system:
                    self.add("PROMPT_INJECTION_SURFACE", here,
                             [s.evidence("user input flows into the system prompt")] + origin(system))

    def run(self) -> list[Finding]:
        for miner in (self.retrieval, self.chunking, self.index_per_request, self.llm_calls, self.parsing, self.http,
                      self.safety, self.memory, self.flags, self.flows):
            miner()
        return self.findings
