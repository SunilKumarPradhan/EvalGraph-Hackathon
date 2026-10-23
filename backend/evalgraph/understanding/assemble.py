"""MAP: assemble the pipeline graph.

Static candidate edges come from (1) call reachability through non-component helpers, (2) facets split
from one function, (3) sequential component calls inside orchestrator functions, (4) LangGraph wiring.
The LLM reduce step may add or remove edges, but an added edge must be backed by static reachability.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict, deque

from pydantic import BaseModel, Field

from ..analyzer.facts import FileFacts
from ..core.schemas import ApplicationGraph, EdgeKind, GraphEdge, GraphNode
from ..llm import LLMError, complete_json
from .components import Component, SymbolGraph
from .signals import SymbolInfo

FLOW_ORDER = ["entrypoint", "agent", "guardrail", "memory", "router", "dataset", "embedding", "vector_db",
              "retriever", "reranker", "prompt", "llm", "parser", "tool", "external_api"]
MAX_HOPS = 4
# facets split from one function are chained only along real pipeline pairs
CHAIN_PAIRS = {("dataset", "embedding"), ("dataset", "vector_db"), ("embedding", "vector_db"), ("vector_db", "retriever"),
               ("retriever", "reranker"), ("retriever", "llm"), ("reranker", "llm"), ("llm", "parser"),
               ("tool", "external_api"), ("router", "llm"), ("guardrail", "llm"), ("memory", "llm"), ("agent", "llm"),
               ("agent", "tool"), ("llm", "tool"), ("agent", "memory")}
# for a call from component A into component B (different functions), which facet pair carries the edge
PREFER = {"entrypoint": ["agent", "guardrail", "router", "memory", "retriever", "llm", "tool", "dataset"],
          "agent": ["guardrail", "router", "memory", "llm", "retriever", "tool"],
          "retriever": ["vector_db", "embedding", "reranker", "dataset", "external_api"],
          "dataset": ["embedding", "vector_db", "external_api"],
          "router": ["parser", "llm", "tool", "retriever"], "llm": ["retriever", "reranker", "tool", "parser", "external_api", "llm"],
          "tool": ["external_api", "vector_db", "retriever", "llm", "tool"], "guardrail": ["llm", "memory"],
          "memory": ["llm", "vector_db"], "reranker": ["llm"], "vector_db": ["embedding"], "prompt": ["llm"]}


class EdgeProposal(BaseModel):
    source: str
    target: str
    kind: EdgeKind = "feeds"
    label: str | None = None


class Reduction(BaseModel):
    add_edges: list[EdgeProposal] = Field(default_factory=list)
    remove_edges: list[str] = Field(default_factory=list)   # edge ids "a->b"


REDUCE_SYSTEM = """You assemble the request pipeline of an LLM application from its components.
You get components (id, type, label, purpose) and candidate edges found by static analysis.
Return edges to add (data flow the candidates miss, e.g. retriever -> llm "feeds") and candidate edge ids to remove
(edges that are not part of how a request flows). Use only the given component ids. Be conservative."""


class Assembler:
    def __init__(self, comps: list[Component], symbols: dict[str, SymbolInfo], g: SymbolGraph, files: list[FileFacts]):
        self.comps, self.symbols, self.g, self.files = comps, symbols, g, files
        self.at: dict[str, list[Component]] = defaultdict(list)
        for c in comps:
            self.at[c.key].append(c)
        self.edges: dict[tuple[str, str], GraphEdge] = {}

    def add(self, a: Component, b: Component, kind: EdgeKind, label: str | None = None) -> None:
        if a.id == b.id:
            return
        if b.type in ("vector_db", "embedding") and kind == "calls":
            kind = "uses"
        key = (a.id, b.id)
        if key not in self.edges or (kind == "routes_to" and self.edges[key].kind != "routes_to"):
            self.edges[key] = GraphEdge(id=f"{a.id}->{b.id}", source=a.id, target=b.id, kind=kind, label=label)

    def first_components(self, start: str, max_hops: int = MAX_HOPS, skip_start: bool = True) -> list[Component]:
        """Nearest components reachable from a symbol through non-component helpers."""
        found, seen = [], {start}
        queue = deque([(start, 0)])
        while queue:
            k, d = queue.popleft()
            if (k != start or not skip_start) and self.at.get(k):
                found += self.at[k]
                continue
            if d >= max_hops:
                continue
            for nxt in self.g.out.get(k, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, d + 1))
        return found

    def langgraph(self) -> set[str]:
        """Wire LangGraph nodes; returns symbol keys of graph builders (their reachability edges are skipped)."""
        builders = set()
        for f in self.files:
            node_fn = {lg.node: lg.fn for lg in f.langgraph if lg.kind == "add_node" and lg.node and lg.fn}
            if not node_fn:
                continue
            fn_key = {d.name: f"{f.path}::{d.qualname}" for d in f.defs}
            builder = next((c for c in self.comps if c.path == f.path and c.type == "agent"), None)
            if builder:
                builders.add(builder.key)
            registries = {r.name: r.entries for g in self.files for r in g.registries}

            def comps_of(node: str) -> list[Component]:
                found: list[Component] = []
                for fn in (node_fn.get(node) or "").split(","):
                    key = fn_key.get(fn.split(".")[-1])
                    if key:
                        found += self.at.get(key) or self.first_components(key, max_hops=1)
                return found

            out: dict[str, list[tuple[str, str | None, bool]]] = defaultdict(list)
            for lg in f.langgraph:
                if lg.kind == "add_edge" and lg.source and lg.target:
                    out[lg.source].append((lg.target, None, False))
                elif lg.kind == "add_conditional_edges" and lg.source:
                    for label, target in lg.mapping.items():
                        if label == "@any":
                            for other in node_fn:
                                if other != lg.source:
                                    out[lg.source].append((other, None, True))
                        elif label.startswith("@registry:"):
                            for key, fn in registries.get(label.split(":", 1)[1], {}).items():
                                out[lg.source].append((target, f"@{key}={fn}", True))
                        else:
                            out[lg.source].append((target, label, True))

            def resolve(node: str, seen: frozenset = frozenset()) -> list[Component]:
                if node in seen or node == "__end__":
                    return []
                found = comps_of(node)
                if found:
                    return found
                return [c for t, _, _ in out.get(node, []) for c in resolve(t, seen | {node})]

            for src, targets in out.items():
                sources = [builder] if src == "__start__" and builder else comps_of(src)
                for target, label, cond in targets:
                    for t in resolve(target):
                        shown = label
                        if label and label.startswith("@"):
                            key, fn = label[1:].split("=", 1)
                            if t.symbol != fn:
                                continue
                            shown = key
                        for s in sources:
                            self.add(s, t, "calls" if src == "__start__" else ("routes_to" if cond else "feeds"),
                                     shown if cond else None)
        return builders

    def build(self) -> ApplicationGraph:
        builders = self.langgraph()
        for key, here in self.at.items():
            # facets split from one function flow in pipeline order
            ordered = sorted(here, key=lambda c: FLOW_ORDER.index(c.type) if c.type in FLOW_ORDER else 99)
            for i, a in enumerate(ordered):
                nxt = next((b for b in ordered[i + 1:] if (a.type, b.type) in CHAIN_PAIRS), None)
                if nxt is not None:
                    self.add(a, nxt, "feeds")
            if ordered and ordered[0].type == "entrypoint":
                nxt = next((b for b in ordered[1:] if b.type not in ("tool", "external_api", "prompt", "memory")), None)
                if nxt is not None:
                    self.add(ordered[0], nxt, "feeds")
            if key in builders:
                continue
            if key.endswith("::<module>"):  # script code calls out only through its entrypoint
                here = [c for c in here if c.type == "entrypoint"]
                if not here:
                    continue
            targets: dict[str, list[Component]] = defaultdict(list)
            for t in self.first_components(key):
                if t.key != key:
                    targets[t.key].append(t)
            for tcomps in targets.values():
                a, b = self.best_pair(here, tcomps)
                if a.type == "llm" and b.type in ("retriever", "reranker", "memory"):
                    self.add(b, a, "feeds")  # the answer LLM calls retrieve(): retrieved context feeds the LLM
                else:
                    self.add(a, b, "calls")
        self.prompt_edges()
        self.reference_edges()
        # orchestrators: consecutive component calls inside one non-component function feed each other
        for key, info in self.symbols.items():
            if key in self.at or key in builders:
                continue
            seq: list[Component] = []
            for call in info.d.calls:
                name = call.callee.replace("()", "").split(".")[-1]
                hit = [c for k2, cs in self.at.items() if k2.endswith("::" + name) for c in cs]
                if hit and (not seq or seq[-1].id != hit[0].id):
                    seq.append(hit[0])
            for a, b in zip(seq, seq[1:]):
                self.add(a, b, "feeds")
        nodes = [GraphNode(id=c.id, type=c.type, label=c.label, file=c.path, symbol=c.symbol, provider=c.provider,
                           config=self.config_of(c), evidence=c.evidence, confidence=c.confidence, description=c.description)
                 for c in self.comps]
        return ApplicationGraph(nodes=nodes, edges=list(self.edges.values()))

    def config_of(self, c: Component) -> dict:
        """Call-site parameters + module constants of its file (referenced ones, or all of a small module)."""
        cfg = dict(c.params)
        f = next((f for f in self.files if f.path == c.path), None)
        if f is not None:
            used = set(self.symbols[c.key].d.names_used)
            small = len(f.consts) <= 8
            for const in f.consts:
                if (small or const.name in used) and not (isinstance(const.value, str) and len(const.value) > 60):
                    cfg[const.name] = const.value
        return dict(list(cfg.items())[:16])

    @staticmethod
    def best_pair(src: list[Component], dst: list[Component]) -> tuple[Component, Component]:
        order = lambda c: FLOW_ORDER.index(c.type) if c.type in FLOW_ORDER else 99  # noqa: E731
        active = [c for c in src if c.type not in ("prompt", "external_api")]  # passive components never call out
        src = sorted(active or src, key=order)
        best, score = (src[0], dst[0]), 10_000
        for a in src:
            prefs = PREFER.get(a.type, [])
            for b in dst:
                if b.type in prefs:
                    s = prefs.index(b.type)
                else:
                    s = 50 + (FLOW_ORDER.index(b.type) if b.type in FLOW_ORDER else 49)
                if a.type == "entrypoint":
                    s -= 1
                if s < score:
                    best, score = (a, b), s
        return best

    def reference_edges(self) -> None:
        """A function that uses an object built elsewhere (`store`, `query_engine`, `weather = tool(...)`) depends on it."""
        built: dict[str, list[Component]] = defaultdict(list)
        for c in self.comps:
            info = self.symbols[c.key]
            for call in info.d.calls:
                if call.target and any(s.line == call.line for s in c.signals):
                    built[call.target.split(".")[-1]].append(c)
        done = set()
        for c in self.comps:
            d = self.symbols[c.key].d
            local = set(d.args) | {call.target.split('.')[-1] for call in d.calls if call.target}
            used = set(d.names_used) - local if d.name != "<module>" else set()  # locals shadow module objects
            for var, targets in built.items():
                targets = [x for x in targets if x.key != c.key and x.type not in ("entrypoint", "prompt")]
                if var in used and targets and (c.key, var) not in done:
                    done.add((c.key, var))
                    a, b = self.best_pair(self.at[c.key], targets)
                    self.add(a, b, "calls")

    def prompt_edges(self) -> None:
        """Prompt templates feed the LLM components that use them."""
        for p in (c for c in self.comps if c.type == "prompt"):
            var = p.id.split(":", 1)[1]
            for c in self.comps:
                if c.type in ("llm", "agent", "router") and var in self.symbols[c.key].d.names_used:
                    self.add(p, c, "feeds")

    def reachable(self, a: str, b: str) -> bool:
        ca = next((c for c in self.comps if c.id == a), None)
        cb = next((c for c in self.comps if c.id == b), None)
        if not ca or not cb:
            return False
        return ca.path == cb.path or cb.key in self.g.reach(ca.key) or ca.key in self.g.reach(cb.key)


def reduce_with_llm(graph: ApplicationGraph, asm: Assembler) -> tuple[ApplicationGraph, dict]:
    stats = {"calls": 1, "added": 0, "removed": 0, "rejected": 0}
    payload = {"components": [{"id": n.id, "type": n.type, "label": n.label, "purpose": n.description} for n in graph.nodes],
               "candidate_edges": [{"id": e.id, "kind": e.kind, "label": e.label} for e in graph.edges]}
    try:
        red, _ = complete_json("pipeline_reduce", REDUCE_SYSTEM, json.dumps(payload), Reduction, max_tokens=2500)
    except LLMError as exc:
        logging.getLogger(__name__).warning("reduce step failed: %s", exc)
        return graph, {**stats, "failed": True}
    ids = {n.id for n in graph.nodes}
    edges = {e.id: e for e in graph.edges}
    for eid in red.remove_edges:
        e = edges.get(eid)
        if e is not None and e.kind in ("calls", "uses"):  # LangGraph/flow edges are facts, not opinions
            del edges[eid]
            stats["removed"] += 1
        else:
            stats["rejected"] += 1
    for p in red.add_edges:
        eid = f"{p.source}->{p.target}"
        if p.source in ids and p.target in ids and p.source != p.target and eid not in edges and asm.reachable(p.source, p.target):
            edges[eid] = GraphEdge(id=eid, source=p.source, target=p.target, kind=p.kind, label=p.label)
            stats["added"] += 1
        else:
            stats["rejected"] += 1
    return ApplicationGraph(nodes=graph.nodes, edges=list(edges.values())), stats
