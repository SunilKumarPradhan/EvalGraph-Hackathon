"""UNDERSTAND (static half, continued): from tagged symbols to pipeline components.

Generic rules, applied in order:
 1. symbol graph   calls (name + alias resolution) + function references (callbacks, LangGraph add_node)
                   + registry dispatch (TOOLS[name](...))
 2. delegation     a call into a project function is not a library signal
 3. infra          classes that construct an LLM client are plumbing; their callers are the components
 4. entrypoints    keep entrypoints that reach AI facets (web/UI > CLI > scripts); prune unreachable code
 5. factories      a factory used by one component folds into it; shared factories stay as infrastructure
 6. absorption     router/guardrail absorb their own LLM call; parsers only stand alone without one
 7. ranking        personalized PageRank seeded by signal weight (budgeting + ordering)
 8. facet split    each remaining facet of a symbol becomes a component
"""

from __future__ import annotations

import re
from collections import defaultdict, deque
from dataclasses import dataclass, field

from ..analyzer.extract import MODULE
from ..analyzer.facts import FileFacts
from ..core.schemas import ConfigValue, Evidence
from .signals import FACET_TYPE, LLM_CTORS, Signal, SymbolInfo

MAX_COMPONENTS = 32
CTOR_FACT = re.compile(r"^(creates LLM client|vector store|embeddings via [A-Z]|reranks via [A-Z]|conversation memory via [A-Z])")
NON_APP = re.compile(r"(^|/)(tests?|eval|evals|scripts|docs|examples?|notebooks?|benchmarks?|migrations|experiments?|sandbox)/|(^|/)test_[^/]+$|conftest\.py$|(^|/)bench[^/]*\.py$")
ROLE_NAME = {"retriever": "Retriever", "vector_db": "Vector Store", "embedding": "Embeddings", "dataset": "Ingestion",
             "llm": "LLM", "agent": "Agent", "memory": "Memory", "entrypoint": "Entry", "router": "Router",
             "reranker": "Reranker", "tool": "Tool", "external_api": "API", "guardrail": "Guardrail", "parser": "Parser",
             "prompt": "Prompt"}
ENTRY_RANK = {"web route @app.post": 0, "web route @router.post": 0, "web route @api.post": 0, "Next.js route": 0,
              "web route @": 1, "user interface": 1, "serverless": 1, "script entry": 3}


@dataclass
class Component:
    id: str
    type: str
    facet: str
    key: str            # symbol key "path::qualname"
    path: str
    qualname: str
    label: str
    signals: list[Signal]
    rank: float = 0.0
    description: str | None = None
    confidence: float = 0.8

    @property
    def symbol(self) -> str:
        return self.qualname.split(".")[-1]

    @property
    def evidence(self) -> list[Evidence]:
        sigs = sorted(self.signals, key=lambda s: -s.weight)[:4]
        return [Evidence(file=self.path, line=s.line, snippet=s.snippet, fact=s.fact) for s in sigs]

    @property
    def params(self) -> dict[str, ConfigValue]:
        out: dict[str, ConfigValue] = {}
        for s in self.signals:
            out.update(s.params)
        return out

    @property
    def provider(self) -> str | None:
        return next((s.provider for s in self.signals if s.provider), None)


@dataclass
class SymbolGraph:
    out: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    inc: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    def add(self, a: str, b: str) -> None:
        if a != b:
            self.out[a].add(b)
            self.inc[b].add(a)

    def reach(self, start: str, limit: int = 10_000) -> set[str]:
        seen, queue = {start}, deque([start])
        while queue and len(seen) < limit:
            for nxt in self.out.get(queue.popleft(), ()):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        return seen


def humanize(name: str, facet: str) -> str:
    if name == MODULE:
        return {"entrypoint": "App Entrypoint"}.get(facet, f"Module {FACET_TYPE.get(facet, facet).title()}")
    words = re.sub(r"([a-z])([A-Z])", r"\1 \2", name).replace("_", " ").split()
    return " ".join(w.capitalize() for w in words) or name


class Resolver:
    """Name resolution from a call or reference to project symbols: same file first, then the module named by the
    receiver, then imported names. Method-name matching, so dynamic dispatch can over-connect (conf_02 §2.4)."""

    def __init__(self, files: list[FileFacts], symbols: dict[str, SymbolInfo]):
        self.by_name: dict[str, list[str]] = defaultdict(list)
        for key, info in symbols.items():
            self.by_name[info.d.name].append(key)
        self.imported: dict[str, set[str]] = defaultdict(set)   # path -> names imported into it
        for f in files:
            for imp in f.imports:
                self.imported[f.path].update(imp.names)
                self.imported[f.path].add(imp.module.rsplit(".", 1)[-1].rsplit("/", 1)[-1])
        self.registries = {r.name: r for f in files for r in f.registries}

    @staticmethod
    def stem(key: str) -> str:
        return key.split("::")[0].rsplit("/", 1)[-1].rsplit(".", 1)[0]

    def resolve(self, path: str, name: str, receiver: str | None = None) -> list[str]:
        targets = self.by_name.get(name, [])
        if receiver:  # module.func(): prefer the file named like the module
            named = [t for t in targets if self.stem(t) == receiver]
            if named:
                return named
        same = [t for t in targets if t.startswith(path + "::")]
        return same or targets

    def resolve_ref(self, path: str, name: str) -> list[str]:
        """Bare references (callbacks, add_node(fn), tools=[fn]) only bind to local or imported functions."""
        targets = self.by_name.get(name, [])
        same = [t for t in targets if t.startswith(path + "::")]
        if same:
            return same
        return targets if name in self.imported[path] else []

    def call_targets(self, path: str, callee: str) -> list[str]:
        parts = [p for p in callee.replace("()", "").replace("[]", "").split(".") if p]
        if not parts:
            return []
        if len(parts) == 1:
            return self.resolve(path, parts[0])
        return self.resolve(path, parts[-1], receiver=parts[-2]) + self.resolve_ref(path, parts[0])

    def registry_targets(self, path: str, name: str) -> list[str]:
        """TOOLS[key](...): every function registered in TOOLS."""
        reg = self.registries.get(name)
        if reg is None:
            return []
        return [t for fn in reg.entries.values() for t in (self.resolve_ref(path, fn) or self.resolve(path, fn))]


def build_symbol_graph(files: list[FileFacts], symbols: dict[str, SymbolInfo], leaves: set[str] = frozenset()) -> SymbolGraph:
    r = Resolver(files, symbols)
    g = SymbolGraph()
    for key, info in symbols.items():
        if key in leaves:
            continue
        for call in info.d.calls:
            for t in r.call_targets(info.path, call.callee):
                g.add(key, t)
        if info.d.name != MODULE:  # module-level code defines things; only its calls are flow
            for name in info.d.names_used:
                for t in r.resolve_ref(info.path, name):
                    g.add(key, t)
        for name in info.d.subscripted:          # TOOLS[key](...)
            for t in r.registry_targets(info.path, name):
                g.add(key, t)
    return g


def _drop_delegated(symbols: dict[str, SymbolInfo]) -> None:
    by_name: dict[str, list[SymbolInfo]] = defaultdict(list)
    for info in symbols.values():
        by_name[info.d.name].append(info)
    for info in symbols.values():
        keep = []
        for s in info.signals:
            # several calls can share a line ("\n\n".join(retrieve(q))): use the one the signal names
            same_line = [c for c in info.d.calls if c.line == s.line and s.snippet and c.snippet == s.snippet]
            call = next((c for c in same_line if c.callee and s.fact.endswith(c.callee)), same_line[0] if same_line else None)
            if call is not None and s.weight >= 0.8:
                bare = call.callee.replace("()", "")
                if "." not in bare and by_name.get(bare):  # plain call of a project function
                    continue
            keep.append(s)
        info.signals = keep


def _infra(files: list[FileFacts], symbols: dict[str, SymbolInfo]) -> set[str]:
    classes = set()
    for f in files:
        for d in f.defs:
            if d.kind == "class" and any(c.callee.replace("()", "").split(".")[-1] in LLM_CTORS for c in d.calls):
                classes.add((f.path, d.qualname))
    return {k for k, info in symbols.items() if (info.path, info.d.qualname.split(".")[0]) in classes}


def _entry_kind(info: SymbolInfo) -> int | None:
    best = None
    for s in info.signals:
        if s.facet == "entrypoint":
            r = next((v for k, v in ENTRY_RANK.items() if s.fact.startswith(k)), 2)
            best = r if best is None else min(best, r)
    return best


def pagerank(g: SymbolGraph, seeds: dict[str, float], nodes: list[str], iters: int = 40, alpha: float = 0.85) -> dict[str, float]:
    if not nodes:
        return {}
    total = sum(seeds.values()) or 1.0
    pers = {n: seeds.get(n, 0.0) / total for n in nodes}
    if not any(pers.values()):
        pers = {n: 1 / len(nodes) for n in nodes}
    rank = dict(pers)
    node_set = set(nodes)
    for _ in range(iters):
        nxt = {n: (1 - alpha) * pers[n] for n in nodes}
        for n in nodes:
            # importance flows both ways: callers of AI code and the code AI calls both matter
            nbrs = [m for m in g.out.get(n, ()) if m in node_set] + [m for m in g.inc.get(n, ()) if m in node_set]
            if not nbrs:
                for m in nodes:
                    nxt[m] += alpha * rank[n] * pers[m]
                continue
            share = alpha * rank[n] / len(nbrs)
            for m in nbrs:
                nxt[m] += share
        rank = nxt
    return rank


def build_components(files: list[FileFacts], symbols: dict[str, SymbolInfo]) -> tuple[list[Component], SymbolGraph, dict]:
    stats: dict = {}
    app = {k: v for k, v in symbols.items() if not NON_APP.search(v.path)}
    _drop_delegated(app)
    infra = _infra(files, app)
    for k in infra:
        app[k].signals = []
    g = build_symbol_graph(files, symbols, leaves=infra)

    facets = {k: info.facets() for k, info in app.items()}

    def direct(k: str, facet: str) -> bool:
        return any(s.facet == facet and not s.weak for s in app[k].signals)

    def weak_count(k: str, facet: str) -> int:
        return len({s.fact.split(" ")[0] for s in app[k].signals if s.facet == facet and s.weak})

    # weak-only facets need two independent hints (e.g. name + referenced constant), never one
    for k in facets:
        facets[k] = [f for f in facets[k] if direct(k, f) or weak_count(k, f) >= 2 or f == "entrypoint"]
    # a facet that only comes from using an object constructed elsewhere belongs to the construction site
    bindings: list[tuple[str, str, str]] = []
    for k in facets:
        for f in list(facets[k]):
            own = [s for s in app[k].signals if s.facet == f and not s.weak]
            vias = {s.via for s in own if s.via}
            if own and all(s.via for s in own) and all(v in facets and f in facets[v] and v != k for v in vias):
                facets[k].remove(f)
                bindings += [(k, v, f) for v in vias]
    stats["bindings"] = bindings
    for user, via, _ in bindings:  # using an object is a dependency on where it was built
        g.add(user, via)
    ai_keys = {k for k, f in facets.items() if any(x != "entrypoint" for x in f)}

    # entrypoints that actually reach AI code; best kind wins
    entries = []
    for k, info in app.items():
        kind = _entry_kind(info)
        if kind is not None and "entrypoint" in facets[k]:
            reach = g.reach(k)
            hits = len(reach & ai_keys)
            if hits:
                entries.append((kind, -hits, k, reach))
    entries.sort()
    if entries:
        best_kind = entries[0][0]
        # every web/UI kind that reaches AI code counts (POST + websocket + GET ...); scripts only if nothing better
        limit = 1 if best_kind < 3 else 3
        entries = [e for e in entries if e[0] <= limit][:5]
        reachable = set().union(*(e[3] for e in entries))
        kept_entries = {e[2] for e in entries}
    else:
        reachable, kept_entries = set(app), set()
    # import-time code of every file on a request path runs too (clients, indexes, settings)
    files_on_path = {app[k].path for k in reachable if k in app}
    for k, info in app.items():
        if info.d.name == MODULE and info.path in files_on_path:
            reachable |= {k} | g.reach(k)
    stats["entrypoints"] = sorted(kept_entries)
    for k in app:
        if k not in reachable:
            facets[k] = []
        elif k not in kept_entries and "entrypoint" in facets[k]:
            facets[k] = [f for f in facets[k] if f != "entrypoint"]

    # factories: ctor-only facet holders folded into their single same-facet user
    live = {k for k, f in facets.items() if f}
    for k in list(live):
        info = app[k]
        for facet in list(facets[k]):
            fs = [s for s in info.signals if s.facet == facet]
            if not fs or not all(CTOR_FACT.match(s.fact) for s in fs):
                continue
            users = {u for u in g.inc.get(k, ()) if u in live and u != k}
            same = [u for u in users if facet in facets.get(u, [])]
            if len(users) == 1 and same:
                facets[k] = [f for f in facets[k] if f != facet]
                app[same[0]].signals += [Signal(s.facet, 0.2, s.line, s.snippet, s.fact, s.provider, s.params) for s in fs]
                app[same[0]].signals[-1].fact = f"{fs[0].fact} (in helper {info.d.name})"
    for k in list(live):  # a caller whose facet is only delegated to a shared factory loses it
        info = app[k]
        for facet in list(facets[k]):
            if facet not in ("vector_db", "embedding", "reranker"):
                continue
            callees = [c for c in g.out.get(k, ()) if facet in facets.get(c, [])]
            own = [s for s in info.signals if s.facet == facet and not any(
                call.line == s.line and call.callee.split("(")[0] in {app[c].d.name for c in callees}
                for call in info.d.calls)]
            if callees and not own:
                facets[k] = [f for f in facets[k] if f != facet]

    # thin wrappers: weak-only facet delegated to a project function with direct evidence of it
    for k in list(live):
        for facet in list(facets[k]):
            if not direct(k, facet) and any(facet in facets.get(c, []) and direct(c, facet) for c in g.out.get(k, ())):
                facets[k] = [f for f in facets[k] if f != facet]
    # per-file dedupe: weak-only symbols fold into a direct-evidence component of the same facet in that file
    for k in list(live):
        for facet in list(facets[k]):
            if direct(k, facet):
                continue
            peers = [o for o in live if o != k and app[o].path == app[k].path and facet in facets.get(o, []) and direct(o, facet)]
            if peers:
                facets[k] = [f for f in facets[k] if f != facet]

    # writers into a shared store are ingestion, not a second vector DB
    for k in list(live):
        if "vector_db" in facets[k]:
            sigs = [s for s in app[k].signals if s.facet == "vector_db"]
            writes = all(s.fact.startswith(("writes vectors", "builds index")) for s in sigs)
            if writes and any("vector_db" in facets.get(c, []) for c in g.out.get(k, ())):
                facets[k] = [f for f in facets[k] if f != "vector_db"] + (["ingest"] if "ingest" not in facets[k] else [])
                app[k].signals += [Signal("ingest", s.weight, s.line, s.snippet, s.fact) for s in sigs]
    # thin runners of a graph/agent builder fold into the builder
    for k in list(live):
        if facets[k] == ["agent"] and all(s.fact.startswith("runs ") for s in app[k].signals if s.facet == "agent"):
            if any("agent" in facets.get(c, []) for c in g.out.get(k, ()) if c != k):
                facets[k] = []
    # private helpers (_name) used by exactly one component fold into it
    for k in list(live):
        info = app[k]
        if info.d.name.startswith("_") and facets[k] and not set(facets[k]) & {"tool", "entrypoint"}:
            users = [u for u in g.inc.get(k, ()) if facets.get(u)]
            if len(users) == 1:
                facets[k] = []

    # absorption
    for k, fs in facets.items():
        if "agent" in fs and "tool" in fs and any(s.fact.startswith("builds agent/graph") for s in app[k].signals):
            fs.remove("tool")  # ToolNode(...) inside a graph builder is wiring; the tools are their own components
        if "llm" in fs and ("router" in fs or "guardrail" in fs):
            fs.remove("llm")
        if "parser" in fs and "llm" in fs:
            fs.remove("parser")
        if "agent" in fs and "llm" in fs and app[k].weight("agent") >= app[k].weight("llm"):
            fs.remove("llm")

    # class collapse: a wrapper class whose methods are all storage/retrieval/ingestion facets is one component
    INFRA = {"vector_db", "embedding", "retriever", "ingest", "reranker"}
    by_class: dict[tuple[str, str], list[str]] = defaultdict(list)
    for k, fs in facets.items():
        q = app[k].d.qualname
        if fs and "." in q:
            by_class[(app[k].path, q.split(".")[0])].append(k)
    stats["collapsed"] = []
    targets: dict[str, list[tuple[str, str]]] = {}   # symbol key -> functions to probe at runtime
    stats["probe_targets"] = targets
    for (path, cls), keys in by_class.items():
        kinds = {f for k in keys for f in facets[k]}
        if len(keys) >= 2 and kinds <= INFRA:
            main = max(keys, key=lambda k: sum(s.weight for s in app[k].signals))
            dominant = max(kinds, key=lambda f: sum(app[k].weight(f) for k in keys))
            for k in keys:
                if k != main:
                    app[main].signals += [s for s in app[k].signals if s.facet in kinds]
                    facets[k] = []
            facets[main] = [dominant]
            app[main].signals.append(Signal(dominant, 0.1, app[main].d.line, app[main].d.snippet, f"wrapper class {cls}"))
            stats["collapsed"].append(f"{path}::{cls}")
            targets[main] = [(app[k].path, app[k].d.qualname) for k in keys]
            app[main].d = app[main].d.model_copy(update={"name": cls, "qualname": cls})

    # plugin grouping: 3+ classes in one package implementing the same method with the same role are one component
    groups: dict[tuple[str, str, tuple], list[str]] = defaultdict(list)
    for k, fs in facets.items():
        q = app[k].d.qualname
        if fs and "." in q:
            groups[(app[k].path.rsplit("/", 1)[0], app[k].d.name, tuple(sorted(fs)))].append(k)
    stats["plugins"] = []
    for (pkg, method, fs), keys in groups.items():
        if len(keys) >= 3:
            main = sorted(keys)[0]
            for k in keys[1:] if keys[0] == main else [x for x in keys if x != main]:
                app[main].signals += [Signal(s.facet, 0.05, s.line, s.snippet, f"{app[k].d.qualname.split('.')[0]}: {s.fact}")
                                      for s in app[k].signals if s.facet in fs and not s.weak][:2]
                facets[k] = []
            classes = sorted(app[k].d.qualname.split(".")[0] for k in keys)
            targets[main] = [(app[k].path, app[k].d.qualname) for k in keys]
            app[main].signals.append(Signal(fs[0], 0.1, app[main].d.line, "", f"{len(keys)} plugin implementations: {', '.join(classes)}"))
            app[main].d = app[main].d.model_copy(update={"name": f"{method}_plugins", "qualname": f"{method}_plugins"})
            stats["plugins"].append(f"{pkg}::{method} x{len(keys)}")

    # package grouping: many same-type side components (fetchers, loaders) in one package are one node
    by_pkg: dict[tuple[str, str], list[str]] = defaultdict(list)
    for k, fs in facets.items():
        if fs == ["external_api"] or fs == ["ingest"]:
            by_pkg[(app[k].path.rsplit("/", 1)[0], fs[0])].append(k)
    for (pkg, facet), keys in by_pkg.items():
        if len(keys) >= 3:
            main = sorted(keys)[0]
            for k in keys:
                if k != main:
                    app[main].signals += [Signal(s.facet, 0.05, s.line, s.snippet, f"{app[k].d.qualname}: {s.fact}")
                                          for s in app[k].signals if s.facet == facet and not s.weak][:1]
                    facets[k] = []
            name = pkg.rsplit("/", 1)[-1] or "app"
            app[main].signals.append(Signal(facet, 0.1, app[main].d.line, "", f"{len(keys)} {facet.replace('_', ' ')} functions in {pkg}/"))
            targets[main] = [(app[k].path, app[k].d.qualname) for k in keys]
            app[main].d = app[main].d.model_copy(update={"name": f"{name}_{'apis' if facet == 'external_api' else 'ingestion'}",
                                                         "qualname": f"{name}_{'apis' if facet == 'external_api' else 'ingestion'}"})

    seeds = {k: sum(s.weight for s in app[k].signals if s.facet in FACET_TYPE) for k in app if facets[k]}
    ranks = pagerank(g, seeds, list(app))

    comps: list[Component] = []
    for k, fs in facets.items():
        info = app[k]
        for facet in fs:
            sigs = [s for s in info.signals if s.facet == facet] or info.signals
            if facet in ("llm", "external_api", "retriever") and info.has("retry"):
                sigs = sigs + [s for s in info.signals if s.facet in ("retry",)]
            name = info.d.name
            if name == MODULE:  # module-level component: name it after the variable it is assigned to, else the file
                stem = info.path.rsplit("/", 1)[-1].split(".")[0]
                line = min(s.line for s in sigs)
                prior = [c for c in info.d.calls if c.target and c.line <= line]
                target = max(prior, key=lambda c: c.line).target if prior and facet != "entrypoint" else None
                name = target.split(".")[-1] if target else stem
            comps.append(Component(
                id=f"{FACET_TYPE[facet]}:{name}",
                type=FACET_TYPE[facet], facet=facet, key=k, path=info.path, qualname=info.d.qualname,
                label=humanize(name, facet), signals=sigs, rank=ranks.get(k, 0.0),
                confidence=0.9 if max((s.weight for s in sigs), default=0) >= 0.8 else 0.6))
    # facets split from one function get role-first labels ("Retriever · Answer", "Vector Store · Answer")
    per_key: dict[str, int] = defaultdict(int)
    for c in comps:
        per_key[c.key] += 1
    for c in comps:
        if per_key[c.key] > 1:
            c.label = f"{ROLE_NAME.get(c.type, c.type.title())} · {c.label}"
    comps.sort(key=lambda c: -c.rank)
    comps = comps[:MAX_COMPONENTS]
    seen: dict[str, int] = defaultdict(int)
    for c in comps:  # unique ids
        seen[c.id] += 1
        if seen[c.id] > 1:
            c.id = f"{c.id}~{c.path.rsplit('/', 1)[-1].split('.')[0]}"
    stats["scope"] = {k for k in reachable if k in app and k not in infra}
    stats["ranks"] = ranks
    stats["symbols"] = len(symbols)
    stats["ai_symbols"] = len(ai_keys)
    stats["pruned"] = len([k for k in ai_keys if k not in reachable])
    return comps, g, stats
