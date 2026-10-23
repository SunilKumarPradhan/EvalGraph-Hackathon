"""LLM Flow Graph: where the data in every prompt comes from, and where every model output goes (prompt provenance).

A small may-flow taint analysis over the extractor's data-flow facts (Python only; JS/TS files simply have none):
 1. sources   call sites tagged with a facet produce labelled data (retriever -> retrieved, LLM -> llm_output,
              HTTP -> external, ...); entrypoint parameters and request bodies are user_input; tool parameters are
              chosen by the model (llm_output)
 2. local     flow-insensitive fixpoint over assignments, mutations (`x.append(y)`), loops and returns; a call with
              no facet passes its receiver + arguments through (string formatting, joins, helpers without code)
 3. calls     project functions get summaries (which parameters reach the return value + what they produce);
              argument labels are pushed into callee parameters; a few rounds until nothing changes
 4. sinks     LLM calls (per argument channel: system, messages, ...), tool invocations, JSON parsing, code/shell/SQL
              execution, outgoing HTTP
Everything stays generic: only facets, call names and Python semantics, never a repo-specific name.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from ..analyzer.extract import MODULE, RECV, RETURN
from ..analyzer.facts import CallFact, FileFacts
from ..core.schemas import ApplicationGraph, Evidence, GraphEdge
from .components import Component, Resolver
from .signals import FACET_TYPE, LLM_METHODS, Signal, SymbolInfo

ROUNDS = 4
LOCAL_PASSES = 8
MAX_TAINTS = 16          # per variable; may-flow sets stay small and readable

SOURCE_FACETS = {"llm": "llm_output", "agent": "llm_output", "retriever": "retrieved", "reranker": "retrieved",
                 "memory": "memory", "external_api": "external"}
RESULT_OF_COMPONENT = {"llm": "llm_output", "agent": "llm_output", "router": "llm_output", "retriever": "retrieved",
                       "reranker": "retrieved", "memory": "memory", "tool": "tool_output", "external_api": "external"}
USER_INPUT_CALLS = {"input", "chat_input", "text_input", "text_area", "get_json", "receive_text", "receive_json"}
USER_INPUT_NAMES = re.compile(r"^(request|req)\.(json|form|args|values|data|body|query_params|files)$|^sys\.argv$")
PARSE_CALLS = {"json.loads", "JSON.parse", "yaml.safe_load", "yaml.load", "ast.literal_eval"}
PARSE_LAST = {"model_validate_json", "parse_raw", "parse_obj_as"}
EXEC_CALLS = re.compile(r"^(exec|eval|compile|os\.system|os\.popen|subprocess\.(run|call|Popen|check_output|check_call)"
                        r"|pd\.eval|.*\.(execute|executescript|executemany))$")
EXEC_OWNER = re.compile(r"cursor|conn|db|session|engine|sql|cur$", re.I)
VALIDATORS = re.compile(r"valid|model_validate|parse_obj|TypeAdapter|schema|isinstance|allowlist|whitelist", re.I)
TOOL_BINDERS = {"bind_tools", "bindTools", "bind_functions", "create_react_agent", "create_tool_calling_agent",
                "create_openai_tools_agent", "AgentExecutor", "initialize_agent"}
LLM_TOOL_KWARGS = {"tools", "functions", "tool_choice", "toolChoice"}
SYSTEM_CHANNELS = {"system", "system_prompt", "systemPrompt", "instructions", "system_message", "preamble"}
PARAM = "@param"         # symbolic label: "this data came in through parameter <name>" (summaries)
KEYED = re.compile(r"^([^\[\]]+)\[([^\[\]]*)\]$")   # state["chunks"] / <return>[chunks]
STATE_ARGS = {"state"}


@dataclass(frozen=True)
class Taint:
    label: str               # user_input | retrieved | memory | tool_output | llm_output | external | prompt | tools
    comp: str | None         # component that produced it (graph node id), if any
    path: str
    line: int | None
    name: str = ""           # what it is: prompt constant name, parameter name, call

    def evidence(self, snippet: str = "") -> Evidence:
        return Evidence(file=self.path, line=self.line, snippet=snippet, fact=f"{self.label} from {self.name or self.comp}")


@dataclass
class Sink:
    kind: str                # llm | tool | parse | exec | http
    key: str                 # symbol key
    path: str
    line: int
    snippet: str
    channels: dict[str, set[Taint]] = field(default_factory=dict)   # argument -> taints
    comp: str | None = None
    target: str | None = None        # tool sinks: the tool component called
    callees: list[str] = field(default_factory=list)   # tool sinks: symbol keys of the tool functions
    in_loop: bool = False
    in_try: bool = False
    validated: bool = False          # the function validates/checks data somewhere
    has_tools: bool = False          # llm sinks: the model can call tools

    def taints(self, *labels: str) -> list[Taint]:
        found = {t for ts in self.channels.values() for t in ts if not labels or t.label in labels}
        return sorted(found, key=lambda t: (t.label, t.path, t.line or 0, t.name))

    def evidence(self, fact: str) -> Evidence:
        return Evidence(file=self.path, line=self.line, snippet=self.snippet, fact=fact)


@dataclass
class Flow:
    source: str
    target: str
    label: str
    kind: str
    via: list[Evidence]


@dataclass
class PromptRecipe:
    comp: str
    path: str
    line: int
    parts: dict[str, list[Taint]]      # label -> where it comes from
    system: list[str]                  # labels inside system-prompt channels
    has_tools: bool


@dataclass
class FlowResult:
    sinks: list[Sink] = field(default_factory=list)
    flows: list[Flow] = field(default_factory=list)
    recipes: list[PromptRecipe] = field(default_factory=list)
    functions: int = 0

    def stats(self) -> dict:
        kinds: dict[str, int] = defaultdict(int)
        for s in self.sinks:
            kinds[s.kind] += 1
        return {"functions": self.functions, "sinks": dict(kinds), "flows": len(self.flows), "recipes": len(self.recipes)}


@dataclass
class _Summary:
    returns_params: set[str] = field(default_factory=set)
    returns: set[Taint] = field(default_factory=set)
    params_in: dict[str, set[Taint]] = field(default_factory=lambda: defaultdict(set))


def _last(callee: str) -> str:
    return callee.replace("()", "").replace("[]", "").split(".")[-1]


def _cap(taints: set[Taint]) -> set[Taint]:
    if len(taints) <= MAX_TAINTS:
        return taints
    return set(sorted(taints, key=lambda t: (t.label == PARAM, t.label, t.path, t.line or 0))[:MAX_TAINTS])


class FlowAnalysis:
    def __init__(self, files: list[FileFacts], symbols: dict[str, SymbolInfo], comps: list[Component],
                 scope: set[str] | None = None):
        self.files = {f.path: f for f in files}
        self.symbols = symbols
        self.keys = [k for k in symbols if not scope or k in scope]
        self.resolver = Resolver(files, symbols)
        self.comps_at: dict[str, list[Component]] = defaultdict(list)
        for c in comps:
            self.comps_at[c.key].append(c)
        self.prompts = {p.name: (f.path, p.line) for f in files for p in f.prompts}
        self.prompts.update({c.id.split(":", 1)[1]: (c.path, min((s.line for s in c.signals), default=None))
                             for c in comps if c.type == "prompt"})
        self.summaries: dict[str, _Summary] = defaultdict(_Summary)
        self.module_env: dict[str, dict[str, set[Taint]]] = {}
        # shared graph state (LangGraph-style): nodes read state["k"] and return {"k": ...} updates
        self.state: dict[str, set[Taint]] = {}
        node_fns = {fn.split(".")[-1] for f in files for lg in f.langgraph if lg.kind == "add_node" and lg.fn
                    for fn in lg.fn.split(",")}
        self.node_fns = {k for k, info in symbols.items() if info.d.name in node_fns}
        self.state_keys: set[str] = set()     # keys some graph node reads from its state
        for k in self.keys:
            arg = self.state_arg(k)
            if arg is None:
                continue
            d = symbols[k].d
            names = [n for a in d.assigns for n in a.sources] + [n for c in d.calls for ns in c.arg_names for n in ns] \
                + [n for c in d.calls for ns in c.kwarg_names.values() for n in ns]
            self.state_keys |= {m.group(2) for n in names if (m := KEYED.match(n)) and m.group(1) == arg}
            self.state_keys |= {c.str_args[0] for c in d.calls if _last(c.callee) == "get" and c.str_args
                                and c.kwarg_names.get(RECV) == [arg]}

    def state_arg(self, key: str) -> str | None:
        args = [a for a in self.symbols[key].d.args if a not in ("self", "cls")]
        named = next((a for a in args if a in STATE_ARGS), None)
        if named:
            return named
        return args[0] if key in self.node_fns and args else None

    # ---------------------------------------------------------------- helpers
    def comp(self, key: str, line: int | None, types: tuple[str, ...]) -> Component | None:
        here = [c for c in self.comps_at.get(key, []) if c.type in types]
        on_line = [c for c in here if any(s.line == line for s in c.signals)]
        return (on_line or here or [None])[0]

    @staticmethod
    def owner(info: SymbolInfo, s: Signal) -> CallFact | None:
        """The call a signal is about. Several calls can share a line (`chain.invoke({"c": "\n".join(docs)})`):
        the one the signal names, else the receiver it names, else the outermost one."""
        same = [c for c in info.d.calls if c.line == s.line]
        if len(same) <= 1:
            return same[0] if same else None
        named = [c for c in same if c.callee and c.callee in s.fact]
        if named:
            return max(named, key=lambda c: len(c.callee))
        recv = [c for c in same if s.fact.startswith(f"uses {c.callee.split('.')[0]} ")]
        return (recv or same)[0]

    def signals_of(self, info: SymbolInfo, call: CallFact) -> list[Signal]:
        return [s for s in info.signals if s.line == call.line and not s.weak and self.owner(info, s) is call]

    @staticmethod
    def model_call(s: Signal, call: CallFact) -> bool:
        """The call runs a model (not: builds a client, wires a graph, registers a node)."""
        if "used implicitly" in s.fact or _last(call.callee) in TOOL_BINDERS:
            return False
        if s.facet == "llm":
            return True
        return s.facet == "agent" and (s.fact.startswith("runs ") or "loop around the LLM" in s.fact
                                       or "function calling" in s.fact
                                       or (s.fact.startswith("uses ") and _last(call.callee) in LLM_METHODS))

    def facet_at(self, info: SymbolInfo, call: CallFact) -> str | None:
        """The strongest source facet a call site carries (direct evidence only)."""
        best, weight = None, 0.0
        for s in self.signals_of(info, call):
            if s.facet in SOURCE_FACETS and s.weight > weight and (s.facet not in ("llm", "agent") or self.model_call(s, call)):
                best, weight = s.facet, s.weight
        return best

    def is_llm_call(self, info: SymbolInfo, call: CallFact) -> bool:
        return any(s.facet in ("llm", "agent") and s.weight >= 0.6 and self.model_call(s, call)
                   for s in self.signals_of(info, call))

    def targets(self, info: SymbolInfo, call: CallFact) -> tuple[list[str], bool]:
        """Project functions a call reaches, and whether it is a registry dispatch (TOOLS[name](...))."""
        bare = call.callee.replace("()", "")
        if bare.endswith("[]") and "." not in bare[:-2]:
            return self.resolver.registry_targets(info.path, bare[:-2]), True
        if "." not in bare and bare not in self.resolver.by_name:  # fn = TOOLS[name]; fn(**args)
            regs = {s.split("[")[0] for a in info.d.assigns if a.target == bare for s in a.sources} & set(self.resolver.registries)
            if regs:
                return [t for r in sorted(regs) for t in self.resolver.registry_targets(info.path, r)], True
        if "." in bare.replace("self.", "", 1) and not bare.startswith("self."):
            parts = bare.split(".")
            # obj.method(): only bind when the receiver looks like a module/class name, not any local variable
            if parts[-2] not in self.resolver.imported[info.path] and not parts[-2][:1].isupper():
                return [], False
        return [t for t in self.resolver.call_targets(info.path, call.callee) if t in self.symbols], False

    # ---------------------------------------------------------------- one function
    def analyze(self, key: str, record: list[Sink] | None = None) -> dict[str, set[Taint]]:
        info = self.symbols[key]
        d = info.d
        env: dict[str, set[Taint]] = defaultdict(set)
        summary = self.summaries[key]
        entry = self.comp(key, None, ("entrypoint",))
        tool = self.comp(key, None, ("tool",))
        for a in d.args:
            if a in ("self", "cls"):
                continue
            env[a].add(Taint(PARAM, None, info.path, d.line, a))
            env[a] |= summary.params_in.get(a, set())
            if entry is not None and d.name != MODULE:
                env[a].add(Taint("user_input", entry.id, info.path, d.line, f"parameter {a} of {d.name}()"))
            if tool is not None:
                env[a].add(Taint("llm_output", None, info.path, d.line, f"tool argument {a} (chosen by the model)"))
        module = self.module_env.get(info.path, {})

        state_arg = self.state_arg(key)

        def lookup(name: str) -> set[Taint]:
            m = KEYED.match(name)
            if m:  # d["k"]: the whole object, plus that key of the shared graph state when d is the state
                base, k = m.groups()
                found = set(env.get(name, set())) | lookup(base)
                if base == state_arg:
                    found |= self.state.get(k, set())
                return found
            parts = name.split(".")
            prompt: set[Taint] = set()
            if parts[0] in self.prompts:
                path, line = self.prompts[parts[0]]
                prompt = {Taint("prompt", None, path, line, parts[0])}
            for i in range(len(parts), 0, -1):
                prefix = ".".join(parts[:i])
                if prefix in env:
                    return env[prefix] | prompt
                if prefix in module and d.name != MODULE:
                    return module[prefix] | prompt
            if USER_INPUT_NAMES.match(name):
                return {Taint("user_input", entry.id if entry else None, info.path, d.line, name)}
            return prompt

        results: list[set[Taint]] = [set() for _ in d.calls]

        def refs(names: list[str], calls: list[int]) -> set[Taint]:
            out: set[Taint] = set()
            for n in names:
                out |= lookup(n)
            for i in calls:
                if 0 <= i < len(results):
                    out |= results[i]
            return out

        def args_of(call: CallFact) -> dict[str, set[Taint]]:
            ch: dict[str, set[Taint]] = {}
            for i, (names, calls) in enumerate(zip(call.arg_names, call.arg_calls)):
                ch[f"arg{i}"] = refs(names, calls)
            for k in set(call.kwarg_names) | set(call.kwarg_calls):
                ch[k] = refs(call.kwarg_names.get(k, []), call.kwarg_calls.get(k, []))
            return ch

        def result(call: CallFact) -> set[Taint]:
            ch = args_of(call)
            last = _last(call.callee)
            facet = self.facet_at(info, call)
            if last in TOOL_BINDERS:
                return set().union(*ch.values()) | {Taint("tools", None, info.path, call.line, call.callee)}
            if facet is not None:
                c = self.comp(key, call.line, (FACET_TYPE[facet],) + (("agent", "llm", "router", "guardrail") if facet in ("llm", "agent") else ()))
                return {Taint(SOURCE_FACETS[facet], c.id if c else None, info.path, call.line, call.callee)}
            if last in USER_INPUT_CALLS and (last != "input" or call.callee == "input"):
                return {Taint("user_input", entry.id if entry else None, info.path, call.line, call.callee)}
            if last == "get" and call.str_args and call.kwarg_names.get(RECV) == [state_arg]:
                return lookup(f"{state_arg}[{call.str_args[0]}]")
            targets, _ = self.targets(info, call)
            if targets:
                out: set[Taint] = set()
                for t in targets:
                    memory = next((c for c in self.comps_at.get(t, []) if c.type == "memory"), None)
                    if memory is not None:  # history is a boundary: past turns are "memory", not their origins
                        out.add(Taint("memory", memory.id, info.path, call.line, call.callee))
                        continue
                    s = self.summaries[t]
                    out |= s.returns
                    params = [p for p in self.symbols[t].d.args if p not in ("self", "cls")]
                    for p in s.returns_params:
                        out |= bound(call, ch, params, p)
                    # what a tool returns is always tool output; other components only label their result when
                    # their own code does not show where it comes from
                    real = any(x.label != PARAM for x in s.returns)
                    for c in self.comps_at.get(t, []):
                        if c.type == "tool" or (c.type in RESULT_OF_COMPONENT and not real):
                            out.add(Taint(RESULT_OF_COMPONENT[c.type], c.id, info.path, call.line, call.callee))
                return out
            return set().union(*ch.values()) if ch else set()

        def bound(call: CallFact, ch: dict[str, set[Taint]], params: list[str], p: str) -> set[Taint]:
            out = set(ch.get(p, set())) | ch.get("**", set())
            if p in params and params.index(p) < len(call.arg_names):
                out |= ch.get(f"arg{params.index(p)}", set())
            return out

        for _ in range(LOCAL_PASSES):
            changed = False
            for i, call in enumerate(d.calls):
                new = _cap(result(call))
                if new != results[i]:
                    results[i], changed = new, True
            for a in d.assigns:
                new = _cap(env[a.target] | refs(a.sources, a.calls))
                if new != env[a.target]:
                    env[a.target], changed = new, True
            if not changed:
                break

        # summary: which parameters reach the return value, what it produces; push argument labels into callees
        ret = env.get(RETURN, set())
        summary.returns_params = {t.name for t in ret if t.label == PARAM and t.path == info.path and t.line == d.line}
        # what callers passed in comes back through returns_params at each call site, per caller; keeping it in
        # `returns` would hand every caller's data to every other caller of a shared helper
        incoming = set().union(*summary.params_in.values()) if summary.params_in else set()
        summary.returns = {t for t in ret if t.label != PARAM and t not in incoming}
        # graph nodes return state updates ({"chunks": retrieve(...)}); so does whatever builds the initial state
        for target, taints in env.items():
            m = KEYED.match(target)
            if m and m.group(1) == RETURN and (state_arg is not None or m.group(2) in self.state_keys):
                self.state[m.group(2)] = _cap(self.state.get(m.group(2), set()) | {t for t in taints if t.label != PARAM})
        for call in d.calls:
            targets, _ = self.targets(info, call)
            if not targets:
                continue
            ch = args_of(call)
            for t in targets:
                params = [p for p in self.symbols[t].d.args if p not in ("self", "cls")]
                for p in params:
                    real = {x for x in bound(call, ch, params, p) if x.label != PARAM}
                    if real:
                        self.summaries[t].params_in[p] |= real
        if record is not None:
            validated = any(VALIDATORS.search(_last(c.callee)) for c in d.calls)
            for call in d.calls:
                sink = self.sink(info, call, args_of(call), validated)
                if sink is not None:
                    record.append(sink)
        return env

    def sink(self, info: SymbolInfo, call: CallFact, ch: dict[str, set[Taint]], validated: bool) -> Sink | None:
        real = {k: {t for t in v if t.label != PARAM} for k, v in ch.items()}
        real = {k: v for k, v in real.items() if v}
        base = dict(key=info.key, path=info.path, line=call.line, snippet=call.snippet, in_loop=call.in_loop,
                    in_try=call.in_try, validated=validated)
        last, bare = _last(call.callee), call.callee.replace("()", "")
        if self.is_llm_call(info, call):
            c = self.comp(info.key, call.line, ("llm", "agent", "router", "guardrail", "parser"))
            tools = bool(LLM_TOOL_KWARGS & set(call.kwargs)) or any(t.label == "tools" for t in real.get(RECV, set()))
            return Sink("llm", channels=real, comp=c.id if c else None, has_tools=tools, **base)
        if bare in PARSE_CALLS or last in PARSE_LAST:
            c = self.comp(info.key, call.line, ("parser", "router", "llm", "agent", "tool"))
            return Sink("parse", channels=real, comp=c.id if c else None, **base)
        if EXEC_CALLS.match(bare) and ("." not in bare or not last.startswith("execute") or EXEC_OWNER.search(bare.split(".")[-2])):
            c = self.comp(info.key, call.line, ("tool", "agent", "llm", "parser", "router", "external_api", "entrypoint"))
            return Sink("exec", channels=real, comp=c.id if c else None, **base)
        targets, registry = self.targets(info, call)
        tools = [c for t in targets for c in self.comps_at.get(t, []) if c.type == "tool"]
        if tools:
            c = self.comp(info.key, call.line, ("agent", "llm", "router", "tool", "entrypoint"))
            # the arguments are checked if the dispatcher or every tool it reaches validates them
            checked = validated or all(any(VALIDATORS.search(_last(x.callee)) for x in self.symbols[t].d.calls) for t in targets)
            base["validated"] = checked
            return Sink("tool", channels=real, comp=c.id if c else None, callees=targets,
                        target=tools[0].id if not registry or len(tools) == 1 else None, **base)
        if any(s.line == call.line and s.facet == "external_api" and not s.weak for s in info.signals):
            c = self.comp(info.key, call.line, ("external_api", "tool"))
            return Sink("http", channels=real, comp=c.id if c else None, **base)
        return None

    # ---------------------------------------------------------------- whole repo
    def run(self) -> FlowResult:
        modules = [k for k in self.keys if self.symbols[k].d.name == MODULE]
        functions = [k for k in self.keys if self.symbols[k].d.name != MODULE]
        def snapshot():
            return ({k: (frozenset(s.returns), frozenset(s.returns_params),
                         frozenset((p, frozenset(v)) for p, v in s.params_in.items())) for k, s in self.summaries.items()},
                    {k: frozenset(v) for k, v in self.state.items()})

        for _ in range(ROUNDS):
            before = snapshot()
            for k in modules:  # import-time code first: module variables (clients, prompts, chains) feed the functions
                self.module_env[self.symbols[k].path] = self.analyze(k)
            for k in functions:
                self.analyze(k)
            if snapshot() == before:
                break
        sinks: list[Sink] = []
        for k in modules + functions:
            self.analyze(k, record=sinks)
        out = FlowResult(sinks=[s for s in sinks if s.channels or s.kind == "llm"], functions=len(self.keys))
        out.flows = self.flows(out.sinks)
        out.recipes = self.recipes(out.sinks)
        return out

    def flows(self, sinks: list[Sink]) -> list[Flow]:
        seen: dict[tuple[str, str, str], Flow] = {}
        for s in sinks:
            dst = s.target if s.kind == "tool" and s.target else s.comp
            if dst is None:
                continue
            for t in s.taints():
                if t.comp is None or t.comp == dst or t.label in ("prompt", "tools"):
                    continue
                key = (t.comp, dst, t.label)
                if key not in seen:
                    seen[key] = Flow(t.comp, dst, t.label, s.kind, [t.evidence(), s.evidence(f"{s.kind} sink {s.snippet[:60]}")])
        return list(seen.values())

    def recipes(self, sinks: list[Sink]) -> list[PromptRecipe]:
        """One recipe per LLM component: every model call it makes, merged."""
        out: dict[str, PromptRecipe] = {}
        for s in sinks:
            if s.kind != "llm" or s.comp is None:
                continue
            r = out.setdefault(s.comp, PromptRecipe(s.comp, s.path, s.line, {}, [], False))
            for t in s.taints():
                if t.label != "tools" and t not in r.parts.setdefault(t.label, []):
                    r.parts[t.label].append(t)
            r.parts = {k: v for k, v in r.parts.items() if v}
            r.system = sorted(set(r.system) | {t.label for k, ts in s.channels.items() if k in SYSTEM_CHANNELS for t in ts})
            r.has_tools |= s.has_tools
        return list(out.values())


def analyze_flows(files: list[FileFacts], symbols: dict[str, SymbolInfo], comps: list[Component],
                  scope: set[str] | None = None) -> FlowResult:
    return FlowAnalysis(files, symbols, comps, scope).run()


# a flow with no static edge yet becomes a "feeds" edge only for these (user input and model output reach almost
# everything, as edges they would only clutter the canvas)
NEW_EDGE = {"retrieved", "memory", "tool_output", "external"}
NEW_EDGE_FROM_LLM = {"parser", "tool", "external_api"}


def annotate_graph(graph: ApplicationGraph, flow: FlowResult) -> int:
    """Put proven data flows on the graph: label existing edges, add missing `feeds` edges. Returns edges added."""
    types = {n.id: n.type for n in graph.nodes}
    edges = {(e.source, e.target): e for e in graph.edges}
    added = 0
    for f in flow.flows:
        if f.source not in types or f.target not in types:
            continue
        e = edges.get((f.source, f.target))
        if e is None and (f.label in NEW_EDGE or (f.label == "llm_output" and types[f.target] in NEW_EDGE_FROM_LLM)):
            e = GraphEdge(id=f"{f.source}->{f.target}", source=f.source, target=f.target, kind="feeds")
            edges[(f.source, f.target)] = e
            graph.edges.append(e)
            added += 1
        if e is not None:
            e.flow_labels = sorted(set(e.flow_labels or []) | {f.label})
    return added
