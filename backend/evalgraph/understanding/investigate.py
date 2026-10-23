"""RECOMMEND (LLM half, PHASE-11): a bounded investigator instead of fixed batches.

The model starts from the AI Map (whole app in a few thousand tokens), then pulls exactly the code it needs with
read-only tools (open a component, read lines, search, callers/callees, data flows, config values). It finishes
with verdicts on the static findings, new findings from the taxonomy, and evaluator proposals with probe questions.

Verification is the same rule as everywhere else, generalised: a cited line counts only if a tool showed it to the
model in this session. Budget: EVALGRAPH_INVESTIGATOR_TURNS turns of up to 4 tool calls. Any failure returns None
and the caller falls back to the batch review, so an analysis never depends on it.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..analyzer.facts import FileFacts
from ..core.schemas import ApplicationGraph
from ..evaluators.catalog import BY_ID, applies
from ..llm import LLMError, complete_chat_json
from ..settings import get_settings
from .buffer import RepoBuffer
from .components import Component, SymbolGraph
from .flow import FlowResult
from .miners import ConfigIndex
from .review import Addition, ReviewOut, Verdict, taxonomy
from .signals import SymbolInfo
from .slices import make_slice

logger = logging.getLogger(__name__)

PROMPT_VERSION = "investigate-v1"
MAX_CALLS_PER_TURN = 4
MAX_READ_LINES = 80
MAX_HITS = 20
MAX_RESULT_CHARS = 9000
MAX_PROBES = 3

SYSTEM = f"""You are an evaluation engineer auditing an LLM application ({PROMPT_VERSION}). You get its AI MAP: components,
pipeline, what every model call is built from, where model output and untrusted data go, and static findings.
Goal: decide where the app will break (quality, safety, latency, cost) and how to evaluate it.

Work in turns. Each turn either asks for tools (calls) or finishes (done=true). Tools are read-only:
- open_component {{"id": node id}}       code of the component's function, numbered lines
- read {{"file": path, "start": n, "end": m}}  at most {MAX_READ_LINES} lines
- search {{"pattern": regex}}               matching lines across the code (max {MAX_HITS})
- callers {{"id": node id}} / callees {{"id": node id}}   functions that call it / that it calls
- flows {{"id": node id}}                   data flowing into and out of the component
- config {{"name": name}}                   value of a constant / config key and where it is set
Look at the code behind the highest-risk findings and recipes first. Be economical: you have few turns.

When done, return:
- verdicts: for static finding ids (rule:node) you checked: keep=false only if the code clearly contradicts it;
  `why` = 2-3 sentences grounded in what you read (what the code does, why it hurts, what to measure).
- additions: new findings ONLY with rule ids from the list, with `file` and `evidence_lines` you were shown.
- evals: evaluator proposals (evaluator ids from the list, on node ids that exist) with a rationale and up to
  {MAX_PROBES} probes. A probe is a complete message a real user of THIS app would send (use its domain, see the
  readme excerpt) that is likely to expose the problem, e.g. a question the corpus cannot answer, a tool request
  with tricky arguments, an injection attempt. Never questions about "the answer", "the context" or the app itself.
Never invent code, values or line numbers. No markdown."""


class ToolCall(BaseModel):
    tool: Literal["open_component", "read", "search", "callers", "callees", "flows", "config"]
    args: dict[str, Any] = Field(default_factory=dict)


class EvalProposal(BaseModel):
    node_id: str
    evaluator_id: str
    rationale: str = ""
    probes: list[str] = Field(default_factory=list)


class Step(BaseModel):
    thought: str = ""
    calls: list[ToolCall] = Field(default_factory=list)
    done: bool = False
    verdicts: list[Verdict] = Field(default_factory=list)
    additions: list[Addition] = Field(default_factory=list)
    evals: list[EvalProposal] = Field(default_factory=list)


@dataclass
class Investigation:
    out: ReviewOut
    seen: dict[str, set[int]]
    evals: list[EvalProposal] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def visible(self, path: str, line: int) -> bool:
        return line in self.seen.get(path, set())


class Tools:
    """Read-only views of the analyzed repo. Every line a tool shows is recorded in `seen`."""

    def __init__(self, buffer: RepoBuffer, files: list[FileFacts], symbols: dict[str, SymbolInfo],
                 comps: list[Component], g: SymbolGraph, flow: FlowResult | None):
        self.buffer, self.symbols, self.g, self.flow = buffer, symbols, g, flow
        self.facts = {f.path: f for f in files}
        self.comps = {c.id: c for c in comps}
        self.index = ConfigIndex(buffer, files)
        self.seen: dict[str, set[int]] = {}

    def show(self, path: str, line: int) -> str:
        self.seen.setdefault(path, set()).add(line)
        return f"{path}:{line}| {self.buffer.line(path, line)}"

    def run(self, call: ToolCall) -> str:
        try:
            text = getattr(self, call.tool)(**{k: v for k, v in call.args.items() if isinstance(k, str)})
        except (TypeError, ValueError, KeyError, re.error) as exc:
            text = f"error: {type(exc).__name__}: {exc}"
        return text if len(text) <= MAX_RESULT_CHARS else text[:MAX_RESULT_CHARS] + "\n… (cut)"

    def _comp(self, id: str) -> Component:
        if id not in self.comps:
            raise KeyError(f"unknown node id {id!r}")
        return self.comps[id]

    def open_component(self, id: str) -> str:
        c = self._comp(id)
        s = make_slice("x", self.symbols[c.key], self.buffer, self.facts)
        self.seen.setdefault(s.path, set()).update(s.lines)
        return s.text.replace("### slice x · ", "### ", 1)

    def read(self, file: str, start: int = 1, end: int | None = None) -> str:
        if file not in self.buffer.files:
            raise KeyError(f"no such file {file!r}")
        start = max(1, int(start))
        end = min(int(end or start + MAX_READ_LINES - 1), start + MAX_READ_LINES - 1, len(self.buffer.files[file].splitlines()))
        return "\n".join(self.show(file, i) for i in range(start, end + 1)) or "(empty range)"

    def search(self, pattern: str) -> str:
        rx = re.compile(pattern)
        hits = []
        for path, text in self.buffer.code_files().items():
            for i, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    hits.append(self.show(path, i))
                    if len(hits) >= MAX_HITS:
                        return "\n".join(hits) + "\n… (more hits)"
        return "\n".join(hits) or "(no matches)"

    def _defs(self, keys) -> str:
        out = []
        for k in sorted(keys)[:MAX_HITS]:
            info = self.symbols.get(k)
            if info is not None:
                out.append(self.show(info.path, info.d.line))
        return "\n".join(out) or "(none)"

    def callers(self, id: str) -> str:
        return self._defs(self.g.inc.get(self._comp(id).key, ()))

    def callees(self, id: str) -> str:
        return self._defs(self.g.out.get(self._comp(id).key, ()))

    def flows(self, id: str) -> str:
        self._comp(id)
        if self.flow is None:
            return "(no data-flow facts for this language)"
        out = []
        for f in self.flow.flows:
            if id in (f.source, f.target):
                via = " → ".join(self.show(e.file, e.line) for e in f.via if e.line)
                out.append(f"{f.source} --{f.label}--> {f.target} ({f.kind})\n  {via}")
        for s in self.flow.sinks:
            if s.comp == id:
                labels = sorted({t.label for t in s.taints()})
                out.append(f"{s.kind} sink ← {', '.join(labels) or 'literals'}\n  {self.show(s.path, s.line)}")
        return "\n".join(out[:MAX_HITS]) or "(no flows recorded)"

    def config(self, name: str) -> str:
        hit = self.index.resolve(name)
        if hit is None:
            return f"(no constant or config key named like {name!r})"
        value, path, line = hit
        return f"{name} = {value!r}" + (f"\n{self.show(path, line)}" if path and line else "")


META_PROBE = re.compile(r"\b(the|your|this) (answer|response|context|retrieved|chunks?|documents? retrieved|app|system)\b"
                        r"|\bsummari[sz]e the answer\b|\bis the answer\b", re.I)


def _readme(buffer: RepoBuffer, limit: int = 1500) -> str:
    path = next((p for p in sorted(buffer.files) if re.match(r"(?i)(.*/)?readme\.(md|txt|rst)$", p)), None)
    return buffer.files[path][:limit] if path else ""


def _user(aimap: str, findings: dict[str, Any], turns: int, readme: str = "") -> str:
    tax = {k: v[1] for k, v in taxonomy().items()}
    evaluators = {e.id: f"{e.name} (applies to {', '.join(e.applies_to[:6])})" for e in BY_ID.values()}
    return json.dumps({"budget": f"{turns} turns, up to {MAX_CALLS_PER_TURN} tool calls each",
                       "rule_ids": tax, "evaluator_ids": evaluators, "finding_ids": sorted(findings)}) \
        + "\n\n" + aimap + (f"\n\n## README EXCERPT (domain of the app)\n{readme}" if readme else "")


def investigate(aimap: str, findings: dict[str, Any], graph: ApplicationGraph, buffer: RepoBuffer,
                files: list[FileFacts], symbols: dict[str, SymbolInfo], comps: list[Component], g: SymbolGraph,
                flow: FlowResult | None) -> Investigation | None:
    """findings: merged finding id (rule:node) -> Finding. Returns None on any LLM failure (caller falls back)."""
    turns = max(1, get_settings().investigator_turns)
    tools = Tools(buffer, files, symbols, comps, g, flow)
    messages = [{"role": "user", "content": _user(aimap, findings, turns, _readme(buffer))}]
    stats = {"calls": 0, "tool_calls": 0, "turns": 0, "finished": False}
    step: Step | None = None
    try:
        for turn in range(turns + 1):
            if turn == turns:  # out of budget: ask for the final answer
                messages.append({"role": "user", "content": "Budget used. Reply now with done=true and your verdicts, additions and evals."})
            step, _ = complete_chat_json("investigate", SYSTEM, messages, Step, max_tokens=3500)
            stats["calls"] += 1
            stats["turns"] = turn + 1
            if step.done or not step.calls or turn == turns:
                break
            results = []
            for call in step.calls[:MAX_CALLS_PER_TURN]:
                stats["tool_calls"] += 1
                results.append(f"## {call.tool} {json.dumps(call.args)}\n{tools.run(call)}")
            messages.append({"role": "assistant", "content": step.model_dump_json(include={"thought", "calls"})})
            messages.append({"role": "user", "content": "\n\n".join(results) + f"\n\n({turns - turn - 1} turns left)"})
    except LLMError as exc:
        logger.warning("investigator failed, falling back to batch review: %s", exc)
        return None
    if step is None:
        return None
    stats["finished"] = step.done
    nodes = {n.id: n for n in graph.nodes}
    evals = [e for e in step.evals if e.node_id in nodes and e.evaluator_id in BY_ID and applies(e.evaluator_id, nodes[e.node_id].type)]
    stats["evals"], stats["evals_rejected"] = len(evals), len(step.evals) - len(evals)
    for e in evals:  # meta-questions ("is the answer relevant?") test nothing when sent to the app
        kept = [p for p in e.probes[:MAX_PROBES] if p.strip() and not META_PROBE.search(p)]
        stats["probes_dropped"] = stats.get("probes_dropped", 0) + len(e.probes) - len(kept)
        e.probes = kept
    return Investigation(out=ReviewOut(verdicts=step.verdicts, additions=step.additions), seen=tools.seen, evals=evals, stats=stats)
