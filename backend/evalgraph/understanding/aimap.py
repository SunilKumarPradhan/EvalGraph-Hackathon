"""AI Map: the whole analysis as one compact, line-referenced text an LLM can read in a single pass.

Replaces "dump the repo" with what matters for evaluation planning: components and their configuration, the
pipeline, what every model call is built from (prompt recipes), where model output and untrusted text go, and the
static findings. Every fact carries file:line, so anything an LLM concludes from it can be checked. ~3-8k tokens.
Deterministic: same repo, same map. Users can download it (GET /api/analyses/{id}/aimap) and paste it anywhere.
"""

from __future__ import annotations

from collections import defaultdict

from ..core.schemas import ApplicationGraph, GraphNode
from .flow import FlowResult
from .miners import Finding

MAX_CHARS = 26_000            # ~6.5k tokens
MAX_FLOWS = 40
MAX_FINDINGS = 45
LABEL_HELP = ("labels: user_input (request), retrieved (RAG context), memory (past turns), tool_output, external (HTTP),"
              " llm_output (model output or model-chosen arguments), prompt (prompt constant)")


def _where(n: GraphNode) -> str:
    lines = [e.line for e in n.evidence if e.line]
    return f"{n.file}:{min(lines)}" if n.file and lines else (n.file or "?")


def _cfg(n: GraphNode) -> str:
    items = [f"{k}={v!r}" if isinstance(v, str) else f"{k}={v}" for k, v in list(n.config.items())[:8]]
    return " ".join(items)


def _taint_ref(t) -> str:
    who = t.comp or t.name
    return f"{who} @{t.path}:{t.line}" if t.line else who


def render(name: str, frameworks: list[str], languages: list[str], graph: ApplicationGraph, flow: FlowResult | None,
           findings: list[Finding], commit: str | None = None, max_chars: int = MAX_CHARS) -> str:
    nodes = {n.id: n for n in graph.nodes}
    out = [f"# AI MAP · {name}{' @ ' + commit[:7] if commit else ''} · {', '.join(languages) or '?'}"
           f" · frameworks: {', '.join(frameworks) or 'none detected'}",
           "# [ids] are graph nodes; path:line points into the repo. " + LABEL_HELP, ""]

    out.append(f"## COMPONENTS ({len(graph.nodes)})")
    for n in graph.nodes:
        cfg = _cfg(n)
        out.append(f"[{n.id}] {n.type} \"{n.label}\" · {_where(n)}" + (f" · {cfg}" if cfg else ""))
        if n.description:
            out.append(f"    purpose: {n.description[:160]}")
        facts = "; ".join(e.fact for e in n.evidence[:2])
        if facts:
            out.append(f"    evidence: {facts[:200]}")
    out.append("")

    out.append(f"## PIPELINE ({len(graph.edges)} edges)")
    by_src: dict[str, list[str]] = defaultdict(list)
    for e in graph.edges:
        tag = e.kind + (f":{e.label}" if e.label else "") + (f" [{','.join(e.flow_labels)}]" if e.flow_labels else "")
        by_src[e.source].append(f"{e.target} ({tag})")
    for src, targets in by_src.items():
        out.append(f"{src} -> " + ", ".join(targets))
    out.append("")

    if flow is not None and flow.recipes:
        out.append("## PROMPT RECIPES (what each model call is built from)")
        for r in flow.recipes:
            parts = []
            for label in sorted(r.parts):
                refs = sorted({_taint_ref(t) for t in r.parts[label]})
                parts.append(f"{label}({'; '.join(refs[:4])}{'; …' if len(refs) > 4 else ''})")
            out.append(f"[{r.comp}] @{r.path}:{r.line} ← " + (" + ".join(parts) or "(only literals / unknown)")
                       + f" · system prompt carries: {', '.join(r.system) or '-'} · can call tools: {'yes' if r.has_tools else 'no'}")
        out.append("")

    if flow is not None and flow.sinks:
        risky = [s for s in flow.sinks if s.kind in ("exec", "tool", "parse", "http") and s.taints("llm_output", "user_input", "external")]
        if risky:
            out.append("## WHERE MODEL OUTPUT / UNTRUSTED DATA LANDS")
            for s in risky[:MAX_FLOWS]:
                labels = sorted({t.label for t in s.taints("llm_output", "user_input", "external")})
                guard = "validated" if s.validated else ("in try" if s.in_try else "no check")
                out.append(f"{s.kind} @{s.path}:{s.line} `{s.snippet[:80]}` ← {', '.join(labels)} · {guard}"
                           + (f" · node {s.comp}" if s.comp else ""))
            out.append("")
        if flow.flows:
            out.append("## DATA FLOWS between components")
            for f in flow.flows[:MAX_FLOWS]:
                via = " → ".join(f"{e.file}:{e.line}" for e in f.via if e.line)
                out.append(f"{f.source} --{f.label}--> {f.target} ({f.kind}) [{via}]")
            if len(flow.flows) > MAX_FLOWS:
                out.append(f"… {len(flow.flows) - MAX_FLOWS} more")
            out.append("")

    merged: dict[str, list] = {}
    for f in findings:  # same ids as the review step: rule_id:node_id
        merged.setdefault(f"{f.rule_id}:{f.node_id}", []).extend(f.evidence)
    out.append(f"## STATIC FINDINGS ({len(merged)}; id = rule:node)")
    for fid, evidence in list(merged.items())[:MAX_FINDINGS]:
        ev = "; ".join(f"{e.fact} ({e.file}:{e.line})" if e.line else e.fact for e in evidence[:3])
        out.append(f"{fid} · {ev[:260]}")
    if len(merged) > MAX_FINDINGS:
        out.append(f"… {len(merged) - MAX_FINDINGS} more")

    text = "\n".join(out)
    if len(text) > max_chars:  # components first; the tail (long flow/finding lists) is what gets cut
        text = text[:max_chars].rsplit("\n", 1)[0] + "\n… (map truncated)"
    return text
