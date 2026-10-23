"""RECOMMEND (LLM half): bottleneck review.

Per batch of components the model sees the slice, the miners' candidate findings and the taxonomy. It may
reject heuristic findings, add findings from the taxonomy (with line numbers it can see) and write the WHY.
Config-flag and secret findings are facts and cannot be rejected.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, Field

from ..core.schemas import ApplicationGraph, Evidence, Hotspot
from ..evaluators.catalog import applies
from ..hotspots.rules import RULES as FLAG_RULES
from ..llm import LLMError, complete_json
from ..settings import get_settings
from .buffer import RepoBuffer
from .miners import TAXONOMY_EXTRA, Finding
from .slices import Slice

logger = logging.getLogger(__name__)

SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}
REJECTABLE = {"RAG_NO_METADATA_FILTER", "LLM_NO_MAX_TOKENS", "LLM_NO_RETRY", "LLM_IN_LOOP", "PARSER_NAIVE_JSON",
              "INDEX_REBUILT_PER_REQUEST", "PROMPT_INJECTION_SURFACE", "NO_EVAL_SUITE",
              "LLM_OUTPUT_UNVALIDATED", "INDIRECT_INJECTION"}  # may-flow: the code can show they don't happen
BATCH = 6


def taxonomy() -> dict[str, tuple]:
    """rule_id -> (severity, title, impacts, evaluators, fix_variant)"""
    t = {r.id: (r.severity, r.title, r.impacts, r.evaluators, r.fix_variant) for r in FLAG_RULES}
    t.update(TAXONOMY_EXTRA)
    return t


SYSTEM = """You review an LLM application's components for evaluation hotspots: places where quality, latency,
cost or safety are likely to break. For each component you get its code slice (numbered lines), the static
findings already detected, and the allowed rule ids.
- For each finding id: keep=true if the code supports it, keep=false if it is clearly wrong; write a 2-3 sentence
  `why` grounded in the code (what it does, why it hurts, what to measure).
- You may add findings ONLY with rule ids from the list, and only with evidence_lines you can see in the slice.
- Never invent configuration values. No markdown."""


class Verdict(BaseModel):
    finding_id: str
    keep: bool = True
    why: str = ""


class Addition(BaseModel):
    rule_id: str
    node_id: str
    file: str | None = None                 # investigator: the file the lines are in (batch review: the slice's)
    evidence_lines: list[int] = Field(default_factory=list)
    why: str = ""


class ReviewOut(BaseModel):
    verdicts: list[Verdict] = Field(default_factory=list)
    additions: list[Addition] = Field(default_factory=list)


def _dedupe(findings: list[Finding]) -> dict[str, Finding]:
    merged: dict[str, Finding] = {}
    for f in findings:
        fid = f"{f.rule_id}:{f.node_id}"
        if fid in merged:
            known = {(e.file, e.line, e.fact) for e in merged[fid].evidence}
            merged[fid].evidence += [e for e in f.evidence if (e.file, e.line, e.fact) not in known]
        else:
            merged[fid] = Finding(f.rule_id, f.node_id, list(f.evidence), dict(f.params))
    return merged


def review(findings: list[Finding], graph: ApplicationGraph, slices: list[Slice], buffer: RepoBuffer,
           variants: list[str], use_llm: bool = True, investigation=None) -> tuple[list[Hotspot], dict]:
    """investigation: an understanding.investigate.Investigation. Its verdicts/additions are applied first (lines
    verified against what its tools showed); the batch review then covers only findings it left without a verdict."""
    tax = taxonomy()
    merged = _dedupe(findings)
    whys: dict[str, str] = {}
    sources = {fid: "static" for fid in merged}
    stats = {"calls": 0, "rejected": 0, "added": 0, "invalid": 0}
    slice_of = {cid: s for s in slices for cid in s.component_ids}
    nodes = {n.id: n for n in graph.nodes}

    def apply(out: ReviewOut, visible) -> None:
        """visible(addition) -> (file, verified lines)"""
        for v in out.verdicts:
            if v.finding_id not in merged:
                stats["invalid"] += 1
                continue
            if v.why.strip():
                whys[v.finding_id] = v.why.strip()[:800]
                sources[v.finding_id] = "static+llm"
            if not v.keep and merged[v.finding_id].rule_id in REJECTABLE:
                del merged[v.finding_id]
                stats["rejected"] += 1
        for a in out.additions:
            path, lines = visible(a)
            if a.rule_id not in tax or a.node_id not in nodes or not lines:
                stats["invalid"] += 1
                continue
            fid = f"{a.rule_id}:{a.node_id}"
            ev = [Evidence(file=path, line=ln, snippet=buffer.line(path, ln)[:200], fact=f"LLM: {tax[a.rule_id][1].lower()}")
                  for ln in lines[:3]]
            if fid in merged:
                merged[fid].evidence += ev
            else:
                merged[fid] = Finding(a.rule_id, a.node_id, ev)
                sources[fid] = "llm"
                stats["added"] += 1
            if a.why.strip():
                whys[fid] = a.why.strip()[:800]

    def in_slice(a: Addition) -> tuple[str, list[int]]:
        s = slice_of.get(a.node_id)
        return (s.path if s else ""), [ln for ln in a.evidence_lines if s and ln in s.lines]

    covered: set[str] = set()
    if investigation is not None:
        def seen(a: Addition) -> tuple[str, list[int]]:
            path = a.file or (slice_of[a.node_id].path if a.node_id in slice_of else "")
            return path, [ln for ln in a.evidence_lines if investigation.visible(path, ln)]

        covered = {v.finding_id for v in investigation.out.verdicts if v.finding_id in merged}
        apply(investigation.out, seen)

    if use_llm:
        by_node: dict[str, list[str]] = defaultdict(list)
        for fid, f in merged.items():
            by_node[f.node_id].append(fid)
        node_ids = [n for n in nodes if n in slice_of
                    and (investigation is None or any(fid not in covered for fid in by_node.get(n, [])))]
        groups = [node_ids[i:i + BATCH] for i in range(0, len(node_ids), BATCH)]

        def call(group: list[str]) -> ReviewOut | None:
            payload = []
            for nid in group:
                n = nodes[nid]
                payload.append({"node_id": nid, "type": n.type, "label": n.label, "purpose": n.description,
                                "slice": slice_of[nid].text,
                                "findings": [{"id": fid, "rule_id": merged[fid].rule_id,
                                              "evidence": [e.fact for e in merged[fid].evidence]}
                                             for fid in by_node.get(nid, []) if fid not in covered]})
            user = json.dumps({"rule_ids": {k: v[1] for k, v in tax.items()}, "components": payload})
            try:
                out, _ = complete_json("bottleneck_review", SYSTEM, user, ReviewOut, max_tokens=3500)
                return out
            except LLMError as exc:
                logger.warning("bottleneck review batch failed: %s", exc)
                return None

        with ThreadPoolExecutor(max_workers=max(1, get_settings().max_concurrency)) as pool:
            results = list(pool.map(call, groups))
        stats["calls"] = len(groups)
        for out in results:
            if out is not None:
                apply(out, in_slice)

    hotspots = []
    for fid, f in merged.items():
        node = nodes.get(f.node_id)
        if node is None or f.rule_id not in tax:
            continue
        severity, title, impacts, evaluators, fix = tax[f.rule_id]
        facts = "; ".join(e.fact for e in f.evidence[:4])
        hotspots.append(Hotspot(
            id=fid, rule_id=f.rule_id, node_id=f.node_id, severity=severity, title=title,
            why=whys.get(fid) or f"{node.label}: {title.lower()}. Evidence: {facts}. Likely impact: {', '.join(impacts)}.",
            evidence=f.evidence[:6], impacts=list(impacts),
            recommended_evaluators=[e for e in evaluators if applies(e, node.type)],
            suggested_variant=fix if fix in variants else None, source=sources.get(fid, "static")))
    hotspots.sort(key=lambda h: (SEVERITY_RANK[h.severity], h.rule_id, h.node_id))
    return hotspots, stats
