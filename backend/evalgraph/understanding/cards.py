"""UNDERSTAND (LLM half): map step. Batched slices -> Component Cards, verified against the buffer."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, Field

from ..core.schemas import NodeType
from ..llm import LLMError, complete_json
from ..settings import get_settings
from .components import Component
from .signals import FACET_TYPE, SymbolInfo
from .slices import Slice, batches

logger = logging.getLogger(__name__)

PROMPT_VERSION = "cards-v1"
RETYPE_OK = {"llm", "agent", "router", "parser", "guardrail", "tool"}  # LLM-driven roles that may be refined

SYSTEM = f"""You map the architecture of an LLM application ({PROMPT_VERSION}).
You get numbered code slices. For each slice decide whether the function is a component of the AI pipeline and
describe it. Component types: entrypoint, agent, router, llm, prompt, retriever, vector_db, embedding, reranker,
tool, memory, guardrail, external_api, parser, dataset (document ingestion), other.
Rules:
- Slices marked [component: ...] were already detected from code; confirm or refine the type, write a label and purpose.
- Slices marked [candidate] may or may not be components: set is_component=false unless the code clearly plays an
  AI-pipeline role (e.g. a tool the agent calls, a custom retriever, a prompt builder, an input check).
- label: 2-4 words, Title Case, role-oriented (e.g. "Knowledge Retriever", "Ticket Tool").
- purpose: one sentence, max 160 chars, about what it does for the pipeline.
- evidence_lines: line numbers FROM THE SLICE that justify the type. Only numbers you can see.
- Do not invent behaviour that is not in the code."""


class CardOut(BaseModel):
    slice_id: str
    is_component: bool = True
    type: NodeType | None = None
    label: str | None = None
    purpose: str | None = None
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    evidence_lines: list[int] = Field(default_factory=list)


class CardBatch(BaseModel):
    cards: list[CardOut] = Field(default_factory=list)


def _render(batch: list[Slice], comps_by_id: dict[str, Component]) -> str:
    parts = []
    for s in batch:
        tag = ("[component: " + ", ".join(f"{cid} ({comps_by_id[cid].type})" for cid in s.component_ids) + "]"
               if s.component_ids else "[candidate]")
        parts.append(f"{tag}\n{s.text}")
    return "\n\n".join(parts) + "\n\nReturn one card per slice id."


def run_cards(slices: list[Slice], comps: list[Component], symbols: dict[str, SymbolInfo],
              calls: dict[str, set[str]] | None = None) -> tuple[list[Component], dict]:
    """Applies verified cards to components (labels, purposes, safe retypes) and adds discovered components.

    calls: symbol key -> directly called symbol keys. A discovered candidate that directly calls an existing component
    of the same type is a wrapper of it (e.g. a LangGraph node function around the real retriever) and is not added:
    the same delegation rule the static pipeline applies.
    """
    stats = {"calls": 0, "cards": 0, "discovered": 0, "rejected": 0, "wrappers": 0, "failed_batches": 0}
    comps_by_id = {c.id: c for c in comps}
    groups = batches(slices)
    if not groups:
        return comps, stats

    def call(batch: list[Slice]) -> CardBatch | None:
        try:
            out, _ = complete_json("component_cards", SYSTEM, _render(batch, comps_by_id), CardBatch, max_tokens=3500)
            return out
        except LLMError as exc:
            logger.warning("card batch failed: %s", exc)
            return None

    with ThreadPoolExecutor(max_workers=max(1, get_settings().max_concurrency)) as pool:
        results = list(pool.map(call, groups))
    stats["calls"] = len(groups)
    by_id = {s.id: s for s in slices}
    for result in results:
        if result is None:
            stats["failed_batches"] += 1
            continue
        for card in result.cards:
            s = by_id.get(card.slice_id)
            if s is None:
                stats["rejected"] += 1
                continue
            lines = [ln for ln in card.evidence_lines if ln in s.lines]
            stats["cards"] += 1
            if s.component_ids:
                for cid in s.component_ids:
                    c = comps_by_id[cid]
                    if card.label:
                        c.label = card.label.strip()[:40]
                    if card.purpose:
                        c.description = card.purpose.strip()[:200]
                    if (card.type and card.type != c.type and len(s.component_ids) == 1 and lines
                            and card.type in RETYPE_OK and c.type in RETYPE_OK):
                        c.type = card.type
            elif card.is_component and card.type and card.type in FACET_TYPE.values() and lines:
                info = symbols[s.key]
                callees = (calls or {}).get(s.key, set())
                if any(c.key in callees and c.type == card.type for c in comps):
                    stats["wrappers"] += 1
                    continue
                from .signals import Signal

                sig = [Signal(next(f for f, t in FACET_TYPE.items() if t == card.type), 0.7, ln,
                              _line(info, ln), f"LLM: {card.purpose or card.type}") for ln in lines[:3]]
                comp = Component(id=f"{card.type}:{info.d.name}", type=card.type,
                                 facet=next(f for f, t in FACET_TYPE.items() if t == card.type), key=s.key,
                                 path=info.path, qualname=info.d.qualname, label=(card.label or info.d.name)[:40],
                                 signals=sig, description=(card.purpose or "")[:200] or None, confidence=0.7)
                if comp.id not in comps_by_id:
                    comps.append(comp)
                    comps_by_id[comp.id] = comp
                    stats["discovered"] += 1
            elif card.is_component and not lines:
                stats["rejected"] += 1
    _facet_labels(comps)
    return comps, stats


TYPE_NAME = {"llm": "LLM", "vector_db": "Vector DB", "external_api": "External API", "entrypoint": "Entrypoint"}


def _facet_labels(comps: list[Component]) -> None:
    """One function split into several nodes (retriever + LLM + memory...) must not show the same label N times."""
    by_key: dict[str, list[Component]] = {}
    for c in comps:
        by_key.setdefault(c.key, []).append(c)
    for group in by_key.values():
        if len(group) > 1 and len({c.label for c in group}) < len(group):
            for c in group:
                role = TYPE_NAME.get(c.type, c.type.replace("_", " ").title())
                if role.lower() not in c.label.lower():
                    c.label = f"{c.label} · {role}"[:48]


def _line(info: SymbolInfo, line: int) -> str:
    calls = [c for c in info.d.calls if c.line == line]
    return calls[0].snippet if calls else info.d.snippet


def cards_payload_preview(slices: list[Slice]) -> str:
    """Debug helper: what the map step would send."""
    return json.dumps([{"id": s.id, "chars": s.size, "components": s.component_ids} for s in slices], indent=1)
