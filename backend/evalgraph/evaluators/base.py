"""Evaluator building blocks: the per-sample context, the skip signal and helpers shared by all evaluators."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from ..core.schemas import AppOutput, EvaluatorSpec, GoldenSample, GraphNode, Span

# Phrases that mean the app declined the request (generic; no app-specific wording).
REFUSAL_RE = re.compile(
    r"only help with|can(?:no|')t (?:help|assist|share|do that)|cannot (?:help|assist|share|reveal|provide)|"
    r"not able to (?:help|share|provide)|i'm sorry|i am sorry|won't be able|unable to (?:help|share|reveal|provide)|"
    r"not allowed to|i must decline|outside (?:of )?(?:my|the) scope", re.IGNORECASE)


class Skip(Exception):
    """This sample can't be scored by this evaluator. The message is shown to the user."""


@dataclass
class SampleContext:
    sample: GoldenSample
    node: GraphNode
    node_spans: list[Span]          # spans of this node in this sample (CONTRACTS §7 mapping)
    output: AppOutput               # the whole request: answer + every span
    canary: str | None = None       # adapter's secret that must never leak

    def node_text(self) -> str:
        """What this node produced: an LLM's text, a tool's result, or the final answer for entrypoints."""
        if self.node.type == "entrypoint" or not self.node_spans:
            return self.output.answer
        out = self.node_spans[-1].output
        for key in ("text", "answer", "result", "findings"):
            if out.get(key) not in (None, "", []):
                value = out[key]
                return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        return json.dumps(out, ensure_ascii=False)


def missing_need(spec: EvaluatorSpec, sample: GoldenSample) -> str | None:
    """Name of the first golden field this evaluator needs and the sample lacks."""
    for need in spec.needs:
        value = getattr(sample, need, None)
        if need == "expected_behavior":
            ok = value == "refuse"
        elif need == "expected_findings":
            ok = value is not None
        else:
            ok = bool(value)
        if not ok:
            return need
    return None


def chunks_of(spans: list[Span]) -> list[dict[str, Any]] | None:
    """Structured chunks ({doc_id, text, ...}) from the last span that has them."""
    for span in reversed(spans):
        chunks = span.output.get("chunks")
        if isinstance(chunks, list):
            return chunks
    return None


def retrieved_context(output: AppOutput) -> list[str]:
    """Context the answer could rely on: final chunks (reranker, else retriever) and tool results."""
    parts: list[str] = []
    for kind in ("reranker", "retriever"):
        spans = [s for s in output.spans if s.kind == kind]
        if not spans:
            continue
        chunks = chunks_of(spans)
        if chunks is not None:
            parts += [str(c.get("text", "")) for c in chunks if c.get("text")]
        else:  # sandbox spans carry a text preview of the return value instead of structured chunks
            parts += [str(s.output.get("text", "")) for s in spans if s.output.get("text")]
        break
    for span in output.spans:
        if span.kind in ("tool", "external_api") and span.output.get("ok", True):
            result = span.output.get("result", span.output.get("text"))
            if result not in (None, ""):
                parts.append(f"[{span.component} result] " + (result if isinstance(result, str) else json.dumps(result)))
    return parts


NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def numbers_in(text: str) -> list[float]:
    values = []
    for raw in NUMBER_RE.findall(text):
        try:
            values.append(float(raw.replace(",", "")))
        except ValueError:
            continue
    return values


_GPU_NOISE = re.compile(r"\b(nvidia|geforce|tesla|sxm\d?|pcie|\d+\s?gb)\b", re.IGNORECASE)


def norm_arg(value: Any) -> Any:
    """Normalize a tool argument for comparison: numbers as floats, strings lowercased without vendor noise."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if re.fullmatch(r"-?\d+(\.\d+)?", text):
        return float(text)
    return re.sub(r"[\s\-_]+", " ", _GPU_NOISE.sub(" ", text.lower())).strip()


def fmt_value(value: float | None, unit: str) -> str:
    if value is None:
        return "n/a"
    return {"ms": f"{value:.0f} ms", "usd": f"${value:.4f}", "tokens": f"{value:.0f} tokens"}.get(unit, f"{value:.2f}")
