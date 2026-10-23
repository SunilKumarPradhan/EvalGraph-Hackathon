"""Evaluator implementations by id. Must cover the catalog exactly (docs/02-CONTRACTS.md §3)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..core.schemas import EvaluatorSpec
from . import deterministic as d
from . import judges as j
from .base import SampleContext
from .catalog import BY_ID


@dataclass(frozen=True)
class Evaluator:
    spec: EvaluatorSpec
    fn: Callable[[SampleContext], tuple[float, str]]
    needs_span: bool  # the node must have executed in the sample; False = scored from the whole request

    @property
    def is_judge(self) -> bool:
        return self.spec.method == "llm_judge"


def _e(evaluator_id: str, fn, needs_span: bool = True) -> tuple[str, Evaluator]:
    return evaluator_id, Evaluator(BY_ID[evaluator_id], fn, needs_span)


EVALUATORS: dict[str, Evaluator] = dict([
    _e("context_relevance", j.context_relevance),
    _e("context_precision", d.context_precision),
    _e("context_recall", d.context_recall),
    _e("faithfulness", j.faithfulness),
    _e("answer_relevance", j.answer_relevance),
    _e("numeric_accuracy", d.numeric_accuracy),
    _e("routing_accuracy", d.routing_accuracy, needs_span=False),     # a blocked request never reaches the router
    _e("parse_success", d.parse_success),
    _e("tool_selection", d.tool_selection, needs_span=False),
    _e("tool_correctness", d.tool_correctness, needs_span=False),     # "never called" is a failure, not a skip
    _e("review_accuracy", d.review_accuracy),
    _e("injection_resistance", d.injection_resistance, needs_span=False),
    _e("safety", j.safety),
    _e("context_integrity", j.context_integrity),
    _e("task_completion", j.task_completion),
    _e("citation_correctness", d.citation_correctness),  # the node's text vs the request's final context
    _e("latency", d.latency),
    _e("cost", d.cost),
    _e("memory_footprint", d.memory_footprint),
])

assert set(EVALUATORS) == set(BY_ID), f"registry/catalog mismatch: {set(EVALUATORS) ^ set(BY_ID)}"
