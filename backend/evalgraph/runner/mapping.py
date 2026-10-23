"""Runtime spans → graph nodes (docs/02-CONTRACTS.md §7).

Rules, first match wins per span:
  1. node.symbol == span.component and the files match (suffix match, forward slashes)
  2. node.symbol == span.component
  3. only one node of span.kind exists
Every node matched by the winning rule gets the span (one function can be several facets of the graph).
Entrypoint nodes without a span of their own get the request's root span: an entrypoint stands for the whole request.
"""

from __future__ import annotations

import logging

from ..core.schemas import GraphNode, Span

logger = logging.getLogger(__name__)


def _norm(path: str | None) -> str:
    return (path or "").replace("\\", "/").lstrip("./")


def _same_file(node_file: str | None, span_file: str) -> bool:
    a, b = _norm(node_file), _norm(span_file)
    return bool(a and b) and (a.endswith(b) or b.endswith(a))


def map_spans(nodes: list[GraphNode], spans: list[Span]) -> tuple[dict[str, list[Span]], list[Span]]:
    """(node_id -> spans in call order, unmapped spans)."""
    by_kind: dict[str, list[GraphNode]] = {}
    for n in nodes:
        by_kind.setdefault(n.type, []).append(n)
    mapped: dict[str, list[Span]] = {}
    unmapped: list[Span] = []
    for span in spans:
        named = [n for n in nodes if n.symbol and n.symbol == span.component]
        targets = [n for n in named if _same_file(n.file, span.file)] or named
        if not targets and len(by_kind.get(span.kind, [])) == 1:
            targets = by_kind[span.kind]
        for n in targets:
            mapped.setdefault(n.id, []).append(span)
        if not targets:
            unmapped.append(span)
    root = next((s for s in reversed(spans) if s.kind == "entrypoint"), None)
    if root is not None:
        for n in by_kind.get("entrypoint", []):
            mapped.setdefault(n.id, [root])
    if unmapped:
        logger.debug("unmapped spans: %s", sorted({s.component for s in unmapped}))
    return mapped, unmapped
