"""Analysis job: source -> AnalysisResult (saved to data/analyses), via the Semantic Code Map pipeline."""

from __future__ import annotations

from datetime import datetime, timezone

from . import storage
from .core.ids import new_id
from .core.schemas import AnalysisResult, AnalysisSource, AnalysisSummary, RuntimeInfo
from .runner.adapters import get_adapter
from .settings import get_settings
from .understanding.pipeline import Report, understand

HEADLINE_ORDER = [
    ("agent", "agent", "agents"), ("llm", "LLM call", "LLM calls"), ("router", "router", "routers"),
    ("parser", "parser", "parsers"), ("retriever", "retriever", "retrievers"), ("reranker", "reranker", "rerankers"),
    ("vector_db", "vector DB", "vector DBs"), ("embedding", "embedding model", "embedding models"),
    ("dataset", "ingestion step", "ingestion steps"), ("tool", "tool", "tools"),
    ("external_api", "external API", "external APIs"), ("guardrail", "guardrail", "guardrails"),
    ("memory", "memory", "memories"), ("prompt", "prompt", "prompts"), ("entrypoint", "entrypoint", "entrypoints"),
]


def headline(counts: dict[str, int], gaps: int) -> str:
    parts = [f"{counts[t]} {one if counts[t] == 1 else many}" for t, one, many in HEADLINE_ORDER if counts.get(t)]
    found = ", ".join(parts) if parts else "no AI components"
    return f"Detected {found} · {gaps} evaluation gap{'s' if gaps != 1 else ''}"


def run_analysis(source: dict, report: Report) -> str:
    u = understand(source, report)
    adapter = get_adapter((u.evalgraph_yaml or {}).get("adapter"))
    sandbox_ok = bool(u.runtime) and u.runtime["plan"].get("mode") != "unsupported"
    counts: dict[str, int] = {}
    for n in u.graph.nodes:
        counts[n.type] = counts.get(n.type, 0) + 1
    summary = AnalysisSummary(
        language=", ".join(u.languages) or "unknown", languages=u.languages, files_scanned=u.files_scanned,
        frameworks=u.frameworks, counts=counts, eval_gaps=len(u.hotspots), runnable=adapter is not None or sandbox_ok,
        adapter=adapter.name if adapter else ("sandbox" if sandbox_ok else None),
        variants=adapter.variants() if adapter else (["as-is"] if sandbox_ok else []),
        headline=headline(counts, len(u.hotspots)), llm_calls=u.llm_calls,
        truncated=u.buffer.truncated, skipped_files=u.buffer.skipped,
    )
    b = u.buffer
    result = AnalysisResult(
        id=new_id("a"), created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        source=AnalysisSource(repo_url=b.repo_url, local_path=b.local_path, commit=b.commit, name=b.name),
        summary=summary, graph=u.graph, hotspots=u.hotspots, recommendations=u.recommendations,
        runtime=RuntimeInfo(**u.runtime) if u.runtime else None,
        mock=get_settings().mock_llm, aimap=u.aimap or None,
    )
    storage.save("analyses", result)
    how = (f" · runnable ({adapter.name} adapter)" if adapter else
           f" · runnable in Docker sandbox ({u.runtime['plan']['reason']})" if sandbox_ok else " · plan mode")
    report("Done", 0.99, summary.headline + how)
    return result.id
