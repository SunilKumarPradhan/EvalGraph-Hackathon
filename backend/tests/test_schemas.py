import json

from evalgraph.core.schemas import AnalysisResult, RunResult
from evalgraph.settings import FIXTURES_DIR


def test_sample_analysis_fixture_is_valid():
    a = AnalysisResult.model_validate_json((FIXTURES_DIR / "sample_analysis.json").read_text(encoding="utf-8"))
    ids = {n.id for n in a.graph.nodes}
    assert all(e.source in ids and e.target in ids for e in a.graph.edges)
    assert all(h.node_id in ids and h.evidence for h in a.hotspots)
    assert all(r.node_id in ids for r in a.recommendations)
    assert a.summary.runnable and a.summary.adapter == "cuda_archer"


def test_sample_run_fixture_is_valid():
    r = RunResult.model_validate_json((FIXTURES_DIR / "sample_run.json").read_text(encoding="utf-8"))
    a = json.loads((FIXTURES_DIR / "sample_analysis.json").read_text(encoding="utf-8"))
    ids = {n["id"] for n in a["graph"]["nodes"]}
    assert r.analysis_id == a["id"] and r.mock is True
    assert all(x.node_id in ids for x in r.results)


def test_bool_config_values_stay_bool():
    a = AnalysisResult.model_validate_json((FIXTURES_DIR / "sample_analysis.json").read_text(encoding="utf-8"))
    reranker = next(n for n in a.graph.nodes if n.id == "reranker:rerank")
    assert reranker.config["USE_RERANKER"] is False and reranker.config["RERANK_TOP_N"] == 4


def test_evaluators_fixture_matches_catalog():
    from evalgraph.evaluators.catalog import CATALOG

    data = json.loads((FIXTURES_DIR / "evaluators.json").read_text(encoding="utf-8"))
    assert data == [e.model_dump() for e in CATALOG]
