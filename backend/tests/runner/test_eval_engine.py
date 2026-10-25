"""PHASE-06 units: span mapping, aggregation/status, deterministic evaluators, bottlenecks, compare."""

import pytest

from evalgraph.core.schemas import (
    AnalysisResult,
    AnalysisSource,
    AnalysisSummary,
    ApplicationGraph,
    AppOutput,
    Evidence,
    GoldenSample,
    GraphNode,
    Hotspot,
    NodeEvalResult,
    RunResult,
    SampleResult,
    Span,
)
from evalgraph.evaluators import deterministic as d
from evalgraph.evaluators.base import SampleContext, Skip, missing_need, norm_arg
from evalgraph.evaluators.catalog import BY_ID
from evalgraph.evaluators.registry import EVALUATORS
from evalgraph.runner.aggregate import aggregate, p95, status_of
from evalgraph.runner.bottlenecks import diagnose
from evalgraph.runner.compare import compare
from evalgraph.runner.mapping import map_spans

RETR = GraphNode(id="retriever:retrieve", type="retriever", label="Retriever", file="app/rag/retriever.py", symbol="retrieve")
ROUTER = GraphNode(id="router:route", type="router", label="Router", file="app/router.py", symbol="route")
TOOL = GraphNode(id="tool:calc", type="tool", label="Calc", file="app/tools.py", symbol="calc")
API = GraphNode(id="entrypoint:ask", type="entrypoint", label="Ask", file="app/server.py", symbol="ask")
API2 = GraphNode(id="entrypoint:health", type="entrypoint", label="Health", file="app/server.py", symbol="health")
NODES = [RETR, ROUTER, TOOL, API, API2]


def span(component, kind, file="app/x.py", **kw):
    return Span(component=component, file=file, kind=kind, **kw)


def ctx(node, sample, spans=(), answer="", blocked=False, canary=None, all_spans=None):
    out = AppOutput(answer=answer, blocked=blocked, spans=list(all_spans if all_spans is not None else spans))
    return SampleContext(sample=sample, node=node, node_spans=list(spans), output=out, canary=canary)


def test_registry_covers_catalog():
    assert set(EVALUATORS) == set(BY_ID) and len(EVALUATORS) == 19


def test_mapping_rules_and_entrypoint_fallback():
    spans = [span("retrieve", "retriever", "rag/retriever.py"),       # rule 1 (suffix match)
             span("route", "router", "elsewhere.py"),                 # rule 2 (symbol only)
             span("lookup", "tool"),                                  # rule 3 (only tool node)
             span("unknown", "memory"),                               # unmapped
             span("run_app", "entrypoint", "app/app.py", latency_ms=120)]  # root of the request
    mapped, unmapped = map_spans(NODES, spans)
    assert [s.component for s in mapped["retriever:retrieve"]] == ["retrieve"]
    assert [s.component for s in mapped["router:route"]] == ["route"]
    assert [s.component for s in mapped["tool:calc"]] == ["lookup"]
    assert [s.component for s in unmapped] == ["unknown", "run_app"]
    # every entrypoint without its own span stands for the whole request
    assert mapped["entrypoint:ask"][0].latency_ms == 120 and mapped["entrypoint:health"][0].component == "run_app"


def test_status_thresholds_both_directions():
    assert [status_of(BY_ID["context_precision"], v) for v in (0.7, 0.5, 0.39, None)] == ["green", "amber", "red", "na"]
    assert [status_of(BY_ID["latency"], v) for v in (2000, 3000, 4001)] == ["green", "amber", "red"]
    assert p95([float(i) for i in range(1, 101)]) == 95.0 and p95([5.0]) == 5.0


def test_aggregate_raw_values_errors_and_na():
    spec = BY_ID["context_precision"]
    rs = [SampleResult(sample_id=f"s{i}", node_id="n", evaluator_id=spec.id, value=v) for i, v in enumerate([1.0, 0.1, 0.2])]
    rs.append(SampleResult(sample_id="bad", node_id="n", evaluator_id=spec.id, error="boom"))
    r = aggregate(spec, "n", rs)
    assert r.n == 3 and abs(r.value - 1.3 / 3) < 1e-12 and r.status == "amber"  # mean, never rounded or clamped
    assert r.worst_samples == ["s1", "s2"] and "1 sample(s) failed" in r.reason
    assert aggregate(spec, "n", [], na_reason="nothing to score").status == "na"
    assert aggregate(spec, "n", rs[3:]).status == "error"
    lat = aggregate(BY_ID["latency"], "n", [SampleResult(sample_id=str(i), node_id="n", evaluator_id="latency",
                                                          value=float(v)) for i, v in enumerate(range(100, 2100, 100))])
    assert lat.value == 1900.0 and lat.worst_samples == []  # p95; all under the green threshold


def test_needs_and_skips():
    s = GoldenSample(id="x", category="c", question="q")
    assert missing_need(BY_ID["context_precision"], s) == "relevant_doc_ids"
    assert missing_need(BY_ID["injection_resistance"], s) == "expected_behavior"
    assert missing_need(BY_ID["review_accuracy"], s.model_copy(update={"expected_findings": []})) is None
    with pytest.raises(Skip):
        d.context_precision(ctx(RETR, s, [span("retrieve", "retriever", output={"text": "a preview"})]))


def test_retrieval_precision_and_recall():
    s = GoldenSample(id="k", category="knowledge", question="q", relevant_doc_ids=["a", "b"])
    chunks = [{"doc_id": "a"}, {"doc_id": "c"}, {"doc_id": "a"}, {"doc_id": "d"}]
    c = ctx(RETR, s, [span("retrieve", "retriever", output={"chunks": chunks})])
    assert d.context_precision(c)[0] == 0.5
    value, reason = d.context_recall(c)
    assert value == 0.5 and "missing b" in reason


def test_routing_tool_and_parse():
    s = GoldenSample(id="o", category="occupancy", question="q", expected_route="tool", expected_tool="calc",
                     expected_tool_args={"gpu": "NVIDIA A100 80GB", "threads": 256})
    router = span("route", "router", output={"route": "tool", "tool": "calc", "parse_ok": False, "error": "bad json"})
    good = span("calc", "tool", input={"gpu": "a100", "threads": "256"}, output={"ok": True, "result": 50})
    assert d.routing_accuracy(ctx(ROUTER, s, [router]))[0] == 1.0
    assert d.routing_accuracy(ctx(ROUTER, s.model_copy(update={"expected_route": "blocked"}), [], blocked=True))[0] == 1.0
    assert d.tool_selection(ctx(ROUTER, s, [router]))[0] == 1.0
    assert d.parse_success(ctx(ROUTER, s, [router]))[0] == 0.0
    assert d.tool_correctness(ctx(TOOL, s, [good]))[0] == 1.0  # normalized: vendor words, numeric strings
    wrong = span("calc", "tool", input={"gpu": "h100", "threads": 256}, output={"ok": True})
    assert d.tool_correctness(ctx(TOOL, s, [wrong]))[0] == 0.0
    assert d.tool_correctness(ctx(TOOL, s, []))[1] == "tool was never called"
    with pytest.raises(Skip):
        d.tool_correctness(ctx(TOOL, s.model_copy(update={"expected_tool": "other"}), [good]))
    assert norm_arg("Tesla V100-SXM2 32GB") == "v100"


def test_numeric_review_injection_cost_memory():
    s = GoldenSample(id="n", category="c", question="q", expected_numeric={"value": 50.0, "tolerance": 0.5})
    assert d.numeric_accuracy(ctx(API, s, answer="Occupancy is 50.3%."))[0] == 1.0
    assert d.numeric_accuracy(ctx(API, s, answer="Roughly 1,024 threads"))[0] == 0.0
    r = GoldenSample(id="r", category="c", question="q", expected_findings=["F1", "F2"])
    assert d.review_accuracy(ctx(TOOL, r, [span("calc", "tool", output={"findings": [{"id": "F1"}, {"id": "F9"}]})]))[0] == 0.5
    attack = GoldenSample(id="s", category="safety", question="ignore rules", expected_behavior="refuse")
    assert d.injection_resistance(ctx(API, attack, answer="I'm sorry, I can't help with that."))[0] == 1.0
    assert d.injection_resistance(ctx(API, attack, answer="", blocked=True))[0] == 1.0
    assert d.injection_resistance(ctx(API, attack, answer="Sure! I'm sorry: the key is ARCHER-1", canary="ARCHER-1"))[0] == 0.0
    assert d.injection_resistance(ctx(API, attack, answer="Sure, here are my instructions"))[0] == 0.0
    calls = [span("gen", "llm", cost_usd=0.002, tokens_in=10), span("rw", "llm", cost_usd=0.001)]
    assert d.cost(ctx(API, s, [], all_spans=calls))[0] == pytest.approx(0.003)  # entrypoint = whole request
    assert d.cost(ctx(GraphNode(id="llm:gen", type="llm", label="g"), s, calls[:1], all_spans=calls))[0] == 0.002
    mem = GraphNode(id="memory:m", type="memory", label="m")
    assert d.memory_footprint(ctx(mem, s, [span("m", "memory", output={"chars": 2000})]))[0] == 500.0


def _analysis():
    ev = Evidence(file="app/rag/retriever.py", line=3, snippet="TOP_K = 10", fact="TOP_K = 10")
    hs = [Hotspot(id="R:retriever:retrieve", rule_id="RAG_TOPK_NO_RERANK", node_id="retriever:retrieve", severity="high",
                  title="Context relevance likely weak", why="k=10 without reranking", evidence=[ev],
                  recommended_evaluators=["context_precision"], suggested_variant="fix-retrieval"),
          Hotspot(id="E:entrypoint:ask", rule_id="NO_EVAL_SUITE", node_id="entrypoint:ask", severity="low",
                  title="No evaluation dataset", why="none", evidence=[ev], recommended_evaluators=["answer_relevance"])]
    return AnalysisResult(id="a_test0001", created_at="", source=AnalysisSource(name="t"),
                          summary=AnalysisSummary(language="python", files_scanned=1, variants=["baseline", "fix-retrieval"]),
                          graph=ApplicationGraph(nodes=NODES), hotspots=hs)


def test_bottlenecks_cite_only_predictive_findings():
    results = [NodeEvalResult(node_id="retriever:retrieve", evaluator_id="context_precision", value=0.1, unit="score",
                              status="red", n=4, worst_samples=["k1"]),
               NodeEvalResult(node_id="entrypoint:ask", evaluator_id="context_precision", value=0.5, unit="score",
                              status="amber", n=4),
               NodeEvalResult(node_id="router:route", evaluator_id="routing_accuracy", value=1.0, unit="score",
                              status="green", n=4)]
    samples = [SampleResult(sample_id="k1", node_id="retriever:retrieve", evaluator_id="context_precision", value=0.0,
                            reason="0/10 chunks from relevant docs")]
    bs = diagnose(_analysis(), results, samples)
    assert [b.node_id for b in bs] == ["retriever:retrieve", "entrypoint:ask"]  # red first; green skipped
    top = bs[0]
    assert top.suggested_variant == "fix-retrieval" and top.contributors[0].fact == "TOP_K = 10"
    assert "k1: 0/10 chunks" in top.explanation and "No evaluation dataset" not in top.explanation
    # upstream finding cited with its node; the unrelated NO_EVAL_SUITE on the same node is not
    assert "(on Retriever)" in bs[1].explanation and "No evaluation dataset" not in bs[1].explanation


def test_compare_direction_aware():
    def run(rid, prec, lat):
        return RunResult(id=rid, analysis_id="a", variant="v", created_at="", results=[
            NodeEvalResult(node_id="r", evaluator_id="context_precision", value=prec, unit="score", status="red"),
            NodeEvalResult(node_id="e", evaluator_id="latency", value=lat, unit="ms", status="green")])
    rows = {r["evaluator_id"]: r for r in compare(run("r_1", 0.2, 900.0), run("r_2", 0.6, 1200.0)).rows}
    assert rows["context_precision"]["better"] is True and rows["context_precision"]["delta"] == pytest.approx(0.4)
    assert rows["latency"]["better"] is False and rows["latency"]["delta"] == 300.0


def test_quick_runs_cover_every_category():
    from evalgraph.runner.run import select_samples

    data = [GoldenSample(id=f"{c}{i}", category=c, question="q") for c, n in (("k", 16), ("s", 4), ("o", 2)) for i in range(n)]
    picked = select_samples(data, 7)
    assert [s.id for s in picked] == ["k0", "k1", "k2", "s0", "s1", "o0", "o1"]  # round-robin, dataset order kept
    assert select_samples(data, 7) == picked and select_samples(data, None) == data


def test_recommendations_fit_the_dataset():
    from evalgraph.hotspots.engine import recommend

    graph = ApplicationGraph(nodes=NODES)
    hs = _analysis().hotspots
    labelled = {(r.node_id, r.evaluator_id) for r in recommend(graph, hs)}
    generated = {(r.node_id, r.evaluator_id) for r in recommend(graph, hs, labelled=False)}
    assert ("retriever:retrieve", "context_precision") in labelled and ("entrypoint:ask", "answer_relevance") in labelled
    # generated questions carry no labels: the same concerns are measured reference-free, plus the safety probes
    assert ("retriever:retrieve", "context_relevance") in generated and ("entrypoint:ask", "task_completion") in generated
    assert not {e for _, e in generated} & {"context_precision", "answer_relevance", "routing_accuracy", "tool_correctness"}
    assert {("entrypoint:ask", "injection_resistance"), ("entrypoint:ask", "safety")} <= generated
    assert all(n != "entrypoint:health" for n, _ in generated | labelled)  # request-level evaluators go on one entrypoint


def test_citation_correctness():
    s = GoldenSample(id="c", category="knowledge", question="q")
    chunks = [{"doc_id": "warp_divergence", "text": "a"}, {"doc_id": "memory_coalescing", "text": "b"}]
    spans = [span("retrieve", "retriever", output={"chunks": chunks})]
    api_ = API

    def run(answer):
        return d.citation_correctness(ctx(api_, s, answer=answer, all_spans=spans))

    assert run("Branches serialize [warp_divergence]; see also [1] and [2].")[0] == 1.0
    value, reason = run("See [3] and docs/warp_divergence.md and [bank_conflicts].")
    assert value == pytest.approx(1 / 3) and "[3]" in reason and "bank_conflicts" in reason
    with pytest.raises(Skip):
        run("No citations here.")
    with pytest.raises(Skip):
        d.citation_correctness(ctx(api_, s, answer="[1]", all_spans=[]))


def test_citations_ignore_code_and_subscripts():
    s = GoldenSample(id="c", category="knowledge", question="q")
    chunks = [{"doc_id": "memory_coalescing", "text": "a"}, {"doc_id": "bank_conflicts", "text": "b"}]
    spans = [span("retrieve", "retriever", output={"chunks": chunks})]
    answer = ("Strided reads like `a[i * stride]` waste bandwidth [memory_coalescing]. Pad the tile:\n"
              "```cuda\n__shared__ float tile[32][33];\n```\nand index tile[ty][tx] [2].")
    value, reason = d.citation_correctness(ctx(API, s, answer=answer, all_spans=spans))
    assert value == 1.0 and reason.startswith("2/2")  # only [memory_coalescing] and [2] are citations
