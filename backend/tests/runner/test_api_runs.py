"""PHASE-06 end to end in mock mode: /runs validation, a CUDA Archer adapter run, a sandbox run, compare."""

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from evalgraph import storage
from evalgraph.analysis import run_analysis
from evalgraph.api.main import app
from evalgraph.core.schemas import AnalysisResult, RunResult

client = TestClient(app)
ARCHER = Path(__file__).resolve().parents[3].parent / "cuda-archer"
ZOO = Path(__file__).resolve().parents[1] / "understanding" / "zoo"


def _wait(job_id: str, timeout: float = 240) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.2)
    raise AssertionError("run did not finish")


@pytest.fixture(scope="module")
def archer_id() -> str:
    return run_analysis({"local_path": str(ARCHER)}, lambda *a, **k: None)


def test_validation_errors(archer_id):
    att = [{"node_id": "retriever:retrieve", "evaluator_id": "context_precision"}]
    assert client.post("/api/runs", json={"analysis_id": "a_00000000", "attachments": att}).status_code == 404
    r = client.post("/api/runs", json={"analysis_id": archer_id, "attachments": att, "variant": "nope"})
    assert r.status_code == 422 and "unknown variant" in r.json()["detail"]
    bad = [{"node_id": "retriever:retrieve", "evaluator_id": "routing_accuracy"}]
    r = client.post("/api/runs", json={"analysis_id": archer_id, "attachments": bad})
    assert r.status_code == 422 and "does not apply" in r.json()["detail"]
    assert client.post("/api/runs", json={"analysis_id": archer_id, "attachments": []}).status_code == 422
    nb = run_analysis({"local_path": str(ZOO / "notebook_rag")}, lambda *a, **k: None)
    r = client.post("/api/runs", json={"analysis_id": nb, "attachments": att})
    assert r.status_code == 409 and "not runnable" in r.json()["detail"]


def test_archer_mock_run_and_compare(archer_id):
    atts = [{"node_id": "retriever:retrieve", "evaluator_id": "context_precision"},
            {"node_id": "retriever:retrieve", "evaluator_id": "context_relevance"},
            {"node_id": "guardrail:screen_input", "evaluator_id": "injection_resistance"},
            {"node_id": "router:route_query", "evaluator_id": "routing_accuracy"},
            {"node_id": "entrypoint:ask", "evaluator_id": "latency"},
            {"node_id": "tool:occupancy_calculator", "evaluator_id": "tool_correctness"}]
    run_ids = {}
    for variant in ("baseline", "fix-safety"):
        body = {"analysis_id": archer_id, "attachments": atts, "variant": variant,
                "categories": ["safety", "occupancy", "knowledge"], "limit": None}
        job = _wait(client.post("/api/runs", json=body).json()["job_id"])
        assert job["status"] == "done", job
        run_ids[variant] = job["result_id"]
    run = RunResult.model_validate(client.get(f"/api/runs/{run_ids['baseline']}").json())
    assert run.mock and run.dataset["size"] == 24 and run.models["judge"]
    res = {(r.node_id, r.evaluator_id): r for r in run.results}
    assert len(res) == len(atts) and all(r.status != "error" for r in res.values())
    assert res[("guardrail:screen_input", "injection_resistance")].status == "red"  # baseline: guardrail off
    assert res[("retriever:retrieve", "context_precision")].n > 0
    assert any(b.node_id == "guardrail:screen_input" and b.suggested_variant == "fix-safety" for b in run.bottlenecks)
    judged = [s for s in run.samples if s.evaluator_id == "context_relevance"]
    assert judged and all("[MOCK]" in s.reason for s in judged)  # mock judge output is never disguised

    cmp = client.get(f"/api/runs/{run_ids['baseline']}/compare/{run_ids['fix-safety']}").json()
    row = next(r for r in cmp["rows"] if r["evaluator_id"] == "injection_resistance")
    assert row["other"] > row["base"] and row["better"] is True  # the fix variant blocks the attacks
    listed = client.get(f"/api/runs?analysis_id={archer_id}").json()
    assert {r["id"] for r in listed} >= set(run_ids.values())


def test_sandbox_run_path(monkeypatch):
    # LocalSandbox (tests only, CLAUDE.md) stands in for Docker; same harness, plan, probes and attribution.
    from evalgraph.runtime import adapter as sandbox_adapter
    from evalgraph.runtime.sandbox import LocalSandbox

    monkeypatch.setattr(sandbox_adapter, "DockerSandbox", LocalSandbox)
    aid = run_analysis({"local_path": str((ZOO / "openai_agent").resolve())}, lambda *a, **k: None)
    a = storage.load("analyses", aid, AnalysisResult)
    a.runtime.plan["install"] = "none"  # never pip-install on the host
    storage.save("analyses", a)
    node = next(n.id for n in a.graph.nodes if n.symbol == "run_agent")
    ep = next(n.id for n in a.graph.nodes if n.type == "entrypoint")
    atts = [{"node_id": node, "evaluator_id": "latency"}, {"node_id": ep, "evaluator_id": "injection_resistance"},
            {"node_id": ep, "evaluator_id": "task_completion"}]
    job = _wait(client.post("/api/runs", json={"analysis_id": aid, "attachments": atts}).json()["job_id"])
    assert job["status"] == "done", job
    run = RunResult.model_validate(client.get(f"/api/runs/{job['result_id']}").json())
    assert run.variant == "as-is" and run.dataset["name"] == "synthetic" and run.dataset["size"] == len(a.runtime.dataset)
    res = {r.evaluator_id: r for r in run.results}
    assert res["latency"].n == run.dataset["size"]  # the traced function ran for every sample
    # the two synthetic injection probes + the probes aimed at this app's injection hotspots (PHASE-11)
    refuse = [s for s in a.runtime.dataset if s.expected_behavior == "refuse" and s.category != "off_topic"]
    assert len(refuse) > 2 and res["injection_resistance"].n == len(refuse)
    assert res["task_completion"].n == run.dataset["size"]
