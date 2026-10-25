"""Rule-based fix patches (conf_07 Q-31): generation from evidence, application in the sandbox, API."""

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from evalgraph import storage
from evalgraph.analysis import run_analysis
from evalgraph.api.main import app
from evalgraph.core.schemas import AnalysisResult, PatchEdit, RunResult
from evalgraph.hotspots.patches import edit_for, unified_diff

client = TestClient(app)
ZOO = Path(__file__).resolve().parents[1] / "understanding" / "zoo"


@pytest.mark.parametrize(("rule", "line", "expected"), [
    ("RAG_TOPK_NO_RERANK", "DEFAULT_TOP_K = 10  # wide", "DEFAULT_TOP_K = 4  # wide"),
    ("RAG_TOPK_NO_RERANK", 'r = s.as_retriever(search_kwargs={"k": 10})', 'r = s.as_retriever(search_kwargs={"k": 4})'),
    ("RAG_TOPK_NO_RERANK", "USE_RERANKER = False", "USE_RERANKER = True"),
    ("RAG_TOPK_NO_RERANK", "TOP_K = 5", None),                      # already small: no change proposed
    ("LLM_HIGH_TEMP", 'llm = ChatOpenAI(model="x", temperature=0.9)', 'llm = ChatOpenAI(model="x", temperature=0.2)'),
    ("LLM_HIGH_TEMP", "temperature: 0.9", "temperature: 0.2"),      # YAML config
    ("LLM_HIGH_TEMP", "const llm = new X({ temperature: 1 });", "const llm = new X({ temperature: 0.2 });"),
    ("EXT_NO_TIMEOUT", "HTTP_TIMEOUT: float | None = None  # c", "HTTP_TIMEOUT: float | None = 10  # c"),
    ("EXT_NO_TIMEOUT", 'return requests.get(f"https://x/{c}").text', 'return requests.get(f"https://x/{c}", timeout=10).text'),
    ("EXT_NO_TIMEOUT", "requests.get(url,", None),                 # call spans lines: not a literal fix
    ("MEMORY_UNBOUNDED", "MEMORY_WINDOW: int | None = None", "MEMORY_WINDOW: int | None = 6"),
    ("MEMORY_UNBOUNDED", "history.append(msg)", None),              # needs real code, so no patch
    ("EXTRA_LLM_HOP", "ENABLE_QUERY_REWRITE = True", "ENABLE_QUERY_REWRITE = False"),
    ("BIG_MODEL_FOR_CLASSIFY", 'ROUTER_MODEL = "large"', 'ROUTER_MODEL = "small"'),
])
def test_literal_edits(rule, line, expected):
    hit = edit_for(rule, line, "app.py")
    assert (hit[0] if hit else None) == expected


@pytest.mark.parametrize(("line", "expected"), [
    ('const res = await fetch("https://x/docs.json");',
     'const res = await fetch("https://x/docs.json", { signal: AbortSignal.timeout(10000) });'),
    ("(await fetch(`https://w/${city}?f=3`)).text()",
     "(await fetch(`https://w/${city}?f=3`, { signal: AbortSignal.timeout(10000) })).text()"),
    ('await fetch(url, { method: "POST", body })', 'await fetch(url, { signal: AbortSignal.timeout(10000), method: "POST", body })'),
    ("await fetch(url, opts)", None),                                   # options built elsewhere: not a literal fix
    ("await fetch(url, { signal })", None),                             # already has a signal
])
def test_js_fetch_timeout(line, expected):
    hit = edit_for("EXT_NO_TIMEOUT", line, "app/route.ts")
    assert (hit[0] if hit else None) == expected


def test_unified_diff():
    d = unified_diff([PatchEdit(file="a.py", line=3, before="K = 10", after="K = 4", summary="K 10 → 4")])
    assert d == "--- a/a.py\n+++ b/a.py\n@@ -3,1 +3,1 @@\n-K = 10\n+K = 4\n"


def _wait(job_id: str, timeout: float = 120) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.2)
    raise AssertionError("job did not finish")


def test_patched_sandbox_run_and_compare(monkeypatch):
    from evalgraph.runtime import adapter as sandbox_adapter
    from evalgraph.runtime.sandbox import LocalSandbox

    monkeypatch.setattr(sandbox_adapter, "DockerSandbox", LocalSandbox)  # tests only (CLAUDE.md)
    aid = run_analysis({"local_path": str((ZOO / "openai_agent").resolve())}, lambda *a, **k: None)
    a = storage.load("analyses", aid, AnalysisResult)
    a.runtime.plan["install"] = "none"
    storage.save("analyses", a)
    fix = next(h for h in a.hotspots if h.rule_id == "LLM_NO_MAX_TOKENS")
    assert fix.patch and fix.patch[0].after.rstrip().endswith("max_tokens=800)")
    original = (ZOO / "openai_agent" / fix.patch[0].file).read_text(encoding="utf-8")

    node = next(n.id for n in a.graph.nodes if n.symbol == "run_agent")
    atts = [{"node_id": node, "evaluator_id": "latency"}]
    base = _wait(client.post("/api/runs", json={"analysis_id": aid, "attachments": atts}).json()["job_id"])
    body = {"analysis_id": aid, "attachments": atts, "patches": [fix.id]}
    job = _wait(client.post("/api/runs", json=body).json()["job_id"])
    assert base["status"] == job["status"] == "done", job
    run = RunResult.model_validate(client.get(f"/api/runs/{job['result_id']}").json())
    assert run.patches == [fix.id] and run.variant == "as-is + 1 patch"
    assert any("Applied 1 edit" in line for line in job["log"]), job["log"]
    assert (ZOO / "openai_agent" / fix.patch[0].file).read_text(encoding="utf-8") == original  # the repo itself is untouched
    rows = client.get(f"/api/runs/{base['result_id']}/compare/{run.id}").json()["rows"]
    assert len(rows) == 1

    diff = client.get(f"/api/analyses/{aid}/patch", params={"hotspots": fix.id})
    assert diff.status_code == 200 and "+" in diff.text and "max_tokens=800" in diff.text


def test_patch_requests_are_validated(monkeypatch):
    from evalgraph.runtime import adapter as sandbox_adapter
    from evalgraph.runtime.sandbox import LocalSandbox

    monkeypatch.setattr(sandbox_adapter, "DockerSandbox", LocalSandbox)  # validation must not depend on Docker being up
    aid = run_analysis({"local_path": str((ZOO / "openai_agent").resolve())}, lambda *a, **k: None)
    a = storage.load("analyses", aid, AnalysisResult)
    node = next(n.id for n in a.graph.nodes if n.symbol == "run_agent")
    atts = [{"node_id": node, "evaluator_id": "latency"}]
    unpatchable = next(h.id for h in a.hotspots if not h.patch)
    for patches, detail in ([["nope"], "unknown hotspot"], [[unpatchable], "no rule-based patch"]):
        r = client.post("/api/runs", json={"analysis_id": aid, "attachments": atts, "patches": patches})
        assert r.status_code == 422 and detail in r.json()["detail"]


APP = Path(__file__).resolve().parents[1] / "runtime" / "fixtures" / "rag_app"


def _local(monkeypatch) -> str:
    from evalgraph.runtime import adapter as sandbox_adapter
    from evalgraph.runtime.sandbox import LocalSandbox

    monkeypatch.setattr(sandbox_adapter, "DockerSandbox", LocalSandbox)  # tests only (CLAUDE.md)
    aid = run_analysis({"local_path": str(APP)}, lambda *a, **k: None)
    a = storage.load("analyses", aid, AnalysisResult)
    a.runtime.plan["install"] = "none"
    storage.save("analyses", a)
    return aid


def test_sweep_runs_one_point_per_value(monkeypatch):
    aid = _local(monkeypatch)
    a = storage.load("analyses", aid, AnalysisResult)
    topk = next(h for h in a.hotspots if h.rule_id == "RAG_TOPK_NO_RERANK")
    edit = next(e for e in topk.patch if e.span)
    assert edit.sweep == ["2", "4", "6", "8"]
    atts = [{"node_id": "retriever:retrieve", "evaluator_id": "latency"}]
    body = {"analysis_id": aid, "hotspot_id": topk.id, "values": ["2", "6"], "attachments": atts, "limit": 2}
    job = _wait(client.post("/api/runs/sweep", json=body).json()["job_id"], timeout=240)
    assert job["status"] == "done", job
    sweep = client.get(f"/api/sweeps/{job['result_id']}").json()
    assert sweep["key"] == "TOP_K" and sweep["original"] == "10" and list(sweep["runs"]) == ["2", "6"]
    for value, rid in sweep["runs"].items():
        run = client.get(f"/api/runs/{rid}").json()
        assert run["variant"] == f"as-is + TOP_K={value}" and run["patches"] == [topk.id]
    base = client.get(f"/api/runs/{sweep['base_run_id']}").json()
    assert base["patches"] == [] and base["dataset"]["size"] == 2


def test_sweep_validation(monkeypatch):
    aid = _local(monkeypatch)
    a = storage.load("analyses", aid, AnalysisResult)
    atts = [{"node_id": "retriever:retrieve", "evaluator_id": "latency"}]
    unsweepable = next(h.id for h in a.hotspots if not any(e.span for e in h.patch))
    topk = next(h.id for h in a.hotspots if h.rule_id == "RAG_TOPK_NO_RERANK")
    cases = [({"hotspot_id": unsweepable}, "no sweepable literal"),
             ({"hotspot_id": topk, "values": [str(i) for i in range(7)]}, "1 to 6 values")]
    for extra, detail in cases:
        r = client.post("/api/runs/sweep", json={"analysis_id": aid, "attachments": atts, **extra})
        assert r.status_code == 422 and detail in r.json()["detail"], r.json()
    r = client.post("/api/runs", json={"analysis_id": aid, "attachments": atts, "patches": [topk],
                                       "patch_values": {topk: "abc"}})
    assert r.status_code == 422  # a non-numeric value is refused
