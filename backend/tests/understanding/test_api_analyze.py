import time
from pathlib import Path

from fastapi.testclient import TestClient

from evalgraph.api.main import app

client = TestClient(app)
ARCHER = str(Path(__file__).resolve().parents[3].parent / "cuda-archer")
ZOO = Path(__file__).parent / "zoo"


def _wait(job_id, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.1)
    raise AssertionError("analysis job did not finish")


def test_analyze_local_path_end_to_end():
    r = client.post("/api/analyze", json={"local_path": ARCHER})
    assert r.status_code == 200
    job = _wait(r.json()["job_id"])
    assert job["status"] == "done", job["error"]
    assert job["log"][0].startswith("INGEST") and "runnable" in job["log"][-1]
    a = client.get(f"/api/analyses/{job['result_id']}").json()
    assert a["summary"]["runnable"] and a["summary"]["adapter"] == "cuda_archer" and a["mock"] is True
    assert len(a["summary"]["variants"]) == 10 and a["summary"]["headline"].startswith("Detected")
    assert any(item["id"] == a["id"] for item in client.get("/api/analyses").json())


def test_analyze_any_repo_is_sandbox_runnable():
    job = _wait(client.post("/api/analyze", json={"local_path": str(ZOO / "anthropic_chatbot")}).json()["job_id"])
    a = client.get(f"/api/analyses/{job['result_id']}").json()
    # any repo with a callable entry is runnable in the Docker sandbox (PHASE-06B)
    assert a["summary"]["runnable"] is True and a["summary"]["adapter"] == "sandbox"
    assert a["runtime"]["plan"]["function"] == "ask_claude" and a["runtime"]["dataset"]
    rules = {h["rule_id"] for h in a["hotspots"]}
    assert {"NO_GUARDRAIL", "HARDCODED_SECRET", "PROMPT_INJECTION_SURFACE", "NO_EVAL_SUITE"} <= rules
    assert all(h["suggested_variant"] is None for h in a["hotspots"])


def test_bad_requests():
    assert client.post("/api/analyze", json={}).status_code == 422
    assert client.post("/api/analyze", json={"repo_url": "https://x.com/a/b", "local_path": "."}).status_code == 422
    assert client.post("/api/analyze", json={"repo_url": "https://evil.example/a/b"}).status_code == 422
    assert client.post("/api/analyze", json={"local_path": "Z:/does/not/exist"}).status_code == 422
    assert client.get("/api/analyses/a_00000000").status_code == 404
    assert client.get("/api/analyses/..%2F..%2Fsecret").status_code == 404


def test_notebook_only_repo_stays_plan_mode():
    job = _wait(client.post("/api/analyze", json={"local_path": str(ZOO / "notebook_rag")}).json()["job_id"])
    a = client.get(f"/api/analyses/{job['result_id']}").json()
    assert a["summary"]["runnable"] is False and a["runtime"]["plan"]["mode"] == "unsupported"
