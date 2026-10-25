"""PHASE-08 replay mode: record a (mock) demo, then replay it through the API with compressed timing."""

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from evalgraph import replay
from evalgraph.api.main import app
from evalgraph.settings import get_settings

client = TestClient(app)
ARCHER = Path(__file__).resolve().parents[2].parent / "cuda-archer"


def _wait(job_id: str, timeout: float = 120) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.1)
    raise AssertionError("job did not finish")


@pytest.fixture(scope="module")
def recording(tmp_path_factory) -> tuple[Path, dict]:
    out = tmp_path_factory.mktemp("demo")
    manifest = replay.record({"local_path": str(ARCHER)}, ["baseline", "fix-safety"], 6, out)
    return out, manifest


@pytest.fixture
def replaying(recording, monkeypatch):
    out, manifest = recording
    monkeypatch.setenv("EVALGRAPH_DEMO_DIR", str(out))
    monkeypatch.setattr(get_settings(), "replay", True)
    monkeypatch.setattr(replay, "ANALYZE_REPLAY_S", 0.6)
    monkeypatch.setattr(replay, "RUN_REPLAY_S", 0.6)
    return manifest


def test_recording_is_honest_about_mock(recording):
    _, m = recording
    assert m["mock"] is True and len(m["runs"]) == 2  # tests always record in mock mode
    assert all(e["log"] and e["log"][-1]["t"] >= e["log"][0]["t"] for e in m["analyses"] + m["runs"])


def test_replay_analysis_and_runs(replaying):
    m = replaying
    rec = m["analyses"][0]
    started = time.time()
    # Wi-Fi-off fallback: a local folder with the recorded repo's name replays the recorded analysis
    job = _wait(client.post("/api/analyze", json={"local_path": str(ARCHER)}).json()["job_id"])
    assert job["status"] == "done" and job["result_id"] == rec["analysis_id"]
    assert time.time() - started < 5  # compressed, not the recorded wall time
    assert job["log"] == [e["line"] for e in rec["log"] if e["line"]]
    assert client.get(f"/api/analyses/{rec['analysis_id']}").json()["replayed"] is True

    recorded = [tuple(a) for a in m["runs"][0]["attachments"]]
    subset = [{"node_id": n, "evaluator_id": e} for n, e in recorded[:3]]
    ids = {}
    for variant in ("baseline", "fix-safety"):
        body = {"analysis_id": rec["analysis_id"], "attachments": subset, "variant": variant, "limit": 6}
        job = _wait(client.post("/api/runs", json=body).json()["job_id"])
        assert job["status"] == "done"
        run = client.get(f"/api/runs/{job['result_id']}").json()
        assert run["replayed"] and run["mock"]  # a replayed mock recording stays labelled mock
        assert {(r["node_id"], r["evaluator_id"]) for r in run["results"]} == set(recorded[:3])
        ids[variant] = run["id"]
    rows = client.get(f"/api/runs/{ids['baseline']}/compare/{ids['fix-safety']}").json()["rows"]
    assert len(rows) == 3

    # an attachment that was never recorded is not invented: the request runs for real (here: mock)
    extra = subset + [{"node_id": "entrypoint:ask", "evaluator_id": "task_completion"}]
    job = _wait(client.post("/api/runs", json={"analysis_id": rec["analysis_id"], "attachments": extra, "limit": 2}).json()["job_id"])
    assert job["status"] == "done"
    assert client.get(f"/api/runs/{job['result_id']}").json()["replayed"] is False


def test_source_matching_rules():
    by_url = {"analyses": [{"analysis_id": "a_1", "name": "cuda-archer", "repo_url": "https://github.com/x/cuda-archer"}]}
    by_dir = {"analyses": [{"analysis_id": "a_2", "name": "CUDA-Archer", "repo_url": None}]}
    m = replay.match_analysis
    assert m({"repo_url": "https://github.com/x/cuda-archer.git/"}, by_url)["analysis_id"] == "a_1"
    assert m({"repo_url": "https://github.com/someone-else/cuda-archer"}, by_url) is None  # URL recordings need the URL
    assert m({"local_path": "C:/work/cuda-archer"}, by_url)["analysis_id"] == "a_1"           # Wi-Fi off fallback
    assert m({"repo_url": "https://github.com/x/cuda-archer"}, by_dir)["analysis_id"] == "a_2"  # folder recording, URL demo
    assert m({"repo_url": "https://github.com/x/other"}, by_dir) is None


def test_replay_never_substitutes_isolated_or_swept_runs():
    from evalgraph.core.schemas import RunRequest

    manifest = {"runs": [{"run_id": "r_1", "analysis_id": "a_1", "variant": "baseline", "attachments": [["n", "e"]]}]}
    att = [{"node_id": "n", "evaluator_id": "e"}]
    assert replay.match_run(RunRequest(analysis_id="a_1", attachments=att), manifest)["run_id"] == "r_1"
    assert replay.match_run(RunRequest(analysis_id="a_1", attachments=att, isolated=True), manifest) is None
    assert replay.match_run(RunRequest(analysis_id="a_1", attachments=att, patches=["h"], patch_values={"h": "2"}), manifest) is None
