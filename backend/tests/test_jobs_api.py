import time

from fastapi.testclient import TestClient

from evalgraph.api.main import app
from evalgraph.jobs import JobManager

client = TestClient(app)


def _wait(manager, job_id, timeout=5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = manager.get(job_id)
        if job.status in ("done", "error"):
            return job
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_job_lifecycle_and_log():
    manager = JobManager()

    def work(report):
        report("Step one", 0.3, "did step one")
        report("Step two", 0.9, "did step two")
        return "a_12345678"

    job = _wait(manager, manager.submit("analyze", work).id)
    assert job.status == "done" and job.progress == 1.0 and job.result_id == "a_12345678"
    assert job.log == ["did step one", "did step two"]


def test_job_error_is_captured():
    manager = JobManager()

    def boom(report):
        raise ValueError("nope")

    job = _wait(manager, manager.submit("run", boom).id)
    assert job.status == "error" and "ValueError: nope" in job.error


def test_health_and_unknown_job():
    body = client.get("/api/health").json()
    assert body["ok"] is True and body["mock"] is True
    assert client.get("/api/jobs/j_deadbeef").status_code == 404
    assert len(client.get("/api/evaluators").json()) == 19
