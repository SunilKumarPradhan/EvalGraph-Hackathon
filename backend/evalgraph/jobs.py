"""Background jobs with progress, persisted to data/jobs so the UI can poll them."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Protocol

from . import storage
from .core.ids import new_id
from .core.schemas import Job, JobKind

logger = logging.getLogger(__name__)


class Reporter(Protocol):
    def __call__(self, stage: str, progress: float, log_line: str | None = None) -> None: ...


JobFn = Callable[[Reporter], str]  # returns the result id


class JobManager:
    def __init__(self, max_workers: int = 2):
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="evalgraph-job")
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def _persist(self, job: Job) -> None:
        storage.save("jobs", job)

    def submit(self, kind: JobKind, fn: JobFn) -> Job:
        job = Job(id=new_id("j"), kind=kind)
        with self._lock:
            self._jobs[job.id] = job
        self._persist(job)
        self._pool.submit(self._run, job.id, fn)
        return job

    def _update(self, job_id: str, **changes) -> None:
        with self._lock:
            job = self._jobs[job_id]
            data = job.model_dump()
            log_line = changes.pop("log_line", None)
            data.update(changes)
            if log_line:
                data["log"] = [*job.log, log_line]
            job = Job.model_validate(data)
            self._jobs[job_id] = job
        self._persist(job)

    def _run(self, job_id: str, fn: JobFn) -> None:
        self._update(job_id, status="running", stage="Starting")

        def report(stage: str, progress: float, log_line: str | None = None) -> None:
            self._update(job_id, stage=stage, progress=max(0.0, min(1.0, progress)), log_line=log_line)

        try:
            result_id = fn(report)
            self._update(job_id, status="done", stage="Done", progress=1.0, result_id=result_id)
        except Exception as exc:
            logger.exception("job %s failed", job_id)
            self._update(job_id, status="error", stage="Failed", error=f"{type(exc).__name__}: {exc}")

    def get(self, job_id: str) -> Job:
        with self._lock:
            if job_id in self._jobs:
                return self._jobs[job_id]
        return storage.load("jobs", job_id, Job)


_manager: JobManager | None = None


def get_jobs() -> JobManager:
    global _manager
    if _manager is None:
        _manager = JobManager()
    return _manager
