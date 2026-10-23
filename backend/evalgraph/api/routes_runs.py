from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .. import storage
from ..core.schemas import (
    JobRef,
    RunComparison,
    RunRequest,
    RunResult,
    SweepRequest,
    SweepResult,
)
from ..jobs import get_jobs
from ..replay import load_manifest, match_run, replay_run
from ..runner.compare import compare
from ..runner.run import RunError, execute_run, execute_sweep, prepare, prepare_sweep
from ..settings import get_settings

router = APIRouter()


def _load(run_id: str) -> RunResult:
    try:
        return storage.load("runs", run_id, RunResult)
    except storage.NotFound:
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")


@router.post("/runs", response_model=JobRef)
def start_run(req: RunRequest) -> JobRef:
    if get_settings().replay and (manifest := load_manifest()) and (entry := match_run(req, manifest)):
        job = get_jobs().submit("run", lambda report: replay_run(req, entry, report))
        return JobRef(job_id=job.id)
    try:
        prepare(req)  # validate now so the UI gets 404/409/422 instead of a failed job
    except RunError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.detail)
    job = get_jobs().submit("run", lambda report: execute_run(req, report))
    return JobRef(job_id=job.id)


@router.post("/runs/sweep", response_model=JobRef)
def start_sweep(req: SweepRequest) -> JobRef:
    try:
        prepare_sweep(req)
    except RunError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.detail)
    job = get_jobs().submit("run", lambda report: execute_sweep(req, report))
    return JobRef(job_id=job.id)


@router.get("/sweeps/{sweep_id}", response_model=SweepResult)
def get_sweep(sweep_id: str) -> SweepResult:
    try:
        return storage.load("sweeps", sweep_id, SweepResult)
    except storage.NotFound:
        raise HTTPException(status_code=404, detail=f"sweep {sweep_id} not found")


@router.get("/runs")
def list_runs(analysis_id: str | None = None) -> list[dict]:
    return [{"id": r["id"], "variant": r["variant"], "created_at": r["created_at"], "overall": r.get("overall"),
             "mock": r.get("mock", False)}
            for r in storage.list_raw("runs") if "analysis_id" in r and analysis_id in (None, r["analysis_id"])]


@router.get("/runs/{run_id}", response_model=RunResult)
def get_run(run_id: str) -> RunResult:
    return _load(run_id)


@router.get("/runs/{base_id}/compare/{other_id}", response_model=RunComparison)
def compare_runs(base_id: str, other_id: str) -> RunComparison:
    base, other = _load(base_id), _load(other_id)
    if base.analysis_id != other.analysis_id:
        raise HTTPException(status_code=422, detail="runs belong to different analyses")
    return compare(base, other)
