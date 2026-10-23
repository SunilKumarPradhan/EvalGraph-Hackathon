"""FastAPI app. Run:  uv run uvicorn evalgraph.api.main:app --reload --port 8000"""

from __future__ import annotations

import logging

from fastapi import APIRouter, FastAPI, HTTPException

from .. import __version__, storage
from ..core.schemas import Health, Job
from ..jobs import get_jobs
from ..settings import get_settings
from .routes_analysis import router as analysis_router
from .routes_runs import router as runs_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(title="EvalGraph", version=__version__)
api = APIRouter(prefix="/api")


@api.get("/health", response_model=Health)
def health() -> Health:
    s = get_settings()
    return Health(mock=s.mock_llm, replay=s.replay, model=s.model, judge_model=s.judge_model)


@api.get("/jobs/{job_id}", response_model=Job)
def job(job_id: str) -> Job:
    try:
        return get_jobs().get(job_id)
    except storage.NotFound:
        raise HTTPException(status_code=404, detail=f"job {job_id} not found")


api.include_router(analysis_router)
api.include_router(runs_router)
app.include_router(api)
