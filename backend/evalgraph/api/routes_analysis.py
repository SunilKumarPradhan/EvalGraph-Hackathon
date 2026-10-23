from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from .. import storage
from ..analysis import run_analysis
from ..analyzer.clone import SourceError, validate_url
from ..core.schemas import (
    AnalysisResult,
    AnalyzeRequest,
    Attachment,
    EvaluatorSpec,
    JobRef,
    RuntimeInfo,
    RuntimeUpdate,
)
from ..evaluators.catalog import CATALOG, applies
from ..hotspots.patches import unified_diff
from ..jobs import get_jobs
from ..replay import load_manifest, match_analysis, replay_analysis
from ..runtime.plan import RunPlan
from ..runtime.secrets import load_secrets, save_secrets
from ..settings import BACKEND_ROOT, get_settings

router = APIRouter()


@router.post("/analyze", response_model=JobRef)
def analyze(req: AnalyzeRequest) -> JobRef:
    if bool(req.repo_url) == bool(req.local_path):
        raise HTTPException(status_code=422, detail="provide exactly one of repo_url or local_path")
    try:
        if req.repo_url:
            source = {"repo_url": validate_url(req.repo_url)}
        else:
            root = Path(req.local_path)
            root = (root if root.is_absolute() else BACKEND_ROOT / root).resolve()
            if not root.is_dir():
                raise SourceError(f"local_path not found: {root}")
            source = {"local_path": str(root)}
    except SourceError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    if get_settings().replay and (manifest := load_manifest()) and (entry := match_analysis(source, manifest)):
        job = get_jobs().submit("analyze", lambda report: replay_analysis(entry, report))
        return JobRef(job_id=job.id)
    job = get_jobs().submit("analyze", lambda report: run_analysis(source, report))
    return JobRef(job_id=job.id)


@router.get("/analyses")
def list_analyses() -> list[dict]:
    return [{"id": a["id"], "name": a["source"]["name"], "created_at": a["created_at"],
             "runnable": a["summary"]["runnable"]} for a in storage.list_raw("analyses") if "source" in a]


@router.get("/analyses/{analysis_id}", response_model=AnalysisResult)
def get_analysis(analysis_id: str) -> AnalysisResult:
    try:
        return storage.load("analyses", analysis_id, AnalysisResult)
    except storage.NotFound:
        raise HTTPException(status_code=404, detail=f"analysis {analysis_id} not found")


@router.get("/evaluators", response_model=list[EvaluatorSpec])
def evaluators() -> list[EvaluatorSpec]:
    return CATALOG


@router.get("/analyses/{analysis_id}/patch", response_class=PlainTextResponse)
def download_patch(analysis_id: str, hotspots: str = "") -> PlainTextResponse:
    """Zero-context unified diff of the rule-based fixes (all, or the comma-separated hotspot ids)."""
    analysis = get_analysis(analysis_id)
    wanted = {h for h in hotspots.split(",") if h}
    edits = {(e.file, e.line): e for h in analysis.hotspots if h.patch and (not wanted or h.id in wanted) for e in h.patch}
    if not edits:
        raise HTTPException(status_code=404, detail="no rule-based patch for the requested hotspots")
    name = f"evalgraph-{analysis.source.name}-fixes.patch"
    return PlainTextResponse(unified_diff(list(edits.values())), media_type="text/x-diff",
                             headers={"Content-Disposition": f'attachment; filename="{name}"'})


@router.get("/analyses/{analysis_id}/aimap", response_class=PlainTextResponse)
def download_aimap(analysis_id: str) -> PlainTextResponse:
    """The AI Map (PHASE-11): the whole analysis as compact, line-referenced text to paste into any LLM."""
    analysis = get_analysis(analysis_id)
    if not analysis.aimap:
        raise HTTPException(status_code=404, detail="this analysis has no AI map (made before PHASE-11)")
    name = f"evalgraph-{analysis.source.name}-aimap.md"
    return PlainTextResponse(analysis.aimap, media_type="text/markdown",
                             headers={"Content-Disposition": f'inline; filename="{name}"'})


EDITABLE_PLAN_KEYS = {"mode", "module", "function", "question_param", "fixed_args", "start_cmd", "port", "path", "method",
                      "body", "response_path", "install"}


@router.put("/analyses/{analysis_id}/runtime", response_model=AnalysisResult)
def update_runtime(analysis_id: str, update: RuntimeUpdate) -> AnalysisResult:
    """Run setup drawer: user edits to the inferred run plan and the generated questions. Marked edited=true."""
    analysis = get_analysis(analysis_id)
    if analysis.summary.adapter not in (None, "sandbox"):
        raise HTTPException(status_code=409, detail="this repo runs through its own adapter; its plan is not editable")
    runtime = analysis.runtime or RuntimeInfo()
    if update.plan is not None:
        unknown = set(update.plan) - EDITABLE_PLAN_KEYS
        if unknown:
            raise HTTPException(status_code=422, detail=f"not editable: {', '.join(sorted(unknown))}")
        try:
            plan = RunPlan(**{**runtime.plan, **update.plan})
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"invalid plan: {exc}")
        if plan.mode == "python_function" and not (plan.module and plan.function and plan.question_param):
            raise HTTPException(status_code=422, detail="a function plan needs module, function and question_param")
        if plan.mode == "http" and not (plan.start_cmd and plan.path):
            raise HTTPException(status_code=422, detail="an http plan needs start_cmd and path")
        runtime.plan = {**plan.model_dump(), "edited": True,
                        "reason": plan.reason if runtime.plan.get("edited") else f"edited by user (inferred: {plan.reason})"}
    if update.dataset is not None:
        ids = [s.id for s in update.dataset]
        if len(ids) != len(set(ids)) or any(not s.question.strip() for s in update.dataset):
            raise HTTPException(status_code=422, detail="questions need unique ids and non-empty text")
        runtime.dataset = update.dataset
    analysis.runtime = runtime
    runnable = runtime.plan.get("mode") not in (None, "unsupported") and bool(runtime.dataset)
    if analysis.summary.adapter in (None, "sandbox"):
        analysis.summary.runnable = runnable
        analysis.summary.adapter = "sandbox" if runnable else None
        analysis.summary.variants = ["as-is"] if runnable else []
    storage.save("analyses", analysis)
    return analysis


@router.put("/analyses/{analysis_id}/attachments", response_model=AnalysisResult)
def update_attachments(analysis_id: str, attachments: list[Attachment]) -> AnalysisResult:
    analysis = get_analysis(analysis_id)
    nodes = {n.id: n for n in analysis.graph.nodes}
    for a in attachments:
        node = nodes.get(a.node_id)
        if node is None or not applies(a.evaluator_id, node.type):
            raise HTTPException(status_code=422, detail=f"invalid attachment {a.evaluator_id} -> {a.node_id}")
    analysis.attachments = list({(a.node_id, a.evaluator_id): a for a in attachments}.values())
    storage.save("analyses", analysis)
    return analysis


@router.put("/analyses/{analysis_id}/dismissed", response_model=AnalysisResult)
def update_dismissed(analysis_id: str, hotspot_ids: list[str]) -> AnalysisResult:
    analysis = get_analysis(analysis_id)
    known = {h.id for h in analysis.hotspots}
    unknown = [h for h in hotspot_ids if h not in known]
    if unknown:
        raise HTTPException(status_code=422, detail=f"unknown hotspot(s): {', '.join(unknown)}")
    analysis.dismissed_hotspots = list(dict.fromkeys(hotspot_ids))
    storage.save("analyses", analysis)
    return analysis


@router.get("/analyses/{analysis_id}/secrets")
def list_secrets(analysis_id: str) -> dict:
    """Names only: secret values are never sent back."""
    analysis = get_analysis(analysis_id)
    wanted = (analysis.runtime.plan.get("app_env") if analysis.runtime else None) or []
    return {"stored": sorted(load_secrets(analysis.id)), "wanted": wanted}


@router.put("/analyses/{analysis_id}/secrets")
def put_secrets(analysis_id: str, changes: dict[str, str | None]) -> dict:
    analysis = get_analysis(analysis_id)
    try:
        stored = save_secrets(analysis.id, changes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    wanted = (analysis.runtime.plan.get("app_env") if analysis.runtime else None) or []
    return {"stored": stored, "wanted": wanted}
