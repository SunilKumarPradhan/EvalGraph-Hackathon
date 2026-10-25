"""execute_run: run the app on its dataset, map spans to the graph, score every attachment, diagnose bottlenecks.

Two ways to run the app (CONTRACTS §8): a registered adapter in-process (CUDA Archer, with variants and a labelled
golden set), or the Docker sandbox for any other runnable repo (synthetic dataset, variant "as-is").
Before the scored samples, one unscored warm-up request absorbs cold starts (model downloads, index builds), so
latency reflects steady state (conf_05 Q-52).
"""

from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime

from .. import storage
from ..core.ids import new_id
from ..core.schemas import (
    AnalysisResult,
    AppOutput,
    GoldenSample,
    NodeEvalResult,
    RunRequest,
    RunResult,
    SampleResult,
    SweepRequest,
    SweepResult,
)
from ..evaluators.base import SampleContext, Skip, missing_need
from ..evaluators.catalog import applies
from ..evaluators.registry import EVALUATORS
from ..hotspots.patches import with_value
from ..jobs import Reporter
from ..llm import LLMError, dial_reachable
from ..settings import get_settings
from .adapters import get_adapter
from .aggregate import aggregate
from .bottlenecks import diagnose
from .mapping import map_spans

logger = logging.getLogger(__name__)
WARMUP_ID = "_warmup"


class RunError(Exception):
    """A request that can't run; carries the HTTP status for the API."""

    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


@dataclass(kw_only=True)  # keyword-only: positional construction once silently swapped `edits` and `isolated`
class Prepared:
    analysis: AnalysisResult
    adapter: object
    variant: str
    attachments: list[tuple[str, str]] = field(default_factory=list)  # (node_id, evaluator_id), deduplicated
    patch_ids: list[str] = field(default_factory=list)                 # hotspot ids applied in the sandbox
    isolated: bool = False
    edits: list[dict] = field(default_factory=list)                    # their whole-line edits, de-duplicated


def _adapter_for(analysis: AnalysisResult):
    name = analysis.summary.adapter
    if name == "sandbox":
        from ..runtime.adapter import SandboxAdapter
        from ..runtime.sandbox import SandboxError

        try:
            return SandboxAdapter(analysis)
        except SandboxError as exc:
            raise RunError(409, str(exc)) from exc
    return get_adapter(name)


def prepare(req: RunRequest) -> Prepared:
    """Validate a run request without running anything (the API calls this before queueing the job)."""
    try:
        analysis = storage.load("analyses", req.analysis_id, AnalysisResult)
    except storage.NotFound as exc:
        raise RunError(404, f"analysis {req.analysis_id} not found") from exc
    if not analysis.summary.runnable:
        reason = (analysis.runtime.plan.get("reason") if analysis.runtime else None) or "no adapter or run plan"
        raise RunError(409, f"this repo is not runnable: {reason}")
    if not get_settings().mock_llm and not dial_reachable():
        raise RunError(503, "DIAL is unreachable from this machine (VPN off?). Connect the VPN, or use mock mode "
                            "(EVALGRAPH_MOCK_LLM=1) or replay (EVALGRAPH_REPLAY=1).")
    adapter = _adapter_for(analysis)
    if adapter is None:
        raise RunError(409, f"adapter '{analysis.summary.adapter}' is not installed on this machine")
    if not adapter.available:
        raise RunError(409, "Docker is not running: start Docker Desktop to run this repo in the sandbox"
                       if analysis.summary.adapter == "sandbox" else f"adapter '{adapter.name}' is unavailable")
    variants = analysis.summary.variants or adapter.variants()
    variant = req.variant
    if variant not in variants:
        if variant == "baseline" and len(variants) == 1:  # the UI default; single-variant runners (sandbox: "as-is")
            variant = variants[0]
        else:
            raise RunError(422, f"unknown variant '{req.variant}'; available: {', '.join(variants)}")
    patch_ids, edits = _patches(req, analysis)
    if patch_ids:
        if analysis.summary.adapter != "sandbox":
            raise RunError(422, "patches run in the Docker sandbox; this repo has its own adapter, use its variants")
        swept = [e for h in analysis.hotspots if h.id in (req.patch_values or {}) for e in h.patch if e.span]
        variant = (f"{variant} + {swept[0].summary.split(' ', 1)[0]}={req.patch_values[next(iter(req.patch_values))]}"
                   if swept and len(patch_ids) == 1
                   else f"{variant} + {len(patch_ids)} patch{'es' if len(patch_ids) > 1 else ''}")
    if req.isolated:
        if analysis.summary.adapter != "sandbox" or (analysis.runtime and analysis.runtime.plan.get("mode") != "python_function"):
            raise RunError(422, "isolated samples apply to sandbox runs that call a Python function")
        variant = f"{variant} (isolated)"
    if not req.attachments:
        raise RunError(422, "attach at least one evaluator")
    nodes = {n.id: n for n in analysis.graph.nodes}
    pairs: list[tuple[str, str]] = []
    for att in req.attachments:
        node = nodes.get(att.node_id)
        if node is None:
            raise RunError(422, f"unknown node '{att.node_id}'")
        if att.evaluator_id not in EVALUATORS:
            raise RunError(422, f"unknown evaluator '{att.evaluator_id}'")
        if not applies(att.evaluator_id, node.type):
            raise RunError(422, f"evaluator '{att.evaluator_id}' does not apply to a {node.type} node ({node.id})")
        if (att.node_id, att.evaluator_id) not in pairs:
            pairs.append((att.node_id, att.evaluator_id))
    return Prepared(analysis=analysis, adapter=adapter, variant=variant, attachments=pairs, patch_ids=patch_ids,
                    edits=edits, isolated=req.isolated)


def _patches(req: RunRequest, analysis: AnalysisResult) -> tuple[list[str], list[dict]]:
    """Requested hotspot ids -> their edits. Two hotspots may share an edit; conflicting edits on one line are refused."""
    if not req.patches:
        return [], []
    by_id = {h.id: h for h in analysis.hotspots}
    edits: dict[tuple[str, int], dict] = {}
    values = req.patch_values or {}
    for hid in values:
        if hid not in req.patches:
            raise RunError(422, f"patch_values for '{hid}' needs that hotspot in patches")
    for hid in dict.fromkeys(req.patches):
        h = by_id.get(hid)
        if h is None:
            raise RunError(422, f"unknown hotspot '{hid}'")
        if not h.patch:
            raise RunError(422, f"hotspot '{hid}' has no rule-based patch")
        chosen = h.patch
        if hid in values:  # a sweep point: only the sweepable literal, at this value
            sweepable = [e for e in h.patch if e.span]  # the same parameter may be set in several places
            if not sweepable:
                raise RunError(422, f"hotspot '{hid}' has no sweepable literal")
            try:
                chosen = [with_value(e, values[hid]) for e in sweepable]
            except ValueError as exc:
                raise RunError(422, str(exc)) from exc
        for e in chosen:
            key = (e.file, e.line)
            if key in edits and edits[key]["after"] != e.after:
                raise RunError(422, f"patches conflict on {e.file}:{e.line}; apply them one at a time")
            edits[key] = e.model_dump()
    return list(dict.fromkeys(req.patches)), list(edits.values())


def _run_app(p: Prepared, samples: list[GoldenSample], report: Reporter) -> dict[str, AppOutput | str]:
    """sample id -> output, or an error message when the app failed on that sample."""
    adapter, n = p.adapter, len(samples)
    results: dict[str, AppOutput | str] = {}
    if hasattr(adapter, "run_batch"):
        report(f"Running {p.analysis.source.name} in the Docker sandbox ({n} samples"
               + (", one process each)" if p.isolated else " + 1 warm-up)"), 0.05,
               "Sandbox started: clone, install, run")
        if p.isolated:  # every sample starts cold in its own process: a warm-up would warm nothing
            outs = adapter.run_batch(samples, "as-is", p.edits, isolated=True)
        else:
            warm = samples[0].model_copy(update={"id": WARMUP_ID})
            outs = adapter.run_batch([warm, *samples], "as-is", p.edits)[1:]
        if p.edits:
            report("Patched run finished", 0.5, f"Applied {len(p.edits)} edit(s) from {len(p.patch_ids)} hotspot fix(es)")
        for s, out in zip(samples, outs):
            entry_error = next((sp.error for sp in out.spans if sp.component == "sandbox_request" and sp.error), None)
            results[s.id] = entry_error or out
        blocked = getattr(adapter, "last_egress_blocked", [])
        report("Sandbox run finished", 0.55, f"App answered {sum(isinstance(v, AppOutput) for v in results.values())}/{n} samples"
               + (f" · egress blocked: {', '.join(blocked)}" if blocked else ""))
        return results
    label = f"{adapter.name} ({p.variant})"
    report(f"Warming up {label}", 0.03)
    try:
        adapter.run(samples[0], p.variant)
    except Exception as exc:  # the scored run reports per-sample errors itself
        logger.warning("warm-up failed: %s", exc)

    def one(sample: GoldenSample) -> tuple[str, AppOutput | str]:
        try:
            return sample.id, adapter.run(sample, p.variant)
        except Exception as exc:  # a crash is a result too
            return sample.id, f"{type(exc).__name__}: {exc}"

    with ThreadPoolExecutor(max_workers=get_settings().max_concurrency) as pool:
        futures = [pool.submit(one, s) for s in samples]
        for i, fut in enumerate(as_completed(futures), 1):
            sid, out = fut.result()
            results[sid] = out
            report(f"Running {label} {i}/{n}", 0.05 + 0.5 * i / n)
    failed = sum(isinstance(v, str) for v in results.values())
    report(f"Ran {n} samples", 0.55, f"Ran {label} on {n} samples" + (f" ({failed} failed)" if failed else ""))
    return results


def _score(p: Prepared, samples: list[GoldenSample], outputs: dict[str, AppOutput | str],
           report: Reporter) -> tuple[list[NodeEvalResult], list[SampleResult]]:
    nodes = {n.id: n for n in p.analysis.graph.nodes}
    mapped = {sid: map_spans(p.analysis.graph.nodes, out.spans)[0]
              for sid, out in outputs.items() if isinstance(out, AppOutput)}
    canary = getattr(p.adapter, "secret_canary", None)
    report("Mapping spans to architecture", 0.58, "Mapped runtime spans to graph nodes")

    tasks: list[tuple[int, object, SampleContext]] = []
    per_att: list[list[SampleResult]] = [[] for _ in p.attachments]
    skips: list[dict[str, int]] = [{} for _ in p.attachments]
    for i, (node_id, eval_id) in enumerate(p.attachments):
        ev, node = EVALUATORS[eval_id], nodes[node_id]
        for s in samples:
            out = outputs[s.id]
            if isinstance(out, str):
                per_att[i].append(SampleResult(sample_id=s.id, node_id=node_id, evaluator_id=eval_id, error=out[:500]))
                continue
            need = missing_need(ev.spec, s)
            if need:
                skips[i][f"lack {need}"] = skips[i].get(f"lack {need}", 0) + 1
                continue
            spans = mapped[s.id].get(node_id, [])
            if ev.needs_span and not spans:
                skips[i]["the node did not run"] = skips[i].get("the node did not run", 0) + 1
                continue
            tasks.append((i, ev, SampleContext(sample=s, node=node, node_spans=spans, output=out, canary=canary)))

    def score(task) -> tuple[int, SampleResult | str]:
        i, ev, ctx = task
        base = {"sample_id": ctx.sample.id, "node_id": ctx.node.id, "evaluator_id": ev.spec.id}
        try:
            value, reason = ev.fn(ctx)
            return i, SampleResult(**base, value=value, reason=reason)
        except Skip as exc:
            return i, str(exc)
        except (LLMError, ValueError, KeyError, TypeError) as exc:
            return i, SampleResult(**base, error=f"{type(exc).__name__}: {exc}"[:500])

    total = len(tasks) or 1
    done = 0
    with ThreadPoolExecutor(max_workers=get_settings().max_concurrency) as pool:
        futures = [pool.submit(score, t) if t[1].is_judge else None for t in tasks]
        for t, fut in zip(tasks, futures):
            i, res = fut.result() if fut else score(t)
            if isinstance(res, str):
                skips[i][res] = skips[i].get(res, 0) + 1
            else:
                per_att[i].append(res)
            done += 1
            if done % 10 == 0 or done == total:
                report(f"Scoring {done}/{total}", 0.6 + 0.3 * done / total)

    results: list[NodeEvalResult] = []
    for i, (node_id, eval_id) in enumerate(p.attachments):
        na = "; ".join(f"{count} sample(s): {why}" for why, count in skips[i].items())
        results.append(aggregate(EVALUATORS[eval_id].spec, node_id, per_att[i], na_reason=f"nothing to score ({na})" if na else ""))
    report("Aggregated scores", 0.92, f"Scored {len(p.attachments)} attachment(s)")
    return results, [r for rs in per_att for r in rs]


def select_samples(samples: list[GoldenSample], limit: int | None) -> list[GoldenSample]:
    """First `limit` samples taken round-robin across categories, in dataset order, so a quick run still exercises
    every part of the app (the first 10 of a dataset are often all one category)."""
    if not limit or limit >= len(samples):
        return samples
    by_cat: dict[str, list[GoldenSample]] = {}
    for s in samples:
        by_cat.setdefault(s.category, []).append(s)
    picked: set[str] = set()
    depth = 0
    while len(picked) < limit:
        for group in by_cat.values():
            if depth < len(group) and len(picked) < limit:
                picked.add(group[depth].id)
        depth += 1
    return [s for s in samples if s.id in picked]


def _scaled(report: Reporter, lo: float, hi: float, prefix: str) -> Reporter:
    def inner(stage: str, progress: float, line: str | None = None) -> None:
        report(f"{prefix}: {stage}", lo + (hi - lo) * progress, f"{prefix}: {line}" if line else None)
    return inner


def prepare_sweep(req: SweepRequest) -> tuple[AnalysisResult, list[str], str]:
    """Validate a sweep: (analysis, values, swept key). Raises RunError."""
    try:
        analysis = storage.load("analyses", req.analysis_id, AnalysisResult)
    except storage.NotFound as exc:
        raise RunError(404, f"analysis {req.analysis_id} not found") from exc
    if analysis.summary.adapter != "sandbox":
        raise RunError(422, "sweeps run in the Docker sandbox; this repo has its own adapter, use its variants")
    h = next((x for x in analysis.hotspots if x.id == req.hotspot_id), None)
    sweepable = [e for e in (h.patch if h else []) if e.span]
    if not sweepable:
        raise RunError(422, f"hotspot '{req.hotspot_id}' has no sweepable literal")
    values = list(dict.fromkeys(req.values or sweepable[0].sweep))
    if not 1 <= len(values) <= 6:
        raise RunError(422, "sweep 1 to 6 values")
    return analysis, values, sweepable[0].summary.split(" ", 1)[0]


def execute_sweep(req: SweepRequest, report: Reporter) -> str:
    """Baseline (as-is) + one patched run per value; every point is an ordinary, comparable RunResult."""
    analysis, values, key = prepare_sweep(req)
    h = next(x for x in analysis.hotspots if x.id == req.hotspot_id)
    first = next(e for e in h.patch if e.span)
    original = re.match(r"-?\d+(?:\.\d+)?|None|null", first.before[first.span[0]:])  # the value in the code
    steps = len(values) + (0 if req.base_run_id else 1)
    common = {"analysis_id": analysis.id, "attachments": req.attachments, "limit": req.limit, "isolated": req.isolated}
    i = 0
    if req.base_run_id:
        base = storage.load("runs", req.base_run_id, RunResult)
        if base.analysis_id != analysis.id or base.patches:
            raise RunError(422, "the baseline must be an unpatched run of the same analysis")
        base_id = base.id
    else:
        base_id = execute_run(RunRequest(**common), _scaled(report, 0, 1 / steps, "baseline"))
        i = 1
    runs: dict[str, str] = {}
    for v in values:
        runs[v] = execute_run(RunRequest(**common, patches=[h.id], patch_values={h.id: v}),
                              _scaled(report, i / steps, (i + 1) / steps, f"{key}={v}"))
        i += 1
    sweep = SweepResult(id=new_id("s"), analysis_id=analysis.id, hotspot_id=h.id, key=key,
                        original=original.group(0) if original else "?", base_run_id=base_id, runs=runs,
                        created_at=datetime.now(UTC).isoformat(timespec="seconds"))
    storage.save("sweeps", sweep)
    report("Done", 1.0, f"Sweep of {key} over {', '.join(values)} finished")
    return sweep.id


def execute_run(req: RunRequest, report: Reporter) -> str:
    start = time.perf_counter()
    p = prepare(req)
    samples = p.adapter.load_dataset()
    if req.categories:
        samples = [s for s in samples if s.category in req.categories]
    samples = select_samples(samples, req.limit)
    if not samples:
        raise RunError(422, "no samples match the request")
    outputs = _run_app(p, samples, report)
    results, sample_results = _score(p, samples, outputs, report)
    report("Diagnosing bottlenecks", 0.95)
    bottlenecks = diagnose(p.analysis, results, sample_results)
    scores = [r.value for r in results if r.unit == "score" and r.value is not None]
    settings = get_settings()
    models = dict(getattr(p.adapter, "models", dict)())
    models["judge"] = settings.judge_model
    run = RunResult(
        id=new_id("r"), analysis_id=p.analysis.id, variant=p.variant, patches=p.patch_ids,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        dataset={"name": p.adapter.dataset_name, "size": len(samples), "warmup": not p.isolated, "isolated": p.isolated,
                 "failed": sum(isinstance(v, str) for v in outputs.values()),
                 "egress_blocked": getattr(p.adapter, "last_egress_blocked", []),
                 # drill-down previews for the UI (question, answer or error, category)
                 "samples": {s.id: {"question": s.question[:400], "category": s.category,
                                    "answer": outputs[s.id].answer[:600] if isinstance(outputs[s.id], AppOutput) else "",
                                    "error": outputs[s.id] if isinstance(outputs[s.id], str) else None}
                             for s in samples}},
        results=results, samples=sample_results, bottlenecks=bottlenecks,
        overall=sum(scores) / len(scores) if scores else None,
        duration_s=round(time.perf_counter() - start, 2), models=models, mock=settings.mock_llm)
    storage.save("runs", run)
    report("Done", 1.0, f"{len(bottlenecks)} bottleneck(s) found")
    return run.id
