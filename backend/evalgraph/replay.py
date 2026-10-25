"""Replay mode (EVALGRAPH_REPLAY=1): serve a recorded run of the demo so the stage demo survives no Wi-Fi / no VPN.

Recording (`python -m evalgraph.replay record`, wrapped by scripts/record_demo.ps1) analyzes the demo repo and runs the
requested variants in-process, capturing every job stage with its timestamp, then writes fixtures/demo/:
  manifest.json                       recorded_at, mock, models, analyses[], runs[]
  analyses/<id>.json, runs/<id>.json  the stored results, unchanged

When replaying, POST /analyze for a recorded source and POST /runs for a recorded (analysis, variant) play the recorded
job log back with its real relative timings, compressed to a short total, and return the recorded result marked
`replayed: true`. Anything not in the manifest runs normally. Nothing is invented: a run request is replayed only if
every requested attachment was measured in the recording, and the result is cut down to exactly those attachments.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path

from . import storage
from .core.ids import new_id
from .core.schemas import AnalysisResult, RunRequest, RunResult
from .settings import FIXTURES_DIR, get_settings

DEMO_DIR = FIXTURES_DIR / "demo"


def demo_dir() -> Path:
    """The recording to replay (EVALGRAPH_DEMO_DIR overrides fixtures/demo, e.g. for tests)."""
    return Path(os.environ.get("EVALGRAPH_DEMO_DIR") or DEMO_DIR)
ANALYZE_REPLAY_S = 8.0
RUN_REPLAY_S = 10.0


def _norm_url(url: str | None) -> str:
    return re.sub(r"(\.git)?/*$", "", (url or "").strip().lower())


def load_manifest(demo: Path | None = None) -> dict | None:
    path = (demo or demo_dir()) / "manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _play(log: list[dict], target_s: float, report) -> None:
    """Re-emit recorded stages with their relative timing, scaled down to at most target_s seconds."""
    total = log[-1]["t"] if log else 0.0
    scale = min(1.0, target_s / total) if total > 0 else 0.0
    start = time.perf_counter()
    for entry in log:
        delay = entry["t"] * scale - (time.perf_counter() - start)
        if delay > 0:
            time.sleep(delay)
        report(entry["stage"], entry["progress"], entry.get("line"))


def match_analysis(source: dict, manifest: dict) -> dict | None:
    """Recorded analysis for this source: the same repo URL; or a local folder with the repo's name (Wi-Fi off fallback);
    or, when the recording was made from a local folder (no URL known), a URL whose repo name matches it."""
    url = _norm_url(source.get("repo_url"))
    local = Path(source["local_path"]).name.lower() if source.get("local_path") else None
    for entry in manifest.get("analyses", []):
        recorded_url = _norm_url(entry.get("repo_url"))
        name = entry["name"].lower()
        if url and (url == recorded_url or (not recorded_url and url.rsplit("/", 1)[-1] == name)):
            return entry
        if local and local == name:
            return entry
    return None


def replay_analysis(entry: dict, report, demo: Path | None = None) -> str:
    demo = demo or demo_dir()
    analysis = AnalysisResult.model_validate_json((demo / "analyses" / f"{entry['analysis_id']}.json").read_text("utf-8"))
    _play(entry.get("log", []), ANALYZE_REPLAY_S, report)
    analysis.replayed = True
    storage.save("analyses", analysis)
    return analysis.id


def match_run(req: RunRequest, manifest: dict) -> dict | None:
    wanted = {(a.node_id, a.evaluator_id) for a in req.attachments}
    if req.isolated or req.patch_values:  # recordings are shared-process, unpatched-value runs: never substitute them
        return None
    for entry in manifest.get("runs", []):
        if sorted(req.patches or []) != sorted(entry.get("patches", [])):
            continue
        if entry["analysis_id"] != req.analysis_id or entry["variant"] != req.variant:
            continue
        if wanted <= {tuple(a) for a in entry["attachments"]}:
            return entry
    return None


def replay_run(req: RunRequest, entry: dict, report, demo: Path | None = None) -> str:
    demo = demo or demo_dir()
    run = RunResult.model_validate_json((demo / "runs" / f"{entry['run_id']}.json").read_text("utf-8"))
    _play(entry.get("log", []), RUN_REPLAY_S, report)
    wanted = {(a.node_id, a.evaluator_id) for a in req.attachments}
    run.results = [r for r in run.results if (r.node_id, r.evaluator_id) in wanted]
    run.samples = [s for s in run.samples if (s.node_id, s.evaluator_id) in wanted]
    run.bottlenecks = [b for b in run.bottlenecks if (b.node_id, b.evaluator_id) in wanted]
    scores = [r.value for r in run.results if r.unit == "score" and r.value is not None]
    run.overall = sum(scores) / len(scores) if scores else None
    run.id = new_id("r")  # each replay is its own run (a subset of the recorded attachments)
    run.replayed = True
    storage.save("runs", run)
    return run.id


# ---------------------------------------------------------------- recording


class _Recorder:
    def __init__(self):
        self.start = time.perf_counter()
        self.log: list[dict] = []

    def __call__(self, stage: str, progress: float, line: str | None = None) -> None:
        self.log.append({"t": round(time.perf_counter() - self.start, 3), "stage": stage,
                         "progress": max(0.0, min(1.0, progress)), "line": line})
        if line:
            print(f"  [{self.log[-1]['t']:6.1f}s] {line}")


def record(source: dict, variants: list[str], limit: int | None, out: Path, append: bool = False) -> dict:
    """append=True: keep the recorded analysis and runs in `out`, add runs for variants not recorded yet."""
    from .analysis import run_analysis
    from .core.schemas import Attachment
    from .runner.run import execute_run

    settings = get_settings()
    out.mkdir(parents=True, exist_ok=True)
    (out / "analyses").mkdir(exist_ok=True)
    (out / "runs").mkdir(exist_ok=True)
    existing = load_manifest(out) if append else None
    if existing:
        if existing["mock"] != settings.mock_llm:
            raise SystemExit("refusing to mix mock and real runs in one recording")
        manifest = existing
        analysis_id = manifest["analyses"][0]["analysis_id"]
        analysis = AnalysisResult.model_validate_json((out / "analyses" / f"{analysis_id}.json").read_text("utf-8"))
        storage.save("analyses", analysis)  # runs need the recorded analysis (same node ids)
        done = {r["variant"] for r in manifest["runs"]}
        variants = [v for v in variants if v not in done]
        print(f"appending {variants} to the recording of {analysis_id} (already recorded: {sorted(done)})")
    else:
        rec = _Recorder()
        print(f"analyzing {source}")
        analysis_id = run_analysis(source, rec)
        analysis = storage.load("analyses", analysis_id, AnalysisResult)
        (out / "analyses" / f"{analysis_id}.json").write_text(analysis.model_dump_json(indent=1), encoding="utf-8")
    manifest = manifest if existing else {
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "mock": settings.mock_llm,
        "models": {"analysis": settings.model, "judge": settings.judge_model},
        "analyses": [{"analysis_id": analysis_id, "name": analysis.source.name, "repo_url": analysis.source.repo_url,
                      "commit": analysis.source.commit, "duration_s": rec.log[-1]["t"] if rec.log else 0, "log": rec.log}],
        "runs": [],
    }
    attachments = [Attachment(node_id=r.node_id, evaluator_id=r.evaluator_id) for r in analysis.recommendations]
    for variant in variants:
        rec = _Recorder()
        print(f"running {variant} ({len(attachments)} attachments, limit {limit})")
        run_id = execute_run(RunRequest(analysis_id=analysis_id, attachments=attachments, variant=variant, limit=limit), rec)
        run = storage.load("runs", run_id, RunResult)
        (out / "runs" / f"{run_id}.json").write_text(run.model_dump_json(indent=1), encoding="utf-8")
        manifest["models"].update(run.models)
        manifest["runs"].append({"run_id": run_id, "analysis_id": analysis_id, "variant": variant, "limit": limit,
                                 "attachments": [[a.node_id, a.evaluator_id] for a in attachments],
                                 "duration_s": run.duration_s, "log": rec.log})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Record the demo for replay mode (EVALGRAPH_REPLAY=1).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("source", help="GitHub URL or local folder of the demo repo")
    r.add_argument("--variants", nargs="+", default=["baseline", "fix-retrieval", "fix-safety", "improved"])
    r.add_argument("--limit", type=int, default=None, help="samples per run (default: the whole dataset)")
    r.add_argument("--out", default=str(demo_dir()))
    r.add_argument("--append", action="store_true", help="add variants to an existing recording (same analysis)")
    args = ap.parse_args()
    src = {"repo_url": args.source} if args.source.startswith("http") else {"local_path": str(Path(args.source).resolve())}
    m = record(src, args.variants, args.limit, Path(args.out), append=args.append)
    kind = "MOCK (canned LLM output, pseudo judge scores): for rehearsal only, never present as real" if m["mock"] \
        else f"real DIAL ({m['models']})"
    print(f"\nrecorded {len(m['runs'])} run(s) to {args.out} · {kind}")


if __name__ == "__main__":
    main()
