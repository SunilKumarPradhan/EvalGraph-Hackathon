"""Run any repo through the Docker sandbox from the command line (needs a running Docker daemon).

    uv run python -m evalgraph.runtime.cli https://github.com/owner/repo [--samples 4]
    uv run python -m evalgraph.runtime.cli ../../cuda-archer --samples 3

Analyzes the repo, prints the inferred run plan and dataset, runs it inside the sandbox (clone + install + run in
the container) and prints what each graph node did per sample.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict

from .. import storage
from ..analysis import run_analysis
from ..core.schemas import AnalysisResult
from .adapter import SandboxAdapter
from .sandbox import DockerSandbox


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="GitHub URL or local folder")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--force-sandbox", action="store_true", help="use the sandbox even when a dedicated adapter exists")
    args = ap.parse_args()
    src = {"repo_url": args.source} if args.source.startswith("http") else {"local_path": args.source}
    aid = run_analysis(src, lambda stage, p, line=None: line and print(f"  {line}"))
    a = storage.load("analyses", aid, AnalysisResult)
    plan = a.runtime.plan if a.runtime else {}
    print(f"\nrun plan: {plan.get('mode')} · {plan.get('reason')}")
    if plan.get("mode") in (None, "unsupported") or (a.summary.adapter != "sandbox" and not args.force_sandbox):
        sys.exit("not runnable in the sandbox (" + (plan.get("reason") or "no plan") + ")")
    if not DockerSandbox.available():
        sys.exit("Docker daemon is not running. Start Docker and retry.")
    adapter = SandboxAdapter(a)
    samples = adapter.load_dataset()[: args.samples]
    print(f"running {len(samples)} samples in the sandbox (first run builds the image)...")
    outs = adapter.run_batch(samples)
    for s, o in zip(samples, outs):
        print(f"\n[{s.id}] {s.question}\n  answer: {o.answer[:200]}")
        for sp in o.spans:
            if sp.error:
                print(f"  error in {sp.component}: {sp.error[:300]}")
        per_node = defaultdict(lambda: [0, 0.0, 0])
        for sp in o.spans:
            row = per_node[sp.component]
            row[0] += 1
            row[1] += sp.latency_ms
            row[2] += sp.tokens_in + sp.tokens_out
        for comp, (n, ms, tok) in per_node.items():
            print(f"    {comp:28} x{n}  {ms:8.0f} ms  {tok:6d} tokens")


if __name__ == "__main__":
    main()
