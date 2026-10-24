"""SandboxAdapter: the generic adapter for any analyzed repo (PHASE-06B), backed by the Docker sandbox.

Same role as the hand-written CUDA Archer adapter (CONTRACTS §8), but driven entirely by the analysis:
run plan, probe manifest and dataset come from AnalysisResult.runtime. Runs the whole dataset in one container
(`run_batch`) because starting a container per sample would be far too slow.
"""

from __future__ import annotations

from ..analyzer.clone import split_tree_url
from ..core.schemas import AnalysisResult, AppOutput, GoldenSample
from ..settings import get_settings
from .plan import RunPlan
from .sandbox import DockerSandbox, SandboxError, cleanup, to_outputs


class SandboxAdapter:
    name = "sandbox"
    dataset_name = "synthetic"
    secret_canary = None

    def __init__(self, analysis: AnalysisResult, sandbox=None):
        if analysis.runtime is None or analysis.runtime.plan.get("mode") in (None, "unsupported"):
            raise SandboxError("this analysis has no runnable plan")
        self.analysis = analysis
        self.sandbox = sandbox or DockerSandbox()

    @property
    def available(self) -> bool:
        return getattr(self.sandbox, "available", lambda: True)()

    def variants(self) -> list[str]:
        return ["as-is"]

    def models(self) -> dict[str, str]:
        # the in-sandbox gateway maps every model the app asks for to these DIAL deployments (conf_05 Q-46)
        s = get_settings()
        app = (f"{s.sandbox_model} (or the app's own model when DIAL has it)" if s.sandbox_honor_model else s.sandbox_model)
        return {"app (via sandbox gateway)": app, "embeddings (via sandbox gateway)": s.sandbox_embed_model}

    def load_dataset(self) -> list[GoldenSample]:
        return list(self.analysis.runtime.dataset)

    def source(self) -> dict:
        src = self.analysis.source
        if not src.repo_url:
            return {"local_path": src.local_path}
        base, ref, sub = split_tree_url(src.repo_url)  # .../tree/<branch>/<folder> analyses run that folder
        return {"repo_url": base, **({"ref": ref} if ref else {}), **({"subdir": sub} if sub else {})}

    def run_batch(self, samples: list[GoldenSample], variant: str = "as-is", patches: list[dict] | None = None,
                  isolated: bool = False) -> list[AppOutput]:
        plan = RunPlan(**self.analysis.runtime.plan)
        kwargs: dict = {"patches": patches} if patches else {}
        if isolated:
            kwargs["isolated"] = True
        from .secrets import load_secrets

        if secrets := load_secrets(self.analysis.id):
            kwargs["secrets"] = secrets
        raw = self.sandbox.run(self.source(), plan, samples, self.analysis.runtime.probes, **kwargs)
        self.last_egress_blocked = sorted({e["host"] for e in raw.egress if not e.get("allowed")})
        try:
            if not raw.status.get("ok"):
                raise SandboxError(f"sandbox run failed at '{raw.status.get('stage')}': {raw.status.get('error')}")
            return to_outputs(raw, samples, self.analysis.graph.nodes)
        finally:
            cleanup(raw)

    def run(self, sample: GoldenSample, variant: str = "as-is") -> AppOutput:
        return self.run_batch([sample], variant)[0]
