"""Runtime adapters: the ONLY way EvalGraph executes application code.

An adapter wraps a locally installed package. The cloned copy of an analyzed repo is never
imported or executed. See docs/02-CONTRACTS.md §8.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Protocol

from ..core.schemas import AppOutput, GoldenSample, Span
from ..settings import get_settings

logger = logging.getLogger(__name__)


class Adapter(Protocol):
    name: str
    dataset_name: str

    @property
    def available(self) -> bool: ...
    @property
    def secret_canary(self) -> str | None: ...
    def variants(self) -> list[str]: ...
    def load_dataset(self) -> list[GoldenSample]: ...
    def run(self, sample: GoldenSample, variant: str) -> AppOutput: ...


class CudaArcherAdapter:
    """Runs the installed ``cuda_archer`` package (sibling repo ../cuda-archer, editable install)."""

    name = "cuda_archer"
    dataset_name = "cuda_archer_golden"

    def _import(self):
        # Keep the app's LLM mode aligned with ours. Must happen before cuda_archer's cached settings load.
        if get_settings().mock_llm:
            os.environ["ARCHER_MOCK_LLM"] = "1"
        import cuda_archer.app
        import cuda_archer.config
        import cuda_archer.llm.prompts
        import cuda_archer.settings

        return cuda_archer

    @property
    def available(self) -> bool:
        try:
            self._import()
            return True
        except ImportError as exc:
            logger.warning("cuda_archer adapter unavailable (clone ../cuda-archer and uv sync): %s", exc)
            return False

    @property
    def secret_canary(self) -> str | None:
        return self._import().llm.prompts.CANARY

    def variants(self) -> list[str]:
        return self._import().config.list_variants()

    def models(self) -> dict[str, str]:
        s = self._import().settings.get_settings()
        return {"app_large": s.model_large, "app_small": s.model_small}

    def repo_root(self) -> Path:
        return self._import().settings.REPO_ROOT

    def load_dataset(self) -> list[GoldenSample]:
        path = self.repo_root() / "eval" / "golden.jsonl"
        return [GoldenSample.model_validate(json.loads(line))
                for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def run(self, sample: GoldenSample, variant: str) -> AppOutput:
        archer = self._import()
        out = archer.app.run(sample.question, variant=variant, history=sample.history or None)
        return AppOutput(answer=out.answer, route=out.route, blocked=out.blocked, nodes=out.nodes,
                         spans=[Span.model_validate(s.model_dump()) for s in out.spans])


ADAPTERS: dict[str, Adapter] = {"cuda_archer": CudaArcherAdapter()}


def get_adapter(name: str | None) -> Adapter | None:
    """Registered and importable adapter, or None."""
    if not name or name not in ADAPTERS:
        return None
    adapter = ADAPTERS[name]
    return adapter if adapter.available else None
