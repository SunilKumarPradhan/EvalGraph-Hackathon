"""Settings from backend/.env and the process environment."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_ROOT.parent
FIXTURES_DIR = PROJECT_ROOT / "fixtures"

# Load .env into os.environ as well, so the in-process CUDA Archer adapter sees DIAL_* / ARCHER_* too.
load_dotenv(BACKEND_ROOT / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_ROOT / ".env", extra="ignore", populate_by_name=True)

    mock_llm: bool = Field(True, validation_alias="EVALGRAPH_MOCK_LLM")
    replay: bool = Field(False, validation_alias="EVALGRAPH_REPLAY")
    data_dir: Path = Field(Path("../data"), validation_alias="EVALGRAPH_DATA_DIR")
    max_concurrency: int = Field(4, validation_alias="EVALGRAPH_MAX_CONCURRENCY")
    # PHASE-11: bounded investigator (LLM explores the AI map with read-only tools); off = batch review only
    investigator: bool = Field(True, validation_alias="EVALGRAPH_INVESTIGATOR")
    investigator_turns: int = Field(6, validation_alias="EVALGRAPH_INVESTIGATOR_TURNS")

    dial_api_key: str | None = Field(None, validation_alias=AliasChoices("DIAL_API_KEY", "LLM_API_KEY"))
    dial_endpoint: str = Field("https://ai-proxy.lab.epam.com",
                               validation_alias=AliasChoices("DIAL_ENDPOINT", "AZURE_ENDPOINT"))
    dial_api_version: str = Field("2024-02-01", validation_alias="DIAL_API_VERSION")
    # DIAL deployments (checked against the DIAL deployment list, 2026-10-03)
    model: str = Field("gpt-4.1-2025-04-14", validation_alias="EVALGRAPH_MODEL")            # analysis
    judge_model: str = Field("claude-haiku-4-5@20251001", validation_alias="EVALGRAPH_JUDGE_MODEL")  # other family than the apps
    sandbox_model: str = Field("gpt-4.1-mini-2025-04-14", validation_alias="EVALGRAPH_SANDBOX_MODEL")  # what sandboxed apps get
    sandbox_embed_model: str = Field("text-embedding-3-small-1", validation_alias="EVALGRAPH_SANDBOX_EMBED_MODEL")
    sandbox_cache: bool = Field(True, validation_alias="EVALGRAPH_SANDBOX_CACHE")  # per-repo download cache volume
    # an app asking for a model DIAL has (e.g. gpt-4o) gets it; otherwise EVALGRAPH_SANDBOX_MODEL (conf_05 Q-46)
    sandbox_honor_model: bool = Field(True, validation_alias="EVALGRAPH_SANDBOX_HONOR_MODEL")
    # "allowlist": the container reaches only package registries, GitHub, Hugging Face and DIAL (conf_05 Q-47)
    sandbox_egress: str = Field("allowlist", validation_alias="EVALGRAPH_SANDBOX_EGRESS")
    egress_allow: str = Field("", validation_alias="EVALGRAPH_EGRESS_ALLOW")  # extra hosts, comma-separated (".x.com" = subdomains)

    def resolved_data_dir(self) -> Path:
        path = self.data_dir if self.data_dir.is_absolute() else (BACKEND_ROOT / self.data_dir)
        return path.resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()
