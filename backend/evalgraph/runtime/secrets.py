"""User-provided app secrets per analysis (conf_05 Q-50): stored server-side, never returned by the API, only passed into
the Docker sandbox by environment-variable name. The DIAL key is never one of them."""

from __future__ import annotations

import json
import re

from ..settings import get_settings

NAME = re.compile(r"^[A-Z][A-Z0-9_]{1,63}$")
FORBIDDEN = ("DIAL_", "EVALGRAPH", "APPSECRET_", "PATH", "HOME", "PYTHON")


def _path(analysis_id: str):
    if not analysis_id.replace("_", "").isalnum():
        raise ValueError("bad analysis id")
    d = get_settings().resolved_data_dir() / "secrets"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{analysis_id}.json"


def load_secrets(analysis_id: str) -> dict[str, str]:
    p = _path(analysis_id)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_secrets(analysis_id: str, changes: dict[str, str | None]) -> list[str]:
    """Merge changes (None deletes a name). Returns the stored names."""
    for name in changes:
        if not NAME.match(name) or name.startswith(FORBIDDEN):
            raise ValueError(f"invalid secret name: {name}")
    current = load_secrets(analysis_id)
    for name, value in changes.items():
        if value is None or value == "":
            current.pop(name, None)
        else:
            current[name] = value
    _path(analysis_id).write_text(json.dumps(current), encoding="utf-8")
    return sorted(current)
