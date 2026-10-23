"""JSON-file storage under data/{kind}/{id}.json."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from .settings import get_settings

T = TypeVar("T", bound=BaseModel)
_lock = threading.Lock()


class NotFound(KeyError):
    pass


def _dir(kind: str) -> Path:
    path = get_settings().resolved_data_dir() / kind
    path.mkdir(parents=True, exist_ok=True)
    return path


def save(kind: str, model: BaseModel) -> None:
    item_id = getattr(model, "id")
    path = _dir(kind) / f"{item_id}.json"
    tmp = path.with_suffix(".tmp")
    with _lock:
        tmp.write_text(model.model_dump_json(indent=1), encoding="utf-8")
        tmp.replace(path)


def load(kind: str, item_id: str, cls: type[T]) -> T:
    if not item_id.replace("_", "").isalnum():
        raise NotFound(item_id)
    path = _dir(kind) / f"{item_id}.json"
    if not path.exists():
        raise NotFound(item_id)
    return cls.model_validate_json(path.read_text(encoding="utf-8"))


def list_raw(kind: str) -> list[dict[str, Any]]:
    """All stored items as dicts, newest first."""
    items = []
    for path in sorted(_dir(kind).glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            items.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return items
