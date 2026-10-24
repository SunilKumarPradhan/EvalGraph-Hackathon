"""Graph-guided probe (stdlib only), activated by sitecustomize inside the app process.

Records a span for every call to a function listed in the probe manifest (file suffix + qualname, taken from
EvalGraph's analysis), using sys.setprofile / threading.setprofile. No source changes to the app.
Span records go to $EVALGRAPH_SPANS (JSONL) tagged with the current sample id.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time

_state = {"sample": None, "lock": threading.Lock(), "targets": {}, "out": None, "depth": threading.local()}
MAX_REPR = 600


def _short(value) -> str:
    try:
        text = value if isinstance(value, str) else repr(value)
    except Exception:
        text = f"<{type(value).__name__}>"
    return text if len(text) <= MAX_REPR else text[:MAX_REPR] + "…"


MAX_ITEMS = 20
MAX_ITEM = 1200


def _item_text(item) -> str | None:
    """Text of one returned chunk/document: a str, or an object/dict with a common text field."""
    if isinstance(item, str):
        return item
    if isinstance(item, (tuple, list)) and item and isinstance(item[0], str):
        return item[0]  # (text, score) pairs
    for attr in ("page_content", "text", "content", "document"):
        value = item.get(attr) if isinstance(item, dict) else getattr(item, attr, None)
        if isinstance(value, str):
            return value
    return None


def _items(value) -> list[str] | None:
    """Returned lists of chunks/documents are kept item by item, so retrieval judges see real passages."""
    if not isinstance(value, (list, tuple)) or not value or len(value) > 200:
        return None
    try:
        texts = [_item_text(v) for v in value[:MAX_ITEMS]]
    except Exception:
        return None
    if any(t is None for t in texts):
        return None
    return [t if len(t) <= MAX_ITEM else t[:MAX_ITEM] + "…" for t in texts]


def set_sample(sample_id: str | None) -> None:
    _state["sample"] = sample_id


def _emit(record: dict) -> None:
    with _state["lock"], open(_state["out"], "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _match(code) -> dict | None:
    qual = getattr(code, "co_qualname", code.co_name)
    for (suffix, target_qual), meta in _state["targets"].items():
        if qual == target_qual and code.co_filename.replace("\\", "/").endswith(suffix):
            return meta
    return None


_open: dict[int, dict] = {}


def _profile(frame, event, arg):
    if event not in ("call", "return"):
        return
    code = frame.f_code
    meta = _state["targets_by_code"].get(code)
    if meta is None:
        if code in _state["misses"]:
            return
        meta = _match(code)
        if meta is None:
            _state["misses"].add(code)
            return
        _state["targets_by_code"][code] = meta
    key = id(frame)
    if event == "call":
        args = {}
        for name in code.co_varnames[: code.co_argcount + code.co_kwonlyargcount]:
            if name in frame.f_locals and name not in ("self", "cls"):
                args[name] = _short(frame.f_locals[name])
        _open[key] = {"start": time.time(), "args": args}
    elif event == "return" and key in _open:
        info = _open.pop(key)
        _emit({"sample_id": _state["sample"], "node_id": meta["node_id"], "component": meta["qualname"].split(".")[-1],
               "file": meta["file"], "kind": meta["kind"], "start": info["start"], "end": time.time(),
               "input": info["args"], "output": _short(arg), "items": _items(arg), "thread": threading.get_ident()})


def install(manifest_path: str, spans_path: str) -> None:
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    _state["targets"] = {(m["file"], m["qualname"]): m for m in manifest}
    _state["targets_by_code"] = {}
    _state["misses"] = set()
    _state["out"] = spans_path
    sys.setprofile(_profile)
    threading.setprofile(_profile)


def install_from_env() -> None:
    manifest = os.environ.get("EVALGRAPH_PROBES")
    spans = os.environ.get("EVALGRAPH_SPANS")
    if manifest and spans and os.path.exists(manifest):
        install(manifest, spans)
        set_sample(os.environ.get("EVALGRAPH_SAMPLE_ID"))
