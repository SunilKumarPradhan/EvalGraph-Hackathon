"""Calls the app's chosen Python function once per sample, inside one process (state such as memory persists).

    python runner_fn.py job.json results.jsonl       (cwd = repo; PYTHONPATH includes the probe site dir)
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import os
import sys
import time
import traceback

SESSION_NAMES = {"session_id", "thread_id", "user_id", "conversation_id", "chat_id", "sid"}
LIST_NAMES = {"history", "chat_history", "messages", "turns"}


def resolve(module: str, qualname: str):
    obj = importlib.import_module(module)
    parts = qualname.split(".")
    for i, part in enumerate(parts):
        nxt = getattr(obj, part)
        if inspect.isclass(nxt) and i < len(parts) - 1:
            nxt = nxt()  # methods need an instance; only zero-arg constructors are supported
        obj = nxt
    return obj


def build_kwargs(fn, question_param: str, question: str, sample_id: str, fixed: dict) -> dict:
    kwargs = {}
    for name, p in inspect.signature(fn).parameters.items():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) or name in ("self", "cls"):
            continue
        if name == question_param:
            kwargs[name] = question
        elif name in fixed:
            kwargs[name] = fixed[name]
        elif p.default is not inspect.Parameter.empty:
            continue
        elif name in SESSION_NAMES:
            kwargs[name] = f"evalgraph-{sample_id}"
        elif name in LIST_NAMES:
            kwargs[name] = []
        else:
            kwargs[name] = None
    return kwargs


def to_text(result) -> str:
    if inspect.isgenerator(result):
        result = "".join(str(x) for x in result)
    for attr in ("content", "text", "answer", "output"):
        if hasattr(result, attr) and isinstance(getattr(result, attr), str):
            return getattr(result, attr)
    if isinstance(result, dict):
        for key in ("answer", "output", "result", "text", "content", "response"):
            if isinstance(result.get(key), str):
                return result[key]
    return result if isinstance(result, str) else json.dumps(result, default=str)


def main() -> None:
    with open(sys.argv[1], encoding="utf-8") as f:
        job = json.load(f)
    plan = job["plan"]
    import probe  # provided by the harness site dir

    fn = resolve(plan["module"], plan["function"])
    down_flag = os.path.join(os.path.dirname(os.path.abspath(sys.argv[2])), "upstream_down")
    with open(sys.argv[2], "a", encoding="utf-8") as out:
        for s in job["samples"]:
            probe.set_sample(s["id"])
            start, answer, error = time.time(), "", None
            if os.path.exists(down_flag):  # the remaining samples would only fail slowly
                out.write(json.dumps({"sample_id": s["id"], "start": start, "end": time.time(), "answer": "",
                                      "error": "LLMUnavailable: DIAL is unreachable from the sandbox (VPN off?)"}) + "\n")
                out.flush()
                continue
            try:
                result = fn(**build_kwargs(fn, plan.get("question_param", "question"), s["question"], s["id"],
                                           plan.get("fixed_args", {})))
                if inspect.isawaitable(result):
                    result = asyncio.run(result)
                answer = to_text(result)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
            out.write(json.dumps({"sample_id": s["id"], "start": start, "end": time.time(), "answer": answer[:4000],
                                  "error": error}) + "\n")
            out.flush()
    probe.set_sample(None)


if __name__ == "__main__":
    main()
