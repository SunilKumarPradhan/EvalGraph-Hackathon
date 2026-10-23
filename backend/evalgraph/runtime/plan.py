"""How to call an analyzed app inside the sandbox (static inference from the analysis).

Preference: call the outermost function on a request path that takes a question-like parameter, in-process
(precise spans, shared memory across samples). Otherwise start the web server and POST to its route.
"""

from __future__ import annotations

import re
from collections import deque
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..understanding.buffer import RepoBuffer
from ..understanding.components import Component, SymbolGraph
from ..understanding.signals import SymbolInfo

# strong names say "the user's question"; weak ones are generic and also used by helpers (a splitter's `text`)
STRONG_PARAMS = ["question", "query", "prompt", "message", "user_input", "user_message", "user_question", "user_query",
                 "user_prompt", "question_text", "query_text", "task", "instruction", "request"]
WEAK_PARAMS = ["input", "q", "text", "msg", "content", "input_text", "utterance"]
QUESTION_PARAMS = STRONG_PARAMS + WEAK_PARAMS
NOT_ENTRY_TYPES = {"dataset", "embedding", "vector_db", "prompt", "parser"}  # helpers, not the request path
JS_BODY_KEYS = ["question", "prompt", "message", "query", "input", "messages"]


class RunPlan(BaseModel):
    mode: Literal["python_function", "http", "unsupported"]
    language: str = "python"
    module: str | None = None
    function: str | None = None
    question_param: str | None = None
    fixed_args: dict[str, Any] = Field(default_factory=dict)
    start_cmd: str | None = None
    port: int = 8765
    path: str | None = None
    method: str = "POST"
    body: dict[str, Any] | None = None
    response_path: str | None = None
    install: str | list = "auto"
    inferred_requirements: list[str] = Field(default_factory=list)  # pip packages guessed from imports (no manifest)
    llm_env: dict[str, str] = Field(default_factory=dict)  # app's own LLM env var names -> role, filled by the harness
    app_env: list[str] = Field(default_factory=list)       # other env vars the app reads (search keys, DB URLs...)
    reason: str = ""
    entry_file: str | None = None


def _module_of(path: str) -> str | None:
    if not path.endswith(".py"):
        return None
    parts = path[:-3].split("/")
    if parts[0] == "src" and len(parts) > 1:
        parts = parts[1:]
    if parts[-1] == "__init__":
        parts = parts[:-1]
    if not all(p.isidentifier() for p in parts):
        return None
    return ".".join(parts)


def _distance(g: SymbolGraph, starts: list[str]) -> dict[str, int]:
    dist = {s: 0 for s in starts}
    queue = deque(starts)
    while queue:
        k = queue.popleft()
        for n in g.out.get(k, ()):
            if n not in dist:
                dist[n] = dist[k] + 1
                queue.append(n)
    return dist


def infer_python(symbols: dict[str, SymbolInfo], comps: list[Component], g: SymbolGraph, scope: set[str],
                 entries: list[str]) -> RunPlan | None:
    comp_keys = {c.key for c in comps if c.type != "entrypoint"}
    helper_keys = {c.key for c in comps if c.type in NOT_ENTRY_TYPES} - {c.key for c in comps if c.type not in NOT_ENTRY_TYPES}
    reaches_ai = {k for k in scope if g.reach(k) & comp_keys}
    dist = _distance(g, entries) if entries else {}
    candidates = []
    for key in scope:
        info = symbols.get(key)
        if info is None or info.d.name.startswith("_") or info.d.name == "<module>" or key not in reaches_ai:
            continue
        if key in helper_keys:  # an ingestion/embedding helper is never the request entry point
            continue
        param = next((p for p in QUESTION_PARAMS if p in info.d.args), None)
        if not param:
            continue
        module = _module_of(info.path)
        if not module:
            continue
        qual = info.d.qualname
        depth = qual.count(".")
        is_route = any(re.search(r"\.(get|post|put|route|websocket)$", d) for d in info.d.decorators)
        weak = param not in STRONG_PARAMS
        # a question-named input first, then outermost (closest to the user), plain functions before methods,
        # non-route before route handlers
        candidates.append((weak, dist.get(key, 99), depth, is_route, -len(g.reach(key) & comp_keys),
                           key, module, qual, param))
    if not candidates:
        return None
    candidates.sort()
    *_, key, module, qual, param = candidates[0]
    return RunPlan(mode="python_function", module=module, function=qual, question_param=param, entry_file=key.split("::")[0],
                   reason=f"calls {module}.{qual}({param}=...) once per sample; it reaches the AI pipeline")


def infer_http(buffer: RepoBuffer, symbols: dict[str, SymbolInfo], entries: list[str]) -> RunPlan | None:
    for key in entries:
        info = symbols.get(key)
        if info is None:
            continue
        path = info.path
        body_key = next((k for k in JS_BODY_KEYS if k in info.d.names_used), "question")
        body = {"messages": [{"role": "user", "content": "{question}"}]} if body_key == "messages" else {body_key: "{question}"}
        m = re.search(r"(^|/)app/(.*)/route\.(ts|js)$", path)
        if m:  # Next.js app router
            route = "/" + re.sub(r"\([^)]*\)/?", "", m.group(2)).strip("/")
            return RunPlan(mode="http", language="typescript", start_cmd="npm run dev -- -p $PORT", path=route,
                           body=body, entry_file=path, reason=f"Next.js route {route}")
        fact = next((s.fact for s in info.signals if s.fact.startswith("web route @")), "")
        r = re.search(r"web route @\w+\.(post|get) (/\S*)", fact)
        if r and path.endswith((".js", ".ts", ".mjs")):
            runner = "node" if path.endswith((".js", ".mjs")) else "npx tsx"
            return RunPlan(mode="http", language="javascript" if runner == "node" else "typescript",
                           start_cmd=f"{runner} {path}", path=r.group(2), method=r.group(1).upper(), body=body,
                           entry_file=path, reason=f"Express route {r.group(1).upper()} {r.group(2)}")
    return None


PIP_NAMES = {
    "yaml": "pyyaml", "dotenv": "python-dotenv", "bs4": "beautifulsoup4", "sklearn": "scikit-learn",
    "cv2": "opencv-python-headless", "PIL": "pillow", "faiss": "faiss-cpu", "docx": "python-docx", "fitz": "pymupdf",
    "llama_index": "llama-index", "google": "google-generativeai", "jwt": "pyjwt", "dateutil": "python-dateutil",
    "magic": "python-magic", "multipart": "python-multipart", "Crypto": "pycryptodome", "telegram": "python-telegram-bot",
}
NOT_ON_PYPI = {"__future__", "typing_extensions"}


def infer_requirements(buffer: RepoBuffer, symbols: dict[str, SymbolInfo]) -> list[str]:
    """Third-party top-level imports of the Python code, as pip package names."""
    import sys

    from ..understanding.pipeline import parse

    local = {p.split("/")[0].removesuffix(".py") for p in buffer.files} | {
        p.rsplit("/", 1)[-1].removesuffix(".py") for p in buffer.files if p.endswith(".py")}
    mods: set[str] = set()
    for f in parse(buffer):
        if not f.path.endswith((".py", ".ipynb")):
            continue
        for imp in f.imports:
            top = imp.module.split(".")[0]
            if top and not imp.module.startswith(".") and top not in sys.stdlib_module_names and top not in local \
                    and top not in NOT_ON_PYPI:
                mods.add(top)
    return sorted({PIP_NAMES.get(m, m.replace("_", "-")) for m in mods})


# Providers whose API the in-sandbox gateway speaks (OpenAI / Azure OpenAI / Anthropic wire formats).
LLM_ENV_HINTS = ("OPENAI", "AZURE", "DIAL", "LLM", "ANTHROPIC", "CLAUDE", "GPT", "AI_PROXY", "OPENROUTER", "DEEPSEEK", "GROQ")
QUOTED_NAME = re.compile(r"""["']([A-Z][A-Z0-9_]{3,})["']""")
DOTENV_NAME = re.compile(r"^\s*(?:export\s+)?([A-Z][A-Z0-9_]{3,})\s*=", re.MULTILINE)

CODE_SUFFIXES = (".py", ".ipynb", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".toml", ".yaml", ".yml", ".json")


ENV_READ = re.compile(r"""(?:getenv|environ(?:\.get)?|env\.get)\(\s*["']([A-Z][A-Z0-9_]{2,})["']"""
                      r"""|environ\[\s*["']([A-Z][A-Z0-9_]{2,})["']\s*\]|process\.env\.([A-Z][A-Z0-9_]{2,})"""
                      r"""|process\.env\[\s*["']([A-Z][A-Z0-9_]{2,})["']\s*\]""")
NOT_SECRET = {"PATH", "HOME", "PORT", "HOST", "DEBUG", "LOG_LEVEL", "LOGLEVEL", "NODE_ENV", "ENV", "ENVIRONMENT", "TZ",
              "LANG", "USER", "PWD", "TMPDIR", "TEMP", "CI"}


def infer_app_env(buffer: RepoBuffer, llm_env: dict[str, str]) -> list[str]:
    """Environment variables the app reads that are not LLM settings: candidates for user-provided app secrets."""
    names: set[str] = set()
    for path, text in buffer.files.items():
        if ".env" in path.rsplit("/", 1)[-1]:
            names.update(DOTENV_NAME.findall(text))
        elif path.endswith(CODE_SUFFIXES):
            names.update(next(g for g in m if g) for m in ENV_READ.findall(text))
    return sorted(n for n in names if n not in llm_env and n not in NOT_SECRET and not n.startswith(("PYTHON", "EVALGRAPH_"))
                  and not any(h in n for h in LLM_ENV_HINTS))


def infer_llm_env(buffer: RepoBuffer) -> dict[str, str]:
    """App-specific LLM env vars (e.g. MYAPP_LLM_KEY, DIAL_ENDPOINT) -> role (key | base | endpoint | version).

    The sandbox never passes real provider settings to the app, so an app that reads its own variable names would see
    nothing. The harness fills each role: dummy key, gateway URL (with /v1 for base URLs), default API version.
    """
    roles: dict[str, str] = {}
    for path, text in buffer.files.items():
        if ".env" in path.rsplit("/", 1)[-1]:
            names = DOTENV_NAME.findall(text)
        elif path.endswith(CODE_SUFFIXES):
            names = QUOTED_NAME.findall(text)
        else:
            continue
        for name in names:
            if not any(h in name for h in LLM_ENV_HINTS):
                continue
            if name.endswith(("API_KEY", "_KEY", "_TOKEN")):
                roles[name] = "key"
            elif name.endswith(("BASE_URL", "API_BASE")):
                roles[name] = "base"
            elif name.endswith(("ENDPOINT", "_URL", "_HOST")):
                roles[name] = "endpoint"
            elif name.endswith("API_VERSION"):
                roles[name] = "version"
    return dict(sorted(roles.items()))


def infer_plan(buffer: RepoBuffer, symbols: dict[str, SymbolInfo], comps: list[Component], g: SymbolGraph,
               scope: set[str], entries: list[str]) -> RunPlan:
    has_python = any(p.endswith(".py") for p in buffer.files)
    plan = infer_python(symbols, comps, g, scope, entries) if has_python else None
    if plan is None:
        plan = infer_http(buffer, symbols, entries)
    if plan is not None and plan.mode == "python_function":
        has_manifest = any(p in buffer.files for p in ("requirements.txt", "pyproject.toml", "setup.py"))
        if not has_manifest:
            plan.inferred_requirements = infer_requirements(buffer, symbols)
    if plan is None:
        only_nb = all(p.endswith(".ipynb") for p in buffer.code_files())
        return RunPlan(mode="unsupported", reason="notebook-only repos are not runnable yet" if only_nb
                       else "no callable function or web route with a question-like input was found")
    plan.llm_env = infer_llm_env(buffer)
    plan.app_env = infer_app_env(buffer, plan.llm_env)
    return plan
