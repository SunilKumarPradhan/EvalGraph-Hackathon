"""LLM gateway inside the sandbox (stdlib only).

The app under test talks to this gateway with a dummy key; the gateway forwards to DIAL with the real key and
records every call (tokens, latency, model) to /out/llm_calls.jsonl. It speaks:
  POST /v1/chat/completions                              OpenAI (also LangChain ChatOpenAI, Vercel AI SDK, LiteLLM)
  POST /openai/deployments/{d}/chat/completions          Azure OpenAI
  POST /v1/embeddings, /openai/deployments/{d}/embeddings
  POST /v1/messages                                      Anthropic (translated to/from chat completions)
  GET  /v1/models
A requested model that exists as a DIAL deployment is used as is (the app runs with the model it was written for);
any other name is mapped to the configured DIAL deployment.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOCK = threading.Lock()


class Config:
    def __init__(self, upstream: str, api_key: str, api_version: str, chat_model: str, embed_model: str, log_path: str,
                 mock: bool = False, deployments: set[str] | None = None):
        self.mock = mock
        self.deployments = deployments or set()  # DIAL deployment ids, when the app's own model may be honoured
        self.net_failures = 0                     # consecutive network failures reaching DIAL (circuit breaker)
        self.upstream = upstream.rstrip("/")
        self.api_key = api_key
        self.api_version = api_version
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.log_path = log_path


def _log(cfg: Config, record: dict) -> None:
    with LOCK, open(cfg.log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _mock(kind: str, body: dict) -> tuple[int, dict]:
    """Deterministic stand-in when no DIAL access: clearly marked, never mistaken for model output."""
    if kind == "embeddings":
        items = body.get("input", [])
        items = items if isinstance(items, list) else [items]
        vecs = [[((zlib.crc32(str(x).encode()) >> (i * 4)) % 97) / 97.0 for i in range(8)] for x in items]
        return 200, {"object": "list", "data": [{"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vecs)],
                     "usage": {"prompt_tokens": sum(len(str(x)) for x in items) // 4, "total_tokens": 0}}
    last = next((m.get("content") for m in reversed(body.get("messages", [])) if m.get("role") == "user"), "")
    text = f"[MOCK] answer to: {str(last)[:80]}"
    prompt = sum(len(str(m.get("content", ""))) for m in body.get("messages", [])) // 4
    return 200, {"id": "mock", "object": "chat.completion", "created": 0,
                 "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": prompt, "completion_tokens": len(text) // 4, "total_tokens": prompt + len(text) // 4}}


UPSTREAM_DOWN_AFTER = 3  # consecutive network failures before every further call fails fast


def upstream_down_flag(cfg: Config) -> str:
    return os.path.join(os.path.dirname(cfg.log_path) or ".", "upstream_down")


def _forward(cfg: Config, deployment: str, kind: str, body: dict) -> tuple[int, dict]:
    if cfg.mock:
        return _mock(kind, body)
    if cfg.net_failures >= UPSTREAM_DOWN_AFTER:  # DIAL is unreachable: don't make every call wait for its timeout
        return 503, {"error": {"message": "gateway: DIAL is unreachable from the sandbox (VPN off?); failing fast"}}
    url = f"{cfg.upstream}/openai/deployments/{deployment}/{kind}?api-version={cfg.api_version}"
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json", "api-key": cfg.api_key})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            cfg.net_failures = 0
            return resp.status, json.loads(resp.read().decode() or "{}")
    except urllib.error.HTTPError as exc:
        cfg.net_failures = 0  # DIAL answered (with an error): the network is fine
        try:
            return exc.code, json.loads(exc.read().decode() or "{}")
        except Exception:
            return exc.code, {"error": {"message": str(exc)}}
    except Exception as exc:  # network errors become 502s the app can see
        cfg.net_failures += 1
        if cfg.net_failures == UPSTREAM_DOWN_AFTER:
            with open(upstream_down_flag(cfg), "w", encoding="utf-8") as f:
                f.write(str(exc))
        return 502, {"error": {"message": f"gateway upstream error: {exc}"}}


def anthropic_to_chat(body: dict) -> dict:
    messages = []
    system = body.get("system")
    if isinstance(system, list):
        system = "".join(b.get("text", "") for b in system if isinstance(b, dict))
    if system:
        messages.append({"role": "system", "content": system})
    for m in body.get("messages", []):
        content = m.get("content")
        if isinstance(content, list):
            content = "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
        messages.append({"role": m.get("role", "user"), "content": content or ""})
    out = {"messages": messages, "max_tokens": body.get("max_tokens", 1024)}
    if "temperature" in body:
        out["temperature"] = body["temperature"]
    return out


def chat_to_anthropic(resp: dict, model: str) -> dict:
    choice = (resp.get("choices") or [{}])[0]
    usage = resp.get("usage") or {}
    return {"id": resp.get("id", "msg_gateway"), "type": "message", "role": "assistant", "model": model,
            "content": [{"type": "text", "text": (choice.get("message") or {}).get("content") or ""}],
            "stop_reason": "end_turn" if choice.get("finish_reason") != "length" else "max_tokens",
            "usage": {"input_tokens": usage.get("prompt_tokens", 0), "output_tokens": usage.get("completion_tokens", 0)}}


def make_handler(cfg: Config):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet
            pass

        def _send(self, code: int, payload: dict) -> None:
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.rstrip("/").endswith("/models"):
                return self._send(200, {"object": "list", "data": [{"id": cfg.chat_model, "object": "model"}]})
            return self._send(404, {"error": {"message": "not found"}})

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length).decode() or "{}")
            except json.JSONDecodeError:
                return self._send(400, {"error": {"message": "invalid JSON"}})
            path = self.path.split("?")[0]
            requested = body.get("model") or (re.search(r"/deployments/([^/]+)/", path) or [None, None])[1] or ""
            start = time.time()
            if body.get("stream"):
                body = {**body, "stream": False}  # streaming is collapsed; the app gets one final chunk-equivalent
            if path.endswith("/messages"):
                api, kind, upstream_body = "anthropic", "chat/completions", anthropic_to_chat(body)
            elif path.endswith("/embeddings"):
                api, kind, upstream_body = "embeddings", "embeddings", {k: v for k, v in body.items() if k != "model"}
            elif path.endswith("/chat/completions") or path.endswith("/completions"):
                api, kind, upstream_body = "openai", "chat/completions", {k: v for k, v in body.items() if k != "model"}
            else:
                return self._send(404, {"error": {"message": f"unsupported path {path}"}})
            deployment = cfg.embed_model if kind == "embeddings" else cfg.chat_model
            if requested in cfg.deployments:
                deployment = requested
            code, resp = _forward(cfg, deployment, kind, upstream_body)
            usage = resp.get("usage") or {}
            _log(cfg, {"ts": start, "end": time.time(), "api": api, "requested_model": requested, "deployment": deployment,
                       "status": code, "tokens_in": usage.get("prompt_tokens", 0),
                       "tokens_out": usage.get("completion_tokens", 0),
                       "preview": json.dumps(upstream_body.get("messages", upstream_body.get("input", "")), ensure_ascii=False)[-3000:]})
            if api == "anthropic" and code == 200:
                return self._send(200, chat_to_anthropic(resp, requested))
            if code == 200 and "model" not in resp:
                resp["model"] = requested
            return self._send(code, resp)

    return Handler


def list_deployments(cfg: Config) -> set[str]:
    req = urllib.request.Request(f"{cfg.upstream}/openai/models?api-version={cfg.api_version}", headers={"Api-Key": cfg.api_key})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return {m.get("id") for m in json.loads(resp.read().decode("utf-8")).get("data", []) if m.get("id")}
    except Exception as exc:  # no list: every request goes to the configured deployment
        print(f"gateway: could not list deployments ({exc})", file=sys.stderr, flush=True)
        return set()


def serve(cfg: Config, port: int) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(cfg))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main() -> None:
    """Started by the container entrypoint as root; reads the key from a root-only file."""
    secret_file = os.environ.get("EVALGRAPH_SECRET_FILE", "/run/evalgraph/secrets.json")
    with open(secret_file, encoding="utf-8") as f:
        s = json.load(f)
    cfg = Config(s["endpoint"], s["api_key"], s.get("api_version", "2024-02-01"), s["chat_model"],
                 s.get("embed_model") or s["chat_model"], os.environ.get("EVALGRAPH_LLM_LOG", "/out/llm_calls.jsonl"),
                 mock=bool(s.get("mock")))
    if s.get("honor_model") and not cfg.mock:
        cfg.deployments = list_deployments(cfg)
        print(f"gateway: {len(cfg.deployments)} DIAL deployments can be requested by name", file=sys.stderr, flush=True)
    port = int(os.environ.get("EVALGRAPH_GATEWAY_PORT", "8919"))
    serve(cfg, port)
    print(f"gateway listening on 127.0.0.1:{port} -> {cfg.upstream}", file=sys.stderr, flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
