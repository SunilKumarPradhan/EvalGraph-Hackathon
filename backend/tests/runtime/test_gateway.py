"""The in-sandbox LLM gateway, against a fake DIAL upstream: real key only upstream, protocols translated, calls logged."""

import json
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from openai import OpenAI

from evalgraph.runtime.sandbox import HARNESS_DIR

sys.path.insert(0, str(HARNESS_DIR))
import gateway


class FakeDial(BaseHTTPRequestHandler):
    seen: list = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode())
        FakeDial.seen.append({"path": self.path, "key": self.headers.get("api-key"), "body": body})
        if "/embeddings" in self.path:
            resp = {"data": [{"embedding": [0.1, 0.2], "index": 0}], "usage": {"prompt_tokens": 3, "total_tokens": 3}}
        else:
            resp = {"id": "x", "object": "chat.completion", "created": 0, "model": "dial",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello from dial"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15}}
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def gw(tmp_path):
    FakeDial.seen = []
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), FakeDial)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    log = tmp_path / "calls.jsonl"
    cfg = gateway.Config(f"http://127.0.0.1:{upstream.server_address[1]}", "REAL-DIAL-KEY", "2024-02-01",
                         "gpt-4o-mini-2024-07-18", "text-embedding-3-small-1", str(log))
    server = gateway.serve(cfg, 0)
    yield f"http://127.0.0.1:{server.server_address[1]}", log
    server.shutdown()
    upstream.shutdown()


def test_openai_sdk_through_gateway(gw):
    url, log = gw
    client = OpenAI(api_key="sk-dummy", base_url=f"{url}/v1")
    r = client.chat.completions.create(model="gpt-4-turbo", messages=[{"role": "user", "content": "hi"}], temperature=0.2)
    assert r.choices[0].message.content == "hello from dial"
    up = FakeDial.seen[-1]
    assert up["key"] == "REAL-DIAL-KEY"                                     # real key only on the upstream hop
    assert up["path"].startswith("/openai/deployments/gpt-4o-mini-2024-07-18/chat/completions")  # model mapped
    assert "model" not in up["body"] and up["body"]["temperature"] == 0.2
    call = json.loads(log.read_text().splitlines()[-1])
    assert call["requested_model"] == "gpt-4-turbo" and call["tokens_in"] == 11 and call["tokens_out"] == 4


def test_anthropic_messages_translated(gw):
    url, _ = gw
    body = {"model": "claude-sonnet-4-5", "max_tokens": 50, "system": "be brief",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]}
    req = urllib.request.Request(f"{url}/v1/messages", data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "x-api-key": "dummy"})
    resp = json.loads(urllib.request.urlopen(req).read())
    assert resp["type"] == "message" and resp["content"][0]["text"] == "hello from dial"
    assert resp["usage"] == {"input_tokens": 11, "output_tokens": 4}
    sent = FakeDial.seen[-1]["body"]["messages"]
    assert sent[0] == {"role": "system", "content": "be brief"} and sent[1]["content"] == "hello"


def test_embeddings_and_azure_paths(gw):
    url, _ = gw
    client = OpenAI(api_key="sk-dummy", base_url=f"{url}/v1")
    assert client.embeddings.create(model="text-embedding-ada-002", input="abc").data[0].embedding == [0.1, 0.2]
    assert "/deployments/text-embedding-3-small-1/embeddings" in FakeDial.seen[-1]["path"]
    req = urllib.request.Request(f"{url}/openai/deployments/whatever/chat/completions?api-version=x",
                                 data=json.dumps({"messages": [{"role": "user", "content": "x"}]}).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    assert json.loads(urllib.request.urlopen(req).read())["choices"][0]["message"]["content"] == "hello from dial"


def test_mock_mode_never_calls_upstream(tmp_path):
    cfg = gateway.Config("http://127.0.0.1:9", "k", "v", "m", "e", str(tmp_path / "c.jsonl"), mock=True)
    server = gateway.serve(cfg, 0)
    try:
        r = OpenAI(api_key="x", base_url=f"http://127.0.0.1:{server.server_address[1]}/v1").chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "ping"}])
        assert r.choices[0].message.content.startswith("[MOCK]")
    finally:
        server.shutdown()


def test_app_model_honoured_when_dial_has_it(gw):
    url, log = gw
    client = OpenAI(api_key="sk-dummy", base_url=f"{url}/v1")
    # the gateway in this fixture knows no deployments yet: every name maps to the configured model
    client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
    assert FakeDial.seen[-1]["path"].startswith("/openai/deployments/gpt-4o-mini-2024-07-18/")


def test_requested_deployment_is_used(tmp_path):
    FakeDial.seen = []
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), FakeDial)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    cfg = gateway.Config(f"http://127.0.0.1:{upstream.server_address[1]}", "K", "2024-02-01", "gpt-4.1-mini",
                         "emb", str(tmp_path / "c.jsonl"), deployments={"gpt-4o"})
    server = gateway.serve(cfg, 0)
    try:
        client = OpenAI(api_key="sk-dummy", base_url=f"http://127.0.0.1:{server.server_address[1]}/v1")
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}])
        assert FakeDial.seen[-1]["path"].startswith("/openai/deployments/gpt-4o/")       # DIAL has it: honoured
        client.chat.completions.create(model="gpt-4-turbo", messages=[{"role": "user", "content": "hi"}])
        assert FakeDial.seen[-1]["path"].startswith("/openai/deployments/gpt-4.1-mini/")  # unknown: mapped
    finally:
        server.shutdown()
        upstream.shutdown()


def test_gateway_fails_fast_when_dial_is_unreachable(tmp_path):
    # nothing listens on this port: every forward is a network failure
    cfg = gateway.Config("http://127.0.0.1:9", "K", "2024-02-01", "m", "e", str(tmp_path / "calls.jsonl"))
    codes = [gateway._forward(cfg, "m", "chat/completions", {"messages": []})[0] for _ in range(5)]
    assert codes[:3] == [502, 502, 502] and codes[3:] == [503, 503]  # after 3 failures: immediate 503s
    assert (tmp_path / "upstream_down").exists()                    # the runner stops calling the app
