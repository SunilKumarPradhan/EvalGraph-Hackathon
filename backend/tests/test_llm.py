import json

import httpx
import pytest
from pydantic import BaseModel

from evalgraph import llm
from evalgraph.settings import Settings


class Judge(BaseModel):
    score: float
    reason: str


class Nested(BaseModel):
    name: str
    tags: list[str]
    inner: Judge
    maybe: int | None


def test_mock_text_and_json():
    assert llm.complete_text("anything", "sys", "user").text == "[MOCK] anything"
    obj, res = llm.complete_json("some_prompt", "s", "u", Nested)
    assert res.mock and obj.name == "" and obj.inner.score == 0.0 and obj.maybe is None


def test_mock_judge_is_deterministic():
    a, _ = llm.complete_json("judge_x", "s", "question one", Judge)
    b, _ = llm.complete_json("judge_x", "s", "question one", Judge)
    c, _ = llm.complete_json("judge_x", "s", "question two", Judge)
    assert a.score == b.score and 0 <= a.score <= 1 and a.reason.startswith("[MOCK]")
    assert a.score != c.score


# ---- real DIAL path against a fake Azure OpenAI endpoint

REAL = Settings(EVALGRAPH_MOCK_LLM=False, DIAL_API_KEY="k", DIAL_ENDPOINT="https://dial.example",
                EVALGRAPH_MODEL="gpt-4o")


def _completion(text, tin=10, tout=5):
    return {"id": "x", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": tin, "completion_tokens": tout, "total_tokens": tin + tout}}


@pytest.fixture
def real_mode(monkeypatch):
    monkeypatch.setattr(llm, "get_settings", lambda: REAL)
    monkeypatch.setattr(llm, "_get_cache", lambda: {})  # plain dict: no .get/.set persistence issues
    yield
    llm.set_client(None)


class DictCache(dict):
    def set(self, k, v):
        self[k] = v


def test_json_repair_retry(real_mode, monkeypatch):
    monkeypatch.setattr(llm, "_get_cache", lambda: DictCache())
    replies = iter(["```json\n{\"score\": \"bad\"}\n```", "{\"score\": 0.75, \"reason\": \"ok\"}"])
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=_completion(next(replies)))

    llm.set_client(llm.DialClient(REAL, http_client=httpx.Client(transport=httpx.MockTransport(handler))))
    obj, res = llm.complete_json("judge_test", "Judge.", "Q", Judge)
    assert obj.score == 0.75 and len(seen) == 2
    assert "Validation error" in seen[1]["messages"][-1]["content"]
    assert res.cost_usd > 0 and res.tokens_in == 10


def test_temperature_fallback(real_mode, monkeypatch):
    monkeypatch.setattr(llm, "_get_cache", lambda: DictCache())
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if "temperature" in body:
            return httpx.Response(400, json={"error": {"message": "temperature is deprecated for this model"}})
        return httpx.Response(200, json=_completion("hi"))

    llm.set_client(llm.DialClient(REAL, http_client=httpx.Client(transport=httpx.MockTransport(handler))))
    assert llm.complete_text("t", "s", "u1", temperature=0.3).text == "hi"
    assert llm.complete_text("t", "s", "u2", temperature=0.3).text == "hi"
    assert ["temperature" in b for b in bodies] == [True, False, False]


def test_cache_hit_skips_network(real_mode, monkeypatch):
    cache = DictCache()
    monkeypatch.setattr(llm, "_get_cache", lambda: cache)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(200, json=_completion("cached answer"))

    llm.set_client(llm.DialClient(REAL, http_client=httpx.Client(transport=httpx.MockTransport(handler))))
    first = llm.complete_text("t", "s", "same")
    second = llm.complete_text("t", "s", "same")
    assert calls["n"] == 1 and not first.cached and second.cached and second.text == "cached answer"


def test_rate_limit_waits_out_the_minute_window(real_mode, monkeypatch):
    monkeypatch.setattr(llm, "_get_cache", lambda: DictCache())
    waits = []
    monkeypatch.setattr(llm.time, "sleep", waits.append)
    minute = {"error": {"message": "Hit token rate limit. Minute limit: 128933 / 120000 tokens. Day limit: 1 / 2000000 tokens.",
                        "code": "429"}}
    replies = iter([httpx.Response(429, json=minute)] * 4 + [httpx.Response(200, json=_completion("fine"))])
    llm.set_client(llm.DialClient(REAL, http_client=httpx.Client(transport=httpx.MockTransport(lambda r: next(replies)))))
    assert llm.complete_text("t", "s", "u").text == "fine"  # 4 rate limits > the 3 normal attempts, still succeeds
    assert len(waits) == 4 and all(15 <= w <= 65 for w in waits)

    day = {"error": {"message": "Hit token rate limit. Minute limit: 10 / 120000 tokens. Day limit: 2000001 / 2000000 tokens.", "code": "429"}}
    llm.set_client(llm.DialClient(REAL, http_client=httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(429, json=day)))))
    with pytest.raises(llm.LLMError, match="day token limit"):
        llm.complete_text("t", "s", "u2")  # no point waiting for a daily window


def test_runs_refuse_quickly_when_dial_is_unreachable(monkeypatch):
    from evalgraph.core.schemas import Attachment, RunRequest
    from evalgraph.runner import run as run_mod

    monkeypatch.setattr(run_mod, "get_settings", lambda: REAL)
    monkeypatch.setattr(run_mod, "dial_reachable", lambda: False)
    monkeypatch.setattr(run_mod.storage, "load", lambda *a, **k: type("A", (), {"summary": type("S", (), {"runnable": True})()})())
    with pytest.raises(run_mod.RunError) as err:
        run_mod.prepare(RunRequest(analysis_id="a_x", attachments=[Attachment(node_id="n", evaluator_id="latency")]))
    assert err.value.status == 503 and "VPN" in err.value.detail
