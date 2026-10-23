"""The LLM half (map, reduce, review) against a scripted fake DIAL endpoint: claims are verified, never trusted."""

import json

import httpx
import pytest

from evalgraph import llm
from evalgraph.settings import Settings
from evalgraph.understanding import cards, pipeline, review

from .conftest import ZOO

REAL = Settings(EVALGRAPH_MOCK_LLM=False, DIAL_API_KEY="k", DIAL_ENDPOINT="https://dial.example", EVALGRAPH_MAX_CONCURRENCY=1)


def _completion(obj):
    text = json.dumps(obj)
    return {"id": "x", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}}


class Script:
    """Answers by prompt type; records what was asked."""

    def __init__(self, fail=False):
        self.fail, self.seen = fail, []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        system, user = body["messages"][0]["content"], body["messages"][-1]["content"]
        self.seen.append(system[:40])
        if self.fail:
            return httpx.Response(400, json={"error": {"message": "boom"}})
        if "map the architecture" in system:
            slices = [ln.split()[2] for ln in user.splitlines() if ln.startswith("### slice ")]
            out = []
            for sid in slices:
                line = int(next(l for l in user.split(f"### slice {sid}")[1].splitlines()[1:] if "|" in l).split("|")[0])
                out.append({"slice_id": sid, "is_component": True, "type": "tool" if sid.startswith("d") else None,
                            "label": f"Label {sid}", "purpose": f"Purpose {sid}", "evidence_lines": [line, 99999]})
            out.append({"slice_id": "nope", "label": "ghost"})
            return httpx.Response(200, json=_completion({"cards": out}))
        if "assemble the request pipeline" in system:
            payload = json.loads(user)
            ids = [c["id"] for c in payload["components"]]
            return httpx.Response(200, json=_completion({"add_edges": [{"source": ids[0], "target": ids[-1], "kind": "feeds"},
                                                                        {"source": "ghost:x", "target": ids[0], "kind": "feeds"}],
                                                         "remove_edges": ["not-an-edge"]}))
        if "evaluation hotspots" in system:
            payload = json.loads(user)
            comp = payload["components"][0]
            first = int(comp["slice"].splitlines()[1].split("|")[0])
            verdicts = [{"finding_id": f["id"], "keep": True, "why": "Grounded reason."} for c in payload["components"] for f in c["findings"]]
            return httpx.Response(200, json=_completion({
                "verdicts": verdicts,
                "additions": [{"rule_id": "LLM_IN_LOOP", "node_id": comp["node_id"], "evidence_lines": [first], "why": "LLM says so."},
                              {"rule_id": "MADE_UP_RULE", "node_id": comp["node_id"], "evidence_lines": [first]},
                              {"rule_id": "LLM_NO_RETRY", "node_id": comp["node_id"], "evidence_lines": [123456]}]}))
        return httpx.Response(200, json=_completion({}))


@pytest.fixture
def live(monkeypatch):
    def install(script):
        for mod in (llm, pipeline, cards, review):
            monkeypatch.setattr(mod, "get_settings", lambda: REAL, raising=False)
        monkeypatch.setattr(llm, "_get_cache", lambda: _NoCache())
        monkeypatch.setattr(pipeline, "dial_reachable", lambda: True)  # the fake DIAL below is "reachable"
        llm.set_client(llm.DialClient(REAL, http_client=httpx.Client(transport=httpx.MockTransport(script))))
        return script
    yield install
    llm.set_client(None)


class _NoCache(dict):
    def set(self, k, v):
        pass


def test_llm_path_is_verified(live):
    script = live(Script())
    u = pipeline.understand({"local_path": str(ZOO / "openai_agent")}, lambda *a, **k: None)
    assert any("map the architecture" in s for s in [*script.seen, ""]) or u.llm_calls
    assert u.llm_calls >= 3
    labelled = [n for n in u.graph.nodes if n.label.startswith("Label ")]
    assert labelled and all(n.description and n.description.startswith("Purpose") for n in labelled)
    assert not any(n.id.startswith("ghost") for n in u.graph.nodes)
    assert not any("ghost" in e.id for e in u.graph.edges)
    llm_added = [h for h in u.hotspots if h.source == "llm"]
    assert all(h.rule_id != "MADE_UP_RULE" for h in u.hotspots)
    for h in llm_added:  # every LLM-added hotspot cites a line that exists
        assert all(e.line and e.line < 1000 for e in h.evidence)
    assert any(h.source == "static+llm" and h.why == "Grounded reason." for h in u.hotspots)
    assert u.stats["review"]["invalid"] >= 2 and u.stats["cards"]["rejected"] >= 1


def test_llm_failures_fall_back_to_static(live):
    live(Script(fail=True))
    u = pipeline.understand({"local_path": str(ZOO / "openai_agent")}, lambda *a, **k: None)
    assert {n.type for n in u.graph.nodes} >= {"agent", "tool", "external_api"}
    assert u.hotspots and all(h.source == "static" for h in u.hotspots)
    assert u.stats["cards"]["failed_batches"] >= 1
