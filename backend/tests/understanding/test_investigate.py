"""Bounded investigator (PHASE-11) against a scripted fake DIAL endpoint: tool results are the only citable lines."""

import json
import re

import httpx
import pytest

from evalgraph import llm
from evalgraph.settings import Settings
from evalgraph.understanding import cards, investigate, pipeline, review

from .conftest import ZOO
from .test_llm_stages import REAL, Script, _completion, _NoCache

APP = ZOO / "openai_agent"


class Investigator(Script):
    """Scripted investigator: turn 1 asks for tools, turn 2 finishes. Everything else answers like Script."""

    def __init__(self, finish=True, fail=False):
        super().__init__()
        self.finish, self.fail_investigator, self.turns = finish, fail, []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        system, messages = body["messages"][0]["content"], body["messages"][1:]
        if "auditing an LLM application" not in system:
            if "evaluation hotspots" in system:  # batch review: no opinions, so investigator verdicts stand out
                return httpx.Response(200, json=_completion({"verdicts": [], "additions": []}))
            return super().__call__(request)
        self.turns.append(messages)
        if self.fail_investigator:
            return httpx.Response(400, json={"error": {"message": "boom"}})
        header = json.loads(messages[0]["content"].split("\n\n", 1)[0])
        self.finding_ids = header["finding_ids"]
        if len(messages) == 1 or not self.finish:
            return httpx.Response(200, json=_completion({"thought": "look", "calls": [
                {"tool": "open_component", "args": {"id": "llm:run_agent"}},
                {"tool": "read", "args": {"file": "agent.py", "start": 1, "end": 5}},
                {"tool": "flows", "args": {"id": "llm:run_agent"}},
                {"tool": "open_component", "args": {"id": "ghost:x"}}]}))
        results = messages[-1]["content"]
        assert "agent.py:3| " in results and "unknown node id" in results
        return httpx.Response(200, json=_completion({"done": True,
            "verdicts": [{"finding_id": self.finding_ids[0], "keep": True, "why": "Investigated reason."},
                         {"finding_id": "MADE_UP:x", "keep": False}],
            "additions": [{"rule_id": "LLM_NO_RETRY", "node_id": "llm:run_agent", "file": "agent.py", "evidence_lines": [3]},
                          {"rule_id": "LLM_HIGH_TEMP", "node_id": "llm:run_agent", "file": "agent.py", "evidence_lines": [40]}],
            "evals": [{"node_id": "llm:run_agent", "evaluator_id": "citation_correctness", "rationale": "Answers cite web results",
                       "probes": ["What is the weather in Oslo? Cite where you got it."]},
                      {"node_id": "llm:run_agent", "evaluator_id": "context_precision"},
                      {"node_id": "ghost:x", "evaluator_id": "safety"}]}))


@pytest.fixture
def live(monkeypatch):
    def install(script, settings=REAL):
        for mod in (llm, pipeline, cards, review, investigate):
            monkeypatch.setattr(mod, "get_settings", lambda: settings, raising=False)
        monkeypatch.setattr(llm, "_get_cache", lambda: _NoCache())
        monkeypatch.setattr(pipeline, "dial_reachable", lambda: True)
        llm.set_client(llm.DialClient(settings, http_client=httpx.Client(transport=httpx.MockTransport(script))))
        return script
    yield install
    llm.set_client(None)


def _run():
    return pipeline.understand({"local_path": str(APP)}, lambda *a, **k: None)


def test_investigator_claims_are_verified(live):
    script = live(Investigator())
    u = _run()
    assert len(script.turns) == 2 and script.turns[0][0]["content"].count("# AI MAP") == 1
    hs = {h.id: h for h in u.hotspots}
    first = hs[script.finding_ids[0]]
    assert first.why == "Investigated reason." and first.source == "static+llm"
    added = hs["LLM_NO_RETRY:llm:run_agent"]  # line 3 was shown by `read`
    assert added.source == "llm" and [(e.file, e.line) for e in added.evidence] == [("agent.py", 3)]
    assert "LLM_HIGH_TEMP:llm:run_agent" not in hs  # line 40 was never shown to the model
    st = u.stats["investigation"]
    assert st["finished"] and st["tool_calls"] == 4 and st["evals"] == 1 and st["evals_rejected"] == 2
    assert u.stats["review"]["invalid"] >= 2
    rec = next(r for r in u.recommendations if r.evaluator_id == "citation_correctness")
    assert rec.node_id == "llm:run_agent" and rec.rationale.startswith("Investigator:")
    probes = [s for s in u.runtime["dataset"] if s["category"] == "probe:citation_correctness"]
    assert probes and probes[0]["question"].startswith("What is the weather in Oslo")


def test_investigator_budget_forces_an_answer(live):
    script = live(Investigator(finish=False), REAL.model_copy(update={"investigator_turns": 2}))
    u = _run()
    assert len(script.turns) == 3  # 2 tool turns + the forced final turn
    assert "Budget used" in script.turns[-1][-1]["content"]
    assert u.stats["investigation"]["turns"] == 3 and not u.stats["investigation"]["finished"]


def test_investigator_failure_falls_back_to_batch_review(live):
    script = live(Investigator(fail=True))
    u = _run()
    assert script.turns and u.stats["investigation"] is None
    assert u.stats["review"]["calls"] >= 1 and u.hotspots


def test_investigator_can_be_switched_off(live):
    script = live(Investigator(), REAL.model_copy(update={"investigator": False}))
    u = _run()
    assert not script.turns and u.stats["investigation"] is None


def test_read_tool_is_capped_and_records_lines(zoo):
    u = zoo["openai_agent"]
    from evalgraph.understanding.components import SymbolGraph

    tools = investigate.Tools(u.buffer, [], {}, [], SymbolGraph(), u.flow)
    text = tools.read("agent.py", 1, 500)
    assert len(text.splitlines()) <= investigate.MAX_READ_LINES and tools.seen["agent.py"] == set(range(1, len(text.splitlines()) + 1))
    assert re.search(r"error: KeyError", tools.run(investigate.ToolCall(tool="read", args={"file": "nope.py"})))
