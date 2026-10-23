from pathlib import Path

from evalgraph.analyzer.extract import extract_file

SRC = '''
from langgraph.graph import StateGraph, START, END
import requests

TOP_K = 10
USE_RERANKER = False
HTTP_TIMEOUT: float | None = None
SYSTEM = (
    "You are a bot. "
    "Internal debug note: build tag X-1."
)
TOOLS = {"a": tool_a, "b": tool_b}

def tool_a(x, *, timeout=None, temperature=0.9): return requests.get("https://x", timeout=timeout)
def tool_b(x): return x

def build():
    g = StateGraph(dict)
    for name, fn in [("guard", guard), ("route", route)]:
        g.add_node(name, fn)
    g.add_edge(START, "guard")
    g.add_conditional_edges("route", pick, {"x": "guard", **{r: "call_tool" for r in TOOLS}})
    g.add_edge("guard", END)
    return g

def call_tool(state):
    return TOOLS[state["r"]](1)
'''


def _facts(tmp_path: Path):
    p = tmp_path / "m.py"
    p.write_text(SRC, encoding="utf-8")
    return extract_file(p, tmp_path)


def test_consts_prompts_registries(tmp_path):
    f = _facts(tmp_path)
    consts = {c.name: c.value for c in f.consts}
    assert consts["TOP_K"] == 10 and consts["USE_RERANKER"] is False and consts["HTTP_TIMEOUT"] is None
    assert [p.name for p in f.prompts if p.has_internal_notes] == ["SYSTEM"]
    assert f.registries[0].entries == {"a": "tool_a", "b": "tool_b"}


def test_defaults_and_kwargs(tmp_path):
    f = _facts(tmp_path)
    tool_a = next(d for d in f.defs if d.name == "tool_a")
    assert tool_a.defaults == {"timeout": None, "temperature": 0.9}
    call = next(c for c in tool_a.calls if c.callee == "requests.get")
    assert call.kwargs == {"timeout": "timeout"} and call.str_args == ["https://x"]


def test_langgraph_facts(tmp_path):
    f = _facts(tmp_path)
    nodes = {lg.node: lg.fn for lg in f.langgraph if lg.kind == "add_node"}
    assert nodes == {"guard": "guard", "route": "route"}
    edges = {(lg.source, lg.target) for lg in f.langgraph if lg.kind == "add_edge"}
    assert edges == {("__start__", "guard"), ("guard", "__end__")}
    cond = next(lg for lg in f.langgraph if lg.kind == "add_conditional_edges")
    assert cond.mapping == {"x": "guard", "@registry:TOOLS": "call_tool"}
    call_tool = next(d for d in f.defs if d.name == "call_tool")
    assert "TOOLS" in call_tool.names_used


def test_syntax_error_is_recorded(tmp_path):
    p = tmp_path / "bad.py"
    p.write_text("def (:\n", encoding="utf-8")
    assert extract_file(p, tmp_path).error.startswith("SyntaxError")


def test_tree_urls_split_into_repo_branch_folder():
    import pytest

    from evalgraph.analyzer.clone import SourceError, split_tree_url, validate_url

    url = validate_url("https://github.com/org/mono/tree/main/apps/chatbot/")
    assert split_tree_url(url) == ("https://github.com/org/mono", "main", "apps/chatbot")
    assert split_tree_url("https://gitlab.com/g/r/-/tree/dev") == ("https://gitlab.com/g/r", "dev", None)
    assert split_tree_url("https://github.com/o/r") == ("https://github.com/o/r", None, None)
    with pytest.raises(SourceError):
        split_tree_url("https://github.com/o/r/tree/main/../../etc")
