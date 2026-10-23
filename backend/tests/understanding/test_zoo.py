"""Generality: eight differently built LLM apps (no CUDA Archer conventions), static path only (mock mode)."""

import pytest

# app -> (required component types, required hotspot rules, required edges as (source_type, target_type))
EXPECT = {
    "langchain_rag": ({"entrypoint", "retriever", "vector_db", "embedding", "dataset", "llm", "prompt"},
                      {"RAG_TOPK_NO_RERANK", "RAG_BIG_CHUNKS", "INDEX_REBUILT_PER_REQUEST", "LLM_HIGH_TEMP",
                       "MEMORY_UNBOUNDED", "NO_GUARDRAIL"},
                      {("entrypoint", "retriever"), ("retriever", "llm"), ("retriever", "vector_db"), ("prompt", "llm")}),
    "llamaindex_qa": ({"entrypoint", "retriever", "vector_db", "dataset", "llm"},
                      {"RAG_TOPK_NO_RERANK", "NO_GUARDRAIL"},
                      {("entrypoint", "retriever"), ("vector_db", "retriever"), ("retriever", "llm")}),
    "openai_agent": ({"entrypoint", "agent", "llm", "tool", "external_api"},
                     {"LLM_IN_LOOP", "PARSER_NAIVE_JSON", "EXT_NO_TIMEOUT", "MEMORY_UNBOUNDED", "NO_GUARDRAIL"},
                     {("entrypoint", "agent"), ("llm", "tool"), ("tool", "external_api")}),
    "anthropic_chatbot": ({"entrypoint", "llm", "memory"},
                          {"HARDCODED_SECRET", "PROMPT_INJECTION_SURFACE", "MEMORY_UNBOUNDED", "NO_GUARDRAIL"},
                          {("entrypoint", "llm")}),
    "langgraph_agent": ({"agent", "llm", "tool", "memory"},
                        {"NO_GUARDRAIL"},
                        {("agent", "llm"), ("llm", "tool"), ("tool", "llm")}),
    "ts_nextjs_rag": ({"entrypoint", "retriever", "vector_db", "embedding", "dataset", "llm", "external_api"},
                      {"RAG_TOPK_NO_RERANK", "RAG_BIG_CHUNKS", "INDEX_REBUILT_PER_REQUEST", "LLM_HIGH_TEMP",
                       "PARSER_NAIVE_JSON", "EXT_NO_TIMEOUT"},
                      {("retriever", "llm"), ("vector_db", "retriever")}),
    "vercel_ai_sdk": ({"entrypoint", "agent", "tool", "external_api"},
                      {"LLM_HIGH_TEMP", "EXT_NO_TIMEOUT", "NO_GUARDRAIL"},
                      {("entrypoint", "agent"), ("agent", "tool"), ("tool", "external_api")}),
    "notebook_rag": ({"retriever", "vector_db", "embedding", "llm"},
                     {"RAG_TOPK_NO_RERANK", "LLM_IN_LOOP"},
                     {("retriever", "llm"), ("retriever", "vector_db")}),
}


@pytest.mark.parametrize("app", sorted(EXPECT))
def test_zoo_app(zoo, app):
    u = zoo[app]
    types, rules, edges = EXPECT[app]
    found_types = {n.type for n in u.graph.nodes}
    assert types <= found_types, f"missing types: {types - found_types}"
    found_rules = {h.rule_id for h in u.hotspots}
    assert rules <= found_rules, f"missing hotspots: {rules - found_rules}"
    t = {n.id: n.type for n in u.graph.nodes}
    found_edges = {(t[e.source], t[e.target]) for e in u.graph.edges}
    assert edges <= found_edges, f"missing edges: {edges - found_edges}"


def test_every_hotspot_has_verifiable_evidence(zoo):
    for name, u in zoo.items():
        nodes = {n.id for n in u.graph.nodes}
        for h in u.hotspots:
            assert h.node_id in nodes and h.evidence, (name, h.id)
            for e in h.evidence:
                if e.line is not None and e.snippet and e.snippet != "<redacted>":
                    # the cited line really says this (snippets are copied from the buffer)
                    assert e.snippet[:30] in u.buffer.files[e.file].splitlines()[e.line - 1], (name, h.id, e)


def test_languages_detected(zoo):
    assert zoo["ts_nextjs_rag"].languages == ["typescript"]
    assert zoo["notebook_rag"].languages == ["python (notebook)"]
    assert "vercel-ai-sdk" in zoo["vercel_ai_sdk"].frameworks


def test_no_false_metadata_or_retriever_on_agents(zoo):
    """Apps without retrieval must not get RAG findings."""
    for app in ("openai_agent", "anthropic_chatbot", "langgraph_agent", "vercel_ai_sdk"):
        assert not {h.rule_id for h in zoo[app].hotspots} & {"RAG_TOPK_NO_RERANK", "RAG_BIG_CHUNKS", "RAG_NO_METADATA_FILTER"}
