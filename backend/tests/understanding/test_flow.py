"""LLM Flow Graph (PHASE-11): what each prompt is built from and where model output goes, static path only."""

from evalgraph.understanding.flow import SYSTEM_CHANNELS


def _recipe(u, comp):
    return next(r for r in u.flow.recipes if r.comp == comp)


def _flows(u):
    return {(f.source, f.target, f.label) for f in u.flow.flows}


def test_rag_prompt_is_built_from_retrieval_and_the_question(zoo):
    r = _recipe(zoo["langchain_rag"], "llm:answer")
    assert {"retrieved", "user_input", "prompt"} <= set(r.parts)
    assert ("retriever:answer", "llm:answer", "retrieved") in _flows(zoo["langchain_rag"])


def test_agent_tool_results_reenter_the_model(zoo):
    u = zoo["openai_agent"]
    r = _recipe(u, "llm:run_agent")
    assert r.has_tools and {"tool_output", "external", "user_input"} <= set(r.parts)
    rules = {h.rule_id: h for h in u.hotspots}
    assert {"INDIRECT_INJECTION", "LLM_OUTPUT_UNVALIDATED"} <= set(rules)
    tool_sink = next(s for s in u.flow.sinks if s.kind == "tool")
    assert {t.comp for t in tool_sink.taints("llm_output")} == {"llm:run_agent"} and not tool_sink.validated


def test_user_input_in_system_prompt_is_a_flow_fact(zoo):
    u = zoo["anthropic_chatbot"]
    assert "user_input" in _recipe(u, "llm:ask_claude").system
    h = next(h for h in u.hotspots if h.rule_id == "PROMPT_INJECTION_SURFACE")
    assert any(e.fact == "user input flows into the system prompt" for e in h.evidence)
    sink = next(s for s in u.flow.sinks if s.kind == "llm")
    assert SYSTEM_CHANNELS & set(sink.channels)


def test_model_chosen_tool_args_reaching_eval(zoo):
    u = zoo["langgraph_agent"]
    h = next(h for h in u.hotspots if h.rule_id == "LLM_OUTPUT_TO_EXEC")
    assert h.node_id == "tool:calculator" and h.severity == "high"
    assert _recipe(u, "llm:call_model").has_tools  # .bind_tools([...]) at import time


def test_js_apps_degrade_gracefully(zoo):
    """No data-flow facts for JS/TS yet: model calls are known, what feeds them is not (no invented flows)."""
    for app in ("ts_nextjs_rag", "vercel_ai_sdk"):
        f = zoo[app].flow
        assert f is not None and not f.flows and all(not s.channels for s in f.sinks)


def test_flow_evidence_points_at_real_lines(zoo, archer):
    for u in [*zoo.values(), archer]:
        for f in u.flow.flows:
            for e in f.via:
                assert e.line and e.line <= len(u.buffer.files[e.file].splitlines()), (u.buffer.name, f, e)


def test_archer_pipeline_from_data_flow(archer):
    flows = _flows(archer)
    assert {("retriever:retrieve", "llm:generate_answer", "retrieved"), ("reranker:rerank", "llm:generate_answer", "retrieved"),
            ("external_api:fetch_driver_matrix", "llm:generate_tool_answer", "external"),
            ("router:route_query", "parser:parse_route", "llm_output"),
            ("memory:conversation_memory", "llm:generate_answer", "memory")} <= flows
    # graph state is per key: the answer LLM reads the question, not the rewritten query; a shared helper
    # (question_block) must not leak one caller's data into another caller
    assert ("llm:rewrite_query", "llm:generate_answer", "llm_output") not in flows
    assert ("llm:rewrite_query", "llm:review_kernel", "llm_output") not in flows
    labelled = {(e.source, e.target): e.flow_labels for e in archer.graph.edges if e.flow_labels}
    assert "retrieved" in labelled[("retriever:retrieve", "llm:generate_answer")]


def test_flow_recommendations(archer, zoo):
    recs = {(r.node_id, r.evaluator_id): r for r in zoo["openai_agent"].recommendations}
    assert ("llm:run_agent", "injection_resistance") in recs
    assert any(r.node_id == "llm:generate_answer" and r.evaluator_id == "faithfulness" for r in archer.recommendations)
    assert ("parser:parse_route", "parse_success") in {(r.node_id, r.evaluator_id) for r in archer.recommendations}


def test_targeted_probes_follow_hotspots(zoo):
    ds = zoo["openai_agent"].runtime["dataset"]
    cats = {s["category"] for s in ds}
    assert "probe:INDIRECT_INJECTION" in cats
    probe = next(s for s in ds if s["category"] == "probe:INDIRECT_INJECTION")
    assert probe["expected_behavior"] == "refuse" and "targeted" in probe["probes"]
