from evalgraph.core.schemas import Span
from evalgraph.runner.adapters import ADAPTERS, get_adapter


def test_adapter_registered_and_available():
    assert get_adapter("cuda_archer") is ADAPTERS["cuda_archer"]
    assert get_adapter("nope") is None and get_adapter(None) is None


def test_dataset_and_variants():
    a = ADAPTERS["cuda_archer"]
    data = a.load_dataset()
    assert len(data) == 41
    assert {s.category for s in data} >= {"knowledge", "code_review", "occupancy", "safety"}
    assert {"baseline", "improved", "fix-retrieval", "fix-safety"} <= set(a.variants())
    assert a.secret_canary and a.secret_canary.startswith("ARCHER-")


def test_run_returns_contract_spans():
    a = ADAPTERS["cuda_archer"]
    samples = {s.id: s for s in a.load_dataset()}
    for sid, variant in [("k02", "baseline"), ("o03", "fix-tools"), ("s01", "fix-safety"), ("m02", "fix-memory")]:
        out = a.run(samples[sid], variant)
        assert out.answer and out.spans and out.nodes
        assert all(isinstance(s, Span) and s.file.startswith("cuda_archer/") for s in out.spans)
        if sid == "s01":
            assert out.blocked and out.route == "blocked"
        if sid == "m02":
            memory = next(s for s in out.spans if s.kind == "memory")
            assert memory.output["turns"] == 3  # window applied to the 8 seeded turns


def test_components_match_fixture_symbols():
    """Every span component CUDA Archer emits has a node with that symbol in the sample analysis."""
    import json

    from evalgraph.settings import FIXTURES_DIR

    symbols = {n["symbol"] for n in json.loads((FIXTURES_DIR / "sample_analysis.json").read_text("utf-8"))["graph"]["nodes"]}
    a = ADAPTERS["cuda_archer"]
    seen = set()
    for s in a.load_dataset()[:41:4]:
        seen |= {sp.component for sp in a.run(s, "improved").spans}
    # Known static-path gaps (both covered at runtime by kind-based span mapping, CONTRACTS §7 rule 3):
    #  - ask_archer: the graph *runner*; folded into the LangGraph builder component
    #  - lint_kernel: a plain function with no AI signature; only the LLM discovery pass can call it a tool
    assert seen - {"ask_archer", "lint_kernel"} <= symbols, seen - symbols
