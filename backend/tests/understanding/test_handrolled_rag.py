"""A RAG app with no RAG library (OpenAI embeddings + cosine ranking): the analysis must still see the retriever."""

from pathlib import Path

from evalgraph.understanding.pipeline import understand

APP = Path(__file__).resolve().parents[1] / "runtime" / "fixtures" / "rag_app"


def test_handrolled_rag_is_understood():
    u = understand({"local_path": str(APP)}, lambda *a, **k: None)
    ids = {n.id for n in u.graph.nodes}
    assert "retriever:retrieve" in ids                       # embeds the query + ranks by cosine similarity
    assert not {"llm:retrieve", "llm:build_index"} & ids     # client.embeddings.create is not a chat call
    assert "retriever:build_index" not in ids                # sorted(files) is not similarity ranking
    edges = {(e.source, e.target, e.kind) for e in u.graph.edges}
    assert ("retriever:retrieve", "llm:answer", "feeds") in edges
    patches = {h.rule_id: [e.summary for e in h.patch] for h in u.hotspots if h.patch}
    assert patches.get("RAG_TOPK_NO_RERANK") == ["TOP_K 10 → 4"]
    assert patches.get("LLM_HIGH_TEMP") == ["TEMPERATURE 0.9 → 0.2"]
