"""Tiny RAG service over local markdown docs (runtime fixture): OpenAI embeddings, cosine search, chat answer."""

import math
from pathlib import Path

from openai import OpenAI

DOCS = Path(__file__).parent / "docs"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 10            # pulls in almost the whole knowledge base for every question
TEMPERATURE = 0.9     # creative sampling on a factual assistant

client = OpenAI()
_index: list[tuple[str, list[float]]] | None = None


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)) or 1.0)


def build_index() -> list[tuple[str, list[float]]]:
    chunks = [p.strip() for doc in sorted(DOCS.glob("*.md")) for p in doc.read_text(encoding="utf-8").split("\n\n")
              if p.strip() and not p.startswith("#")]
    vectors = client.embeddings.create(model=EMBED_MODEL, input=chunks).data
    return [(chunk, v.embedding) for chunk, v in zip(chunks, vectors)]


def retrieve(question: str) -> list[str]:
    global _index
    if _index is None:
        _index = build_index()
    query = client.embeddings.create(model=EMBED_MODEL, input=[question]).data[0].embedding
    ranked = sorted(_index, key=lambda item: _cosine(query, item[1]), reverse=True)
    return [chunk for chunk, _ in ranked[:TOP_K]]


def answer(question: str) -> str:
    context = "\n\n".join(retrieve(question))
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=TEMPERATURE,
        max_tokens=400,
        messages=[{"role": "system", "content": "You answer questions about the Orbit coffee machine using the context."},
                  {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
    )
    return response.choices[0].message.content or ""
