from fastapi import FastAPI
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel

Settings.llm = OpenAI(model="gpt-4o", temperature=0.2)
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=12)

app = FastAPI()


class Query(BaseModel):
    question: str


@app.post("/query")
def query(q: Query) -> dict:
    response = query_engine.query(q.question)
    return {"answer": str(response), "sources": [n.node.metadata for n in response.source_nodes]}


@app.get("/health")
def health() -> dict:
    return {"ok": True}
