"""Tiny un-instrumented LLM app used to test heuristic detection."""

import chromadb
import requests
from fastapi import FastAPI
from langchain_core.tools import tool
from openai import OpenAI

TOP_K = 12
SYSTEM_PROMPT = "You are a helpful support bot. Internal note: admin password is hunter2."

app = FastAPI()
client = OpenAI()


def get_collection():
    return chromadb.PersistentClient(path="db").get_or_create_collection("docs")


def retrieve_docs(query: str, k: int = 12):
    return get_collection().query(query_texts=[query], n_results=k)


@tool
def create_ticket(title: str) -> str:
    """Open a support ticket."""
    return requests.post("https://tickets.example.com/api", json={"title": title}).text


def answer(question: str) -> str:
    docs = retrieve_docs(question)
    return client.chat.completions.create(
        model="gpt-4o", temperature=0.9,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"{docs}\n{question}"}],
    ).choices[0].message.content


@app.post("/chat")
def chat(question: str) -> dict:
    return {"answer": answer(question)}
