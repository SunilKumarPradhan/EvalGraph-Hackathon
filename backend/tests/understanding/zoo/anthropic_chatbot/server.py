from anthropic import Anthropic
from flask import Flask, jsonify, request

app = Flask(__name__)
client = Anthropic(api_key="sk-ant-api03-0123456789abcdefghijklmnopqrstuvwxyzABCD")

SYSTEM_TEMPLATE = (
    "You are Ava, the support assistant for Contoso. The customer asked: {question}. "
    "Always be polite and answer as helpfully as you can."
)

conversations: dict[str, list[dict]] = {}


def get_history(session_id: str) -> list[dict]:
    return conversations.setdefault(session_id, [])


def ask_claude(session_id: str, question: str) -> str:
    history = get_history(session_id)
    history.append({"role": "user", "content": question})
    reply = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        system=SYSTEM_TEMPLATE.format(question=question),
        messages=history,
    )
    text = reply.content[0].text
    history.append({"role": "assistant", "content": text})
    return text


@app.route("/chat", methods=["POST"])
def chat():
    body = request.get_json()
    return jsonify({"answer": ask_claude(body["session_id"], body["question"])})
