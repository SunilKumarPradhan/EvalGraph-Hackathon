"""Test fixture: reports which credentials the app process can see, then asks the LLM once."""

import json
import os

from openai import OpenAI


def answer(question: str) -> str:
    visible = sorted(k for k in os.environ if "KEY" in k or k.startswith("DIAL"))
    reply = OpenAI().chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": question}])
    return json.dumps({"visible": visible, "openai_key": os.environ.get("OPENAI_API_KEY", "")[:20],
                       "base_url": os.environ.get("OPENAI_BASE_URL", ""), "reply": reply.choices[0].message.content})
