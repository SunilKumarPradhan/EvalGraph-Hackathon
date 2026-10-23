"""Minimal tool-calling agent on the raw OpenAI SDK."""

import json

import requests
from openai import OpenAI

client = OpenAI()

TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "search_web", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}}},
]


def get_weather(city: str) -> str:
    return requests.get(f"https://wttr.in/{city}?format=3").text


def search_web(q: str) -> str:
    return requests.get("https://api.duckduckgo.com/", params={"q": q, "format": "json"}).text[:2000]


TOOL_FUNCS = {"get_weather": get_weather, "search_web": search_web}


def run_agent(task: str) -> str:
    messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": task}]
    while True:
        response = client.chat.completions.create(model="gpt-4o", messages=messages, tools=TOOLS)
        msg = response.choices[0].message
        if not msg.tool_calls:
            return msg.content
        messages.append(msg)
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments)
            result = TOOL_FUNCS[call.function.name](**args)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})


if __name__ == "__main__":
    print(run_agent(input("task> ")))
