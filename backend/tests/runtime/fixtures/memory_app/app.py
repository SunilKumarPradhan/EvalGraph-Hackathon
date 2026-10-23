"""Test fixture: a chat app whose memory lives in the process (like an in-memory checkpointer)."""

HISTORY: list[str] = []


def chat(message: str) -> str:
    HISTORY.append(message)
    return f"turn {len(HISTORY)}"
