"""Test fixture: tries one allowlisted host and one that is not, through whatever proxy the environment sets."""

import json
import urllib.request


def _try(url: str) -> str:
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            return f"ok {r.status}"
    except Exception as exc:  # the result is the point
        return f"blocked: {type(exc).__name__}"


def answer(question: str) -> str:
    return json.dumps({"pypi": _try("https://pypi.org/simple/requests/"), "example": _try("https://example.com/")})
