"""The only way EvalGraph talks to an LLM: DIAL through the Azure OpenAI SDK.

Same pattern as CUDA Archer's DialClient (itself adapted from NoteSmith): retries, self-healing
for deployments that reject `temperature`, token usage + cost. Adds JSON-schema responses with
one repair retry, a disk cache for temperature-0 calls, and a mock mode (no network) for
offline development.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import threading
import time
import types
import typing
from dataclasses import dataclass
from typing import Any, TypeVar, get_args, get_origin

import httpx
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    AzureOpenAI,
    BadRequestError,
    RateLimitError,
)
from pydantic import BaseModel, ValidationError

from .settings import FIXTURES_DIR, Settings, get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# USD per 1M tokens (input, output); public list prices, for comparing runs only.
PRICES_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1": (2.00, 8.00),
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "text-embedding-3-small": (0.02, 0.0),
}
MAX_RETRIES = 3
RATE_LIMIT_RETRIES = 6  # 429s: up to ~5 min of waiting for the per-minute token window
MOCK_DIR = FIXTURES_DIR / "mock_llm"


class LLMError(RuntimeError):
    pass


class LLMJsonError(LLMError):
    pass


@dataclass
class LLMResult:
    text: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    cached: bool = False
    mock: bool = False


def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    key = max((k for k in PRICES_PER_1M if model.startswith(k)), key=len, default=None)
    if key is None:
        return 0.0
    price_in, price_out = PRICES_PER_1M[key]
    return round(tokens_in * price_in / 1e6 + tokens_out * price_out / 1e6, 8)


_reach = {"at": 0.0, "ok": True}


def dial_reachable(timeout: float = 5.0) -> bool:
    """Can this machine reach DIAL at all (VPN)? Any HTTP answer counts. Cached for 30 s."""
    if time.time() - _reach["at"] < 30:
        return _reach["ok"]
    try:
        httpx.get(get_settings().dial_endpoint, timeout=timeout)
        ok = True
    except httpx.HTTPError:
        ok = False
    _reach.update(at=time.time(), ok=ok)
    return ok


# ---------------------------------------------------------------- DIAL client


class DialClient:
    def __init__(self, settings: Settings, http_client: httpx.Client | None = None, timeout: int = 60):
        if not settings.dial_api_key:
            raise LLMError("DIAL_API_KEY is not set (or set EVALGRAPH_MOCK_LLM=1).")
        self._client = AzureOpenAI(
            api_key=settings.dial_api_key,
            api_version=settings.dial_api_version,
            azure_endpoint=settings.dial_endpoint,
            timeout=timeout,
            max_retries=0,
            http_client=http_client,
        )
        self._no_temperature: set[str] = set()

    def _create(self, messages: list[dict[str, str]], model: str, temperature: float, max_tokens: int):
        kwargs: dict[str, Any] = {"model": model, "messages": messages, "max_tokens": max_tokens}
        if model not in self._no_temperature:
            kwargs["temperature"] = temperature
        try:
            return self._client.chat.completions.create(**kwargs)
        except BadRequestError as exc:
            if "temperature" in kwargs and "temperature" in str(exc).lower():
                logger.warning("DIAL deployment %s rejected temperature; retrying without it", model)
                self._no_temperature.add(model)
                kwargs.pop("temperature")
                return self._client.chat.completions.create(**kwargs)
            raise

    def chat(self, messages: list[dict[str, str]], model: str, temperature: float, max_tokens: int) -> LLMResult:
        last: Exception | None = None
        attempt = 0
        rate_limited = 0
        while attempt < MAX_RETRIES:
            attempt += 1
            try:
                response = self._create(messages, model, temperature, max_tokens)
                text = (response.choices[0].message.content or "").strip()
                usage = response.usage
                tin = usage.prompt_tokens if usage else 0
                tout = usage.completion_tokens if usage else 0
                return LLMResult(text=text, model=model, tokens_in=tin, tokens_out=tout,
                                 cost_usd=estimate_cost(model, tin, tout))
            except AuthenticationError as exc:
                raise LLMError(f"DIAL authentication failed: {exc}") from exc
            except BadRequestError as exc:
                raise LLMError(f"DIAL rejected the request (400): {exc}") from exc
            except RateLimitError as exc:
                exceeded = {k.lower() for k, used, cap in re.findall(r"(\w+) limit: (\d+) / (\d+)", str(exc))
                            if int(used) >= int(cap)}
                if exceeded & {"day", "week", "month"}:
                    raise LLMError(f"DIAL {'/'.join(sorted(exceeded))} token limit reached for {model}: {exc}") from exc
                # DIAL limits tokens per minute per deployment: wait out the window, with its own retry budget
                rate_limited += 1
                if rate_limited <= RATE_LIMIT_RETRIES:
                    attempt -= 1
                last, wait = exc, min(15 * rate_limited, 60) + random.uniform(0, 5)
            except (APITimeoutError, APIConnectionError) as exc:
                last, wait = exc, 2 ** attempt
            logger.warning("DIAL %s on %s (attempt %s/%s, rate-limited %s); waiting %.0fs", type(last).__name__, model,
                           attempt, MAX_RETRIES, rate_limited, wait)
            if attempt < MAX_RETRIES:
                time.sleep(wait)
        raise LLMError(f"DIAL call failed after {MAX_RETRIES} attempts ({rate_limited} rate-limited): {last}")


_client: DialClient | None = None
_client_lock = threading.Lock()


def _dial() -> DialClient:
    global _client
    with _client_lock:
        if _client is None:
            _client = DialClient(get_settings())
        return _client


def set_client(client: DialClient | None) -> None:
    """Test hook: inject a client (e.g. one backed by httpx.MockTransport)."""
    global _client
    _client = client


# ---------------------------------------------------------------- cache


_cache = None


def _get_cache():
    global _cache
    if _cache is None:
        import diskcache

        _cache = diskcache.Cache(str(get_settings().resolved_data_dir() / "cache" / "llm"))
    return _cache


def _cache_key(model: str, system: str, user: str, temperature: float, max_tokens: int) -> str:
    raw = json.dumps([model, system, user, temperature, max_tokens], ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- mock


def _hash_unit(text: str) -> float:
    """Deterministic pseudo-random number in [0, 1) from text."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF


def _mock_value(annotation: Any, name: str = "") -> Any:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if annotation is type(None):
        return None
    if origin in (typing.Union, types.UnionType):
        if type(None) in args:
            return None
        return _mock_value(args[0], name)
    if origin is typing.Literal:
        return args[0]
    if origin in (list, set, tuple):
        return []
    if origin is dict:
        return {}
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return mock_instance(annotation)
    return {str: "", int: 0, float: 0.0, bool: False}.get(annotation, None)


def mock_instance(schema: type[T]) -> T:
    """Smallest valid instance of a pydantic model (required fields filled with empty values)."""
    data: dict[str, Any] = {}
    for name, field in schema.model_fields.items():
        if field.is_required():
            data[name] = _mock_value(field.annotation, name)
    return schema.model_validate(data)


def _mock_json(prompt_name: str, user: str, schema: type[T]) -> T:
    fixture = MOCK_DIR / f"{prompt_name}.json"
    if fixture.exists():
        return schema.model_validate_json(fixture.read_text(encoding="utf-8"))
    instance = mock_instance(schema)
    if prompt_name.startswith("judge_"):
        data = instance.model_dump()
        if "score" in data:
            data["score"] = round(_hash_unit(prompt_name + user), 3)
        if "total" in data and "supported" in data:  # counting judges
            data["total"] = 10
            data["supported"] = min(10, int(_hash_unit(prompt_name + user) * 11))
        if "reason" in data:
            data["reason"] = "[MOCK] deterministic pseudo score"
        instance = schema.model_validate(data)
    return instance


# ---------------------------------------------------------------- public API


def _log(prompt_name: str, result: LLMResult, ms: float) -> None:
    logger.info("llm %s model=%s tokens=%s+%s cost=$%.5f cached=%s mock=%s %.0fms", prompt_name, result.model,
                result.tokens_in, result.tokens_out, result.cost_usd, result.cached, result.mock, ms)


def complete_text(prompt_name: str, system: str, user: str, *, model: str | None = None,
                  temperature: float = 0.0, max_tokens: int = 1500) -> LLMResult:
    settings = get_settings()
    model = model or settings.model
    start = time.perf_counter()
    if settings.mock_llm:
        result = LLMResult(text=f"[MOCK] {prompt_name}", model=model, mock=True)
        _log(prompt_name, result, 0)
        return result
    key = _cache_key(model, system, user, temperature, max_tokens)
    if temperature == 0:
        hit = _get_cache().get(key)
        if hit is not None:
            result = LLMResult(**{**hit, "cached": True})
            _log(prompt_name, result, (time.perf_counter() - start) * 1000)
            return result
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    result = _dial().chat(messages, model, temperature, max_tokens)
    if temperature == 0:
        _get_cache().set(key, {**result.__dict__, "cached": False})
    _log(prompt_name, result, (time.perf_counter() - start) * 1000)
    return result


def _strip_fences(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    match = re.search(r"[\[{].*[\]}]", text, re.S)
    return match.group(0) if match else text


def complete_json(prompt_name: str, system: str, user: str, schema: type[T], *, model: str | None = None,
                  temperature: float = 0.0, max_tokens: int = 4000) -> tuple[T, LLMResult]:
    settings = get_settings()
    if settings.mock_llm:
        result = LLMResult(text="", model=model or settings.model, mock=True)
        _log(prompt_name, result, 0)
        return _mock_json(prompt_name, user, schema), result
    instruction = ("\n\nReturn only raw JSON matching this JSON schema. No markdown fences, no prose.\n"
                   + json.dumps(schema.model_json_schema()))
    result = complete_text(prompt_name, system + instruction, user, model=model,
                           temperature=temperature, max_tokens=max_tokens)
    try:
        return schema.model_validate_json(_strip_fences(result.text)), result
    except ValidationError as first_error:
        repair_user = (f"{user}\n\nYour previous reply was invalid:\n{result.text[:2000]}\n\n"
                       f"Validation error: {str(first_error)[:1500]}\nReturn corrected JSON only.")
        retry = complete_text(prompt_name, system + instruction, repair_user, model=model,
                              temperature=temperature, max_tokens=max_tokens)
        try:
            return schema.model_validate_json(_strip_fences(retry.text)), retry
        except ValidationError as exc:
            raise LLMJsonError(f"{prompt_name}: model did not return valid {schema.__name__}: {exc}") from exc


def complete_chat_json(prompt_name: str, system: str, messages: list[dict[str, str]], schema: type[T], *,
                       model: str | None = None, temperature: float = 0.0, max_tokens: int = 3000) -> tuple[T, LLMResult]:
    """One turn of a multi-turn conversation whose replies are JSON (the investigator's tool loop).

    `messages` alternate user/assistant and end with a user turn. Same retry, cache and mock rules as complete_json.
    """
    settings = get_settings()
    model = model or settings.model
    if settings.mock_llm:
        result = LLMResult(text="", model=model, mock=True)
        _log(prompt_name, result, 0)
        return _mock_json(prompt_name, messages[-1]["content"] if messages else "", schema), result
    instruction = ("\n\nEvery reply is only raw JSON matching this JSON schema. No markdown fences, no prose.\n"
                   + json.dumps(schema.model_json_schema()))

    def ask(turns: list[dict[str, str]]) -> LLMResult:
        start = time.perf_counter()
        key = _cache_key(model, system + instruction, json.dumps(turns, ensure_ascii=False), temperature, max_tokens)
        if temperature == 0:
            hit = _get_cache().get(key)
            if hit is not None:
                result = LLMResult(**{**hit, "cached": True})
                _log(prompt_name, result, (time.perf_counter() - start) * 1000)
                return result
        result = _dial().chat([{"role": "system", "content": system + instruction}, *turns], model, temperature, max_tokens)
        if temperature == 0:
            _get_cache().set(key, {**result.__dict__, "cached": False})
        _log(prompt_name, result, (time.perf_counter() - start) * 1000)
        return result

    result = ask(messages)
    try:
        return schema.model_validate_json(_strip_fences(result.text)), result
    except ValidationError as first_error:
        repair = [*messages, {"role": "assistant", "content": result.text[:2000]},
                  {"role": "user", "content": f"Invalid reply. Validation error: {str(first_error)[:1500]}\nReturn corrected JSON only."}]
        retry = ask(repair)
        try:
            return schema.model_validate_json(_strip_fences(retry.text)), retry
        except ValidationError as exc:
            raise LLMJsonError(f"{prompt_name}: model did not return valid {schema.__name__}: {exc}") from exc
