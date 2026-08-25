"""LLM factory helper.

Provides a simple interface to create LLM clients for use in nodes.
Students should use this helper so the lab works with any supported provider.

Usage in nodes:
    from .llm import get_llm
    llm = get_llm()
    response = llm.invoke("Hello")
"""

from __future__ import annotations

import os
import random
import time
from collections.abc import Callable
from functools import lru_cache
from typing import Any, TypeVar

T = TypeVar("T")

#: Free-tier Gemini/OpenAI keys have tight per-minute limits, and this graph
#: issues up to three calls per scenario. Retrying a 429 with exponential
#: backoff keeps a whole scenario run from silently degrading to fallbacks.
_MAX_LLM_RETRIES = 4
_BASE_BACKOFF_SECONDS = 2.0


def _is_rate_limit(exc: Exception) -> bool:
    """True when an exception looks like a provider rate-limit/quota rejection."""
    text = f"{type(exc).__name__} {exc}".lower()
    return any(
        marker in text
        for marker in ("429", "resource_exhausted", "rate limit", "ratelimit", "quota")
    )


def call_with_retry(operation: Callable[[], T]) -> T:
    """Run an LLM call, retrying transient rate-limit errors with backoff.

    Only rate-limit style failures are retried; everything else propagates
    immediately so genuine bugs (bad key, bad schema) surface at once. The
    caller's own try/except still provides the final fallback.
    """
    for attempt in range(_MAX_LLM_RETRIES):
        try:
            return operation()
        except Exception as exc:
            if not _is_rate_limit(exc) or attempt == _MAX_LLM_RETRIES - 1:
                raise
            delay = _BASE_BACKOFF_SECONDS * (2**attempt) + random.uniform(0, 1)
            time.sleep(delay)
    # Unreachable: the final iteration either returns or re-raises above.
    raise RuntimeError("call_with_retry exhausted without a result")


@lru_cache(maxsize=8)
def _cached_llm(model: str | None, temperature: float) -> Any:
    """Cache clients so each node call reuses one connection pool."""
    return _build_llm(model, temperature)


def get_llm(model: str | None = None, temperature: float = 0.0) -> Any:
    """Return a (cached) LLM client from environment configuration."""
    return _cached_llm(model, temperature)


def _build_llm(model: str | None = None, temperature: float = 0.0) -> Any:
    """Create an LLM client from environment configuration.

    Checks for API keys in this order:
    1. GEMINI_API_KEY → ChatGoogleGenerativeAI
    2. OPENAI_API_KEY → ChatOpenAI
    3. ANTHROPIC_API_KEY → ChatAnthropic

    Override model with the `model` parameter or LLM_MODEL env var.
    """
    if os.getenv("GEMINI_API_KEY"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-google-genai") from exc
        return ChatGoogleGenerativeAI(
            model=model or os.getenv("LLM_MODEL", "gemini-2.5-flash"),
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=temperature,
        )

    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-openai") from exc
        return ChatOpenAI(
            model=model or os.getenv("LLM_MODEL", "gpt-4o-mini"),
            temperature=temperature,
        )

    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-anthropic") from exc
        return ChatAnthropic(
            model=model or os.getenv("LLM_MODEL", "claude-sonnet-4-20250514"),
            temperature=temperature,
        )

    raise RuntimeError(
        "No LLM API key found. Set GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY in .env\n"
        "See .env.example for configuration."
    )
