"""Minimal OpenRouter client. The API key is read from the environment only."""
from __future__ import annotations

import os
import re
import time

import requests

API_URL = "https://openrouter.ai/api/v1/chat/completions"
_MIN_MAX_TOKENS = 16          # gpt-4.1 returns HTTP 400 below this


def _api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError('OPENROUTER_API_KEY is not set. Run: export OPENROUTER_API_KEY="..."')
    return key


def ask_llm(user: str, model: str = "openai/gpt-4.1", system: str = "", temperature: float = 1.0,
            max_tokens: int = 600, retries: int = 4, timeout: int = 120) -> str | None:
    """One chat completion, or None if every attempt fails."""
    messages = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": user}]
    payload = {"model": model, "messages": messages, "temperature": temperature,
               "max_tokens": max(max_tokens, _MIN_MAX_TOKENS)}
    headers = {"Authorization": "Bearer " + _api_key(), "Content-Type": "application/json"}
    for attempt in range(retries):
        try:
            data = requests.post(API_URL, headers=headers, json=payload, timeout=timeout).json()
            if "choices" in data:
                return data["choices"][0]["message"]["content"]
        except Exception:
            pass
        time.sleep(1.5 * (attempt + 1))
    return None


def ask_llm_int(user: str, low: int = 1, high: int = 7, **kw) -> int | None:
    """Ask for a single integer and return it when it falls inside [low, high]."""
    text = ask_llm(user, **kw)
    if text is None:
        return None
    m = re.search(r"-?\d+", text)
    if not m:
        return None
    v = int(m.group())
    return v if low <= v <= high else None
