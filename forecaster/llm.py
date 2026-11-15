"""Language-model clients.

OpenRouterLLM calls real models through OpenRouter's OpenAI-compatible API
using plain HTTP (no SDK needed). ScriptedLLM replays fixed responses so the
control logic, guardrails and evaluation can run offline and repeatably.

Both return parsed JSON. Model output is never trusted as a command: callers
validate it against the shared-state contract before using it.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class LLMError(RuntimeError):
    pass


def parse_json(text: str) -> Dict[str, Any]:
    """Extract the first JSON object from model text (handles ```json fences)."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise LLMError("model returned no JSON object")
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise LLMError(f"model returned malformed JSON: {exc}") from exc


@dataclass
class CallRecord:
    tag: str
    model: str
    seconds: float
    ok: bool


@dataclass
class BaseLLM:
    calls: List[CallRecord] = field(default_factory=list)

    def complete_json(self, tag: str, model: str, system: str, user: str) -> Dict[str, Any]:
        raise NotImplementedError

    @property
    def n_calls(self) -> int:
        return len(self.calls)


@dataclass
class OpenRouterLLM(BaseLLM):
    api_key: str = ""
    temperature: float = 0.2
    timeout: int = 90

    def complete_json(self, tag: str, model: str, system: str, user: str) -> Dict[str, Any]:
        if not self.api_key:
            raise LLMError("OPENROUTER_API_KEY is not set")
        body = {
            "model": model,
            "temperature": self.temperature,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": "Stock Opportunity Forecaster",
        }
        started = time.time()
        try:
            resp = requests.post(OPENROUTER_URL, json=body, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"] or ""
            result = parse_json(content)
            self.calls.append(CallRecord(tag, model, time.time() - started, True))
            return result
        except (requests.RequestException, KeyError, IndexError, LLMError) as exc:
            self.calls.append(CallRecord(tag, model, time.time() - started, False))
            raise LLMError(f"{tag}: {exc}") from exc


@dataclass
class ScriptedLLM(BaseLLM):
    """Returns pre-written responses keyed by call tag.

    A response may be a dict (returned as-is), the string "MALFORMED" (raises,
    to test retries), or a list of responses consumed in order.
    """

    script: Dict[str, Any] = field(default_factory=dict)

    def complete_json(self, tag: str, model: str, system: str, user: str) -> Dict[str, Any]:
        value = self.script.get(tag)
        if isinstance(value, list):
            value = value.pop(0) if value else None
        ok = isinstance(value, dict)
        self.calls.append(CallRecord(tag, model, 0.0, ok))
        if value is None:
            raise LLMError(f"no scripted response for '{tag}'")
        if not ok:
            raise LLMError(f"{tag}: malformed scripted response")
        return json.loads(json.dumps(value))  # defensive copy
