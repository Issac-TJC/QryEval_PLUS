"""Persistent request cache and hard API budget enforcement."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from qryeval_plus.llm.base import LLMProvider, LLMProviderError, LLMResponse, LLMToolCall


class BudgetExceeded(LLMProviderError):
    """Raised before an external request would exceed a frozen run budget."""


class CachedBudgetProvider(LLMProvider):
    """Wrap a provider with content-addressed SQLite caching and a run ledger."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        cache_path: str | None,
        namespace: str,
        max_requests: int = 2000,
        max_tokens: int = 10_000_000,
        input_price_per_million: float = 0.0,
        output_price_per_million: float = 0.0,
    ):
        if max_requests <= 0 or max_tokens <= 0:
            raise ValueError("LLM request and token budgets must be positive.")
        self._provider = provider
        self.name = provider.name
        self.model = provider.model
        self.namespace = str(namespace or "default")
        self.max_requests = int(max_requests)
        self.max_tokens = int(max_tokens)
        self.input_price_per_million = float(input_price_per_million)
        self.output_price_per_million = float(output_price_per_million)
        self._lock = threading.RLock()
        self._cache_path = cache_path or ":memory:"
        if self._cache_path != ":memory:":
            Path(self._cache_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        try:
            self._db = sqlite3.connect(self._cache_path, check_same_thread=False)
            if self._cache_path != ":memory:":
                self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS llm_cache ("
                "cache_key TEXT PRIMARY KEY, response_json TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP)"
            )
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS budget_events ("
                "namespace TEXT NOT NULL, cache_key TEXT NOT NULL, input_tokens INTEGER NOT NULL, "
                "output_tokens INTEGER NOT NULL, total_tokens INTEGER NOT NULL, cost_usd REAL NOT NULL, "
                "PRIMARY KEY(namespace, cache_key))"
            )
            self._db.commit()
        except sqlite3.DatabaseError as exc:
            raise LLMProviderError("Unable to open LLM cache '{}': {}".format(self._cache_path, exc)) from exc

    def __getattr__(self, name: str):
        return getattr(self._provider, name)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def generate(self, messages: List[Dict[str, Any]]) -> LLMResponse:
        response = self.complete(messages)
        if not str(response.content or "").strip():
            raise LLMProviderError("Provider '{}' returned an empty answer.".format(self.name))
        return response

    def complete(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Any = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        payload = {
            "namespace": self.namespace,
            "provider": self.name,
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "max_tokens": max_tokens,
            "provider_settings": {
                "endpoint": getattr(self._provider, "endpoint", None),
                "default_max_tokens": getattr(self._provider, "max_tokens", None),
                "temperature": getattr(self._provider, "temperature", None),
                "extra_body": getattr(self._provider, "extra_body", None),
            },
        }
        cache_key = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        with self._lock:
            row = self._db.execute(
                "SELECT response_json FROM llm_cache WHERE cache_key = ?", (cache_key,)
            ).fetchone()
            if row:
                return _decode_response(row[0], cache_hit=True)
            budget = self.budget_snapshot()
            if budget["requests"] >= self.max_requests:
                raise BudgetExceeded(
                    "LLM request budget exhausted ({}/{}).".format(
                        budget["requests"], self.max_requests
                    )
                )
            if budget["tokens"] >= self.max_tokens:
                raise BudgetExceeded(
                    "LLM token budget exhausted ({}/{}).".format(
                        budget["tokens"], self.max_tokens
                    )
                )

        response = self._provider.complete(
            messages, tools=tools, tool_choice=tool_choice, max_tokens=max_tokens
        )
        input_tokens = int(response.usage.get("prompt_tokens", 0) or 0)
        output_tokens = int(response.usage.get("completion_tokens", 0) or 0)
        total_tokens = int(
            response.usage.get("total_tokens", input_tokens + output_tokens) or 0
        )
        cost = (
            input_tokens * self.input_price_per_million
            + output_tokens * self.output_price_per_million
        ) / 1_000_000.0
        stored = LLMResponse(
            **{
                **response.__dict__,
                "cache_hit": False,
                "estimated_cost_usd": cost,
            }
        )
        with self._lock:
            self._db.execute(
                "INSERT OR IGNORE INTO llm_cache(cache_key, response_json) VALUES (?, ?)",
                (cache_key, _encode_response(stored)),
            )
            self._db.execute(
                "INSERT OR IGNORE INTO budget_events(namespace, cache_key, input_tokens, output_tokens, total_tokens, cost_usd) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (self.namespace, cache_key, input_tokens, output_tokens, total_tokens, cost),
            )
            self._db.commit()
        return stored

    def budget_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            row = self._db.execute(
                "SELECT COUNT(*), COALESCE(SUM(total_tokens), 0), COALESCE(SUM(input_tokens), 0), "
                "COALESCE(SUM(output_tokens), 0), COALESCE(SUM(cost_usd), 0) "
                "FROM budget_events WHERE namespace = ?",
                (self.namespace,),
            ).fetchone()
        return {
            "requests": int(row[0]),
            "tokens": int(row[1]),
            "input_tokens": int(row[2]),
            "output_tokens": int(row[3]),
            "estimated_cost_usd": float(row[4]),
            "max_requests": self.max_requests,
            "max_tokens": self.max_tokens,
        }


def _encode_response(response: LLMResponse) -> str:
    payload = {
        **response.__dict__,
        "tool_calls": [call.__dict__ for call in response.tool_calls],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _decode_response(payload: str, cache_hit: bool) -> LLMResponse:
    try:
        data = json.loads(payload)
        data["tool_calls"] = [LLMToolCall(**item) for item in data.get("tool_calls", [])]
        data["cache_hit"] = cache_hit
        if cache_hit:
            data["estimated_cost_usd"] = 0.0
        return LLMResponse(**data)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise LLMProviderError("Cached LLM response is corrupt.") from exc
