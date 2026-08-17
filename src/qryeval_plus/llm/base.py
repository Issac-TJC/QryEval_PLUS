"""Shared types for answer-generation providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class LLMProviderError(RuntimeError):
    """Raised when an LLM request cannot produce a usable answer."""


@dataclass(frozen=True)
class LLMToolCall:
    """Normalized function call returned by a chat-completion provider."""

    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass(frozen=True)
class LLMResponse:
    """Normalized response returned by every provider."""

    content: str
    provider: str
    model: str
    duration_seconds: float
    usage: Dict[str, Any] = field(default_factory=dict)
    request_id: str = ""
    tool_calls: List[LLMToolCall] = field(default_factory=list)
    finish_reason: str = ""
    reasoning_content: str = ""


class LLMProvider:
    """Provider interface consumed by the RAG agent."""

    name = "unknown"
    model = "unknown"

    def generate(self, messages: List[Dict[str, Any]]) -> LLMResponse:
        raise NotImplementedError

    def complete(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Any = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """Return text and/or tool calls for one model turn."""
        if tools or tool_choice is not None or max_tokens is not None:
            raise NotImplementedError("This provider does not support tool calling.")
        return self.generate(messages)
