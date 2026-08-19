"""HTTPS client for OpenAI-compatible chat-completion APIs."""

from __future__ import annotations

import json
import os
import socket
import time
from typing import Any, Dict, List, Optional
from urllib import error, request
from urllib.parse import urlparse

from qryeval_plus.llm.base import (
    LLMProvider,
    LLMProviderError,
    LLMResponse,
    LLMToolCall,
)


_RETRYABLE_HTTP_STATUS = {408, 429, 500, 502, 503, 504}


class OpenAICompatibleProvider(LLMProvider):
    """Call a provider that implements the Chat Completions HTTP contract."""

    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        model: str,
        api_key_env: str,
        api_key_required: bool,
        api_path: str = "/chat/completions",
        timeout_seconds: float = 120.0,
        max_retries: int = 2,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ):
        self.name = _nonempty(name, "provider name")
        self.base_url = _nonempty(base_url, "rag:baseUrl").rstrip("/")
        self.model = _nonempty(model, "rag:model")
        self.api_key_env = _nonempty(api_key_env, "rag:apiKeyEnv")
        self.api_key_required = bool(api_key_required)
        self.api_path = "/" + str(api_path).strip().lstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.extra_body = dict(extra_body or {})

        if not self.base_url.startswith(("https://", "http://")):
            raise ValueError("rag:baseUrl must start with http:// or https://.")
        parsed_url = urlparse(self.base_url)
        if parsed_url.scheme == "http" and parsed_url.hostname not in {
            "127.0.0.1", "localhost", "::1"
        }:
            raise ValueError(
                "Plain HTTP is allowed only for a loopback LLM endpoint."
            )
        if self.timeout_seconds <= 0:
            raise ValueError("rag:timeoutSeconds must be greater than zero.")
        if self.max_retries < 0:
            raise ValueError("rag:maxRetries cannot be negative.")
        if self.max_tokens is not None and int(self.max_tokens) <= 0:
            raise ValueError("rag:maxTokens must be greater than zero.")

    @property
    def endpoint(self) -> str:
        return self.base_url + self.api_path

    def generate(self, messages: List[Dict[str, Any]]) -> LLMResponse:
        response = self.complete(messages)
        if response.content is None or str(response.content).strip() == "":
            raise LLMProviderError(
                "Provider '{}' returned an empty answer.".format(self.name)
            )
        return response

    def complete(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Any = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        api_key = os.environ.get(self.api_key_env, "").strip()
        if self.api_key_required and not api_key:
            raise LLMProviderError(
                "Environment variable '{}' is not set for provider '{}'.".format(
                    self.api_key_env, self.name
                )
            )

        body: Dict[str, Any] = {
            "model": self.model,
            "messages": _validate_messages(messages),
            "stream": False,
        }
        effective_max_tokens = self.max_tokens if max_tokens is None else max_tokens
        if effective_max_tokens is not None:
            body["max_tokens"] = int(effective_max_tokens)
        if self.temperature is not None:
            body["temperature"] = float(self.temperature)
        body.update(self.extra_body)
        if tools:
            body["tools"] = _validate_tools(tools)
        if tool_choice is not None:
            body["tool_choice"] = tool_choice

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "qryeval-plus/0.4",
        }
        if api_key:
            headers["Authorization"] = "Bearer " + api_key

        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        started = time.monotonic()
        response_data = self._post(payload, headers)
        duration = time.monotonic() - started

        try:
            choice = response_data["choices"][0]
            message = choice["message"]
            content = message.get("content")
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMProviderError(
                "Provider '{}' returned an invalid chat-completion response.".format(
                    self.name
                )
            ) from exc

        parsed_tool_calls = []
        for item in message.get("tool_calls") or []:
            try:
                function = item["function"]
                arguments = json.loads(function.get("arguments") or "{}")
                if not isinstance(arguments, dict):
                    raise TypeError("tool arguments must be an object")
                parsed_tool_calls.append(LLMToolCall(
                    id=str(item["id"]),
                    name=str(function["name"]),
                    arguments=arguments,
                ))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise LLMProviderError(
                    "Provider '{}' returned an invalid tool call.".format(self.name)
                ) from exc

        if not parsed_tool_calls and (content is None or str(content).strip() == ""):
            raise LLMProviderError(
                "Provider '{}' returned neither content nor tool calls.".format(
                    self.name
                )
            )

        usage = response_data.get("usage")
        return LLMResponse(
            content="" if content is None else str(content),
            provider=self.name,
            model=str(response_data.get("model") or self.model),
            duration_seconds=duration,
            usage=usage if isinstance(usage, dict) else {},
            request_id=str(response_data.get("id") or ""),
            tool_calls=parsed_tool_calls,
            finish_reason=str(choice.get("finish_reason") or ""),
            reasoning_content=str(message.get("reasoning_content") or ""),
        )

    def _post(self, payload: bytes, headers: Dict[str, str]) -> Dict[str, Any]:
        last_error: Optional[BaseException] = None
        for attempt in range(self.max_retries + 1):
            req = request.Request(
                self.endpoint, data=payload, headers=headers, method="POST"
            )
            try:
                with request.urlopen(req, timeout=self.timeout_seconds) as response:
                    raw = response.read()
                parsed = json.loads(raw.decode("utf-8"))
                if not isinstance(parsed, dict):
                    raise LLMProviderError(
                        "Provider '{}' returned a non-object JSON response.".format(
                            self.name
                        )
                    )
                return parsed
            except error.HTTPError as exc:
                last_error = exc
                if exc.code not in _RETRYABLE_HTTP_STATUS or attempt >= self.max_retries:
                    detail = _http_error_detail(exc)
                    raise LLMProviderError(
                        "Provider '{}' returned HTTP {}{}.".format(
                            self.name, exc.code, detail
                        )
                    ) from exc
            except (error.URLError, socket.timeout, TimeoutError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise LLMProviderError(
                        "Provider '{}' request failed: {}.".format(
                            self.name, _safe_reason(exc)
                        )
                    ) from exc
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise LLMProviderError(
                    "Provider '{}' returned invalid JSON.".format(self.name)
                ) from exc

            time.sleep(min(0.25 * (2 ** attempt), 2.0))

        raise LLMProviderError(
            "Provider '{}' request failed: {}.".format(
                self.name, _safe_reason(last_error)
            )
        )


def _nonempty(value: Any, label: str) -> str:
    result = "" if value is None else str(value).strip()
    if not result:
        raise ValueError("{} cannot be empty.".format(label))
    return result


def _validate_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(messages, list) or not messages:
        raise LLMProviderError("LLM messages must be a non-empty list.")

    normalized = []
    valid_roles = {"system", "user", "assistant", "tool"}
    for message in messages:
        if not isinstance(message, dict):
            raise LLMProviderError("Every LLM message must be an object.")
        role = str(message.get("role", "")).strip().lower()
        content = message.get("content")
        tool_calls = message.get("tool_calls")
        if role not in valid_roles or (content is None and not tool_calls):
            raise LLMProviderError(
                "LLM messages require a supported role and content or tool calls."
            )
        normalized_message: Dict[str, Any] = {
            "role": role,
            "content": None if content is None else str(content),
        }
        if role == "assistant" and tool_calls:
            normalized_message["tool_calls"] = tool_calls
        if role == "tool":
            tool_call_id = str(message.get("tool_call_id", "")).strip()
            if not tool_call_id:
                raise LLMProviderError("Tool messages require tool_call_id.")
            normalized_message["tool_call_id"] = tool_call_id
            if message.get("name"):
                normalized_message["name"] = str(message["name"])
        normalized.append(normalized_message)
    return normalized


def _validate_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(tools, list) or not tools:
        raise LLMProviderError("tools must be a non-empty list.")
    normalized = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise LLMProviderError("Every tool must be a function definition.")
        function = tool.get("function")
        if not isinstance(function, dict) or not str(function.get("name", "")).strip():
            raise LLMProviderError("Every function tool requires a name.")
        normalized.append(tool)
    return normalized


def _http_error_detail(exc: error.HTTPError) -> str:
    try:
        raw = exc.read(2048).decode("utf-8", errors="replace")
        payload = json.loads(raw)
        message = payload.get("error", {}).get("message")
        if message:
            return ": " + str(message).replace("\n", " ")[:300]
    except (AttributeError, TypeError, ValueError):
        pass
    return ""


def _safe_reason(exc: Optional[BaseException]) -> str:
    if exc is None:
        return "unknown transport error"
    reason = getattr(exc, "reason", exc)
    return str(reason).replace("\n", " ")[:300]
