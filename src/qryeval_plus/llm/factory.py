"""Build LLM providers from RAG task configuration."""

from __future__ import annotations

from typing import Any, Dict

from qryeval_plus.llm.openai_compatible import OpenAICompatibleProvider


SUPPORTED_PROVIDERS = {"deepseek", "openai-compatible", "openai_compatible"}


def create_provider(parameters: Dict[str, Any]) -> OpenAICompatibleProvider:
    provider = str(parameters.get("rag:provider", "")).strip().lower()
    if not provider:
        raise ValueError("Missing parameter 'rag:provider'.")
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            "Unsupported rag:provider '{}'. Expected deepseek or "
            "openai-compatible.".format(provider)
        )

    if provider == "deepseek":
        return OpenAICompatibleProvider(
            name="deepseek",
            base_url=parameters.get("rag:baseUrl", "https://api.deepseek.com"),
            model=parameters.get("rag:model", "deepseek-v4-flash"),
            api_key_env=parameters.get("rag:apiKeyEnv", "DEEPSEEK_API_KEY"),
            api_key_required=True,
            api_path=parameters.get("rag:apiPath", "/chat/completions"),
            timeout_seconds=parameters.get("rag:timeoutSeconds", 120),
            max_retries=parameters.get("rag:maxRetries", 2),
            max_tokens=parameters.get("rag:maxTokens"),
            temperature=parameters.get("rag:temperature", 0),
            extra_body={
                "thinking": {
                    "type": "enabled" if _as_bool(
                        parameters.get("rag:thinking", False), "rag:thinking"
                    ) else "disabled"
                }
            },
        )

    return OpenAICompatibleProvider(
        name="openai-compatible",
        base_url=parameters.get("rag:baseUrl"),
        model=parameters.get("rag:model"),
        api_key_env=parameters.get("rag:apiKeyEnv", "OPENAI_API_KEY"),
        api_key_required=_as_bool(
            parameters.get("rag:apiKeyRequired", True), "rag:apiKeyRequired"
        ),
        api_path=parameters.get("rag:apiPath", "/chat/completions"),
        timeout_seconds=parameters.get("rag:timeoutSeconds", 120),
        max_retries=parameters.get("rag:maxRetries", 2),
        max_tokens=parameters.get("rag:maxTokens"),
        temperature=parameters.get("rag:temperature"),
    )


def provider_summary(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Return non-secret provider settings for validation and doctor output."""
    instance = create_provider(parameters)
    return {
        "provider": instance.name,
        "model": instance.model,
        "base_url": instance.base_url,
        "api_key_env": instance.api_key_env,
        "api_key_required": instance.api_key_required,
    }


def _as_bool(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "on"}:
            return True
        if normalized in {"false", "no", "0", "off"}:
            return False
    raise ValueError("{} must be a boolean.".format(label))
