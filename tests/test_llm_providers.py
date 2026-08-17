import json
from urllib import error

import pytest

from qryeval_plus.llm import LLMProviderError, create_provider
from qryeval_plus.rag.RagPrompt import RagPrompt


class FakeResponse:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return self._payload


def test_deepseek_provider_sends_chat_completion(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["authorization"] = req.get_header("Authorization")
        captured["timeout"] = timeout
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return FakeResponse({
            "id": "request-1",
            "model": "deepseek-v4-flash",
            "choices": [{"message": {"content": "V-2 rocket"}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 3},
        })

    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-for-test")
    monkeypatch.setattr(
        "qryeval_plus.llm.openai_compatible.request.urlopen", fake_urlopen
    )
    provider = create_provider({
        "rag:provider": "deepseek",
        "rag:model": "deepseek-v4-flash",
        "rag:thinking": False,
        "rag:maxTokens": 64,
    })

    response = provider.generate([
        {"role": "system", "content": "Use the evidence."},
        {"role": "user", "content": "Question and context"},
    ])

    assert response.content == "V-2 rocket"
    assert response.provider == "deepseek"
    assert response.usage["completion_tokens"] == 3
    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    assert captured["authorization"] == "Bearer secret-for-test"
    assert captured["timeout"] == 120
    assert captured["body"]["stream"] is False
    assert captured["body"]["thinking"] == {"type": "disabled"}
    assert captured["body"]["max_tokens"] == 64
    assert captured["body"]["temperature"] == 0


def test_deepseek_provider_parses_tool_calls(monkeypatch):
    captured = {}

    def fake_urlopen(req, timeout):
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return FakeResponse({
            "id": "request-tool-1",
            "model": "deepseek-v4-flash",
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "search_bm25",
                            "arguments": json.dumps({"query": "Colorado beetle crop"}),
                        },
                    }],
                },
            }],
            "usage": {"total_tokens": 25},
        })

    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-for-test")
    monkeypatch.setattr(
        "qryeval_plus.llm.openai_compatible.request.urlopen", fake_urlopen
    )
    provider = create_provider({"rag:provider": "deepseek"})
    tools = [{
        "type": "function",
        "function": {
            "name": "search_bm25",
            "description": "search",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }]

    response = provider.complete(
        [{"role": "user", "content": "Find evidence"}],
        tools=tools,
        tool_choice="auto",
        max_tokens=256,
    )

    assert response.content == ""
    assert response.finish_reason == "tool_calls"
    assert response.tool_calls[0].name == "search_bm25"
    assert response.tool_calls[0].arguments == {"query": "Colorado beetle crop"}
    assert captured["body"]["tools"] == tools
    assert captured["body"]["tool_choice"] == "auto"
    assert captured["body"]["max_tokens"] == 256


def test_deepseek_provider_requires_environment_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    provider = create_provider({"rag:provider": "deepseek"})

    with pytest.raises(LLMProviderError, match="DEEPSEEK_API_KEY"):
        provider.generate([{"role": "user", "content": "test"}])


def test_provider_retries_transient_transport_error(monkeypatch):
    attempts = []

    def flaky_urlopen(req, timeout):
        attempts.append(req.full_url)
        if len(attempts) == 1:
            raise error.URLError("temporary")
        return FakeResponse({
            "id": "request-after-retry",
            "choices": [{"finish_reason": "stop", "message": {"content": "answer"}}],
        })

    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-for-test")
    monkeypatch.setattr(
        "qryeval_plus.llm.openai_compatible.request.urlopen", flaky_urlopen
    )
    monkeypatch.setattr("qryeval_plus.llm.openai_compatible.time.sleep", lambda _: None)
    response = create_provider({
        "rag:provider": "deepseek", "rag:maxRetries": 1,
    }).generate([{"role": "user", "content": "test"}])

    assert response.content == "answer"
    assert len(attempts) == 2


def test_openai_compatible_provider_supports_local_keyless_api():
    provider = create_provider({
        "rag:provider": "openai-compatible",
        "rag:baseUrl": "http://127.0.0.1:11434/v1",
        "rag:model": "local-model",
        "rag:apiKeyRequired": False,
    })

    assert provider.endpoint == "http://127.0.0.1:11434/v1/chat/completions"
    assert provider.api_key_required is False


def test_openai_compatible_provider_rejects_remote_plain_http():
    with pytest.raises(ValueError, match="loopback"):
        create_provider({
            "rag:provider": "openai-compatible",
            "rag:baseUrl": "http://llm.example/v1",
            "rag:model": "remote-model",
            "rag:apiKeyEnv": "REMOTE_API_KEY",
        })


def test_rag_prompt_contains_only_standard_chat_roles():
    messages = RagPrompt({"rag:prompt": 1}).build(
        "What powered it?", ["It was powered by a V-2 rocket."]
    )

    assert [message["role"] for message in messages] == ["system", "user"]


def test_fixed_rag_benchmark_mode_records_failure_and_retries_on_resume(tmp_path, monkeypatch):
    from qryeval_plus.llm import LLMResponse
    from qryeval_plus.rag.Agent import Agent

    class Encoder:
        def encode_text(self, text):
            return [0.0]

    class FailingProvider:
        name = "mock"
        model = "failure"

        def generate(self, messages):
            raise LLMProviderError("temporary outage")

    monkeypatch.setattr(
        "qryeval_plus.retrieval.DenseEncoder.DenseEncoder.get", lambda _: Encoder()
    )
    parameters = {
        "type": "rag", "agentDepth": 5,
        "rag:dense:modelPath": "unused", "rag:psgLen": 150,
        "rag:psgStride": 140, "rag:psgCnt": 6,
        "rag:provider": "deepseek", "rag:fallback": False,
        "rag:continueOnError": True,
        "agent:checkpointPath": str(tmp_path / "checkpoint.jsonl"),
    }
    first = Agent(parameters, provider=FailingProvider()).execute({
        "q1": {"qstring": "question", "ranking": []},
    })
    assert first["q1"]["answer"] == ""
    assert first["q1"]["llm"]["success"] is False

    class SuccessfulProvider:
        name = "mock"
        model = "success"

        def generate(self, messages):
            return LLMResponse(
                content="answer", provider=self.name, model=self.model,
                duration_seconds=0.0,
            )

    resumed_parameters = {**parameters, "agent:resume": True}
    resumed = Agent(resumed_parameters, provider=SuccessfulProvider()).execute({
        "q1": {"qstring": "question", "ranking": []},
    })
    assert resumed["q1"]["answer"] == "answer"
    assert resumed["q1"]["llm"]["success"] is True
