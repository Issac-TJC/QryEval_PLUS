import asyncio
import json
import time

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from qryeval_plus.service.app import create_app
from qryeval_plus.service.queue import InferenceQueue, QueueOverloaded
from qryeval_plus.llm import LLMProviderError


class FakeEngine:
    corpus_version = "fixture-v1"

    def __init__(self, delay=0):
        self.ready = False
        self.delay = delay
        self.responses = {}
        self.calls = 0

    def start(self):
        self.ready = True

    def close(self):
        self.ready = False

    def cache_key(self, question, policy):
        return policy + ":" + question

    def cached(self, question, policy):
        value = self.responses.get(self.cache_key(question, policy))
        return {**value, "cache_hit": True} if value else None

    def answer(self, question, policy, request_id=None):
        self.calls += 1
        time.sleep(self.delay)
        value = {
            "request_id": request_id or "generated", "answer": "SMERSH",
            "citations": [{"doc_id": "d1", "passage": "SMERSH means death to spies."}],
            "policy": policy, "grounding_status": "answered", "cache_hit": False,
            "usage": {"total_tokens": 8},
            "estimated_cost_usd": 0.0,
            "latency_ms": {"queue": 0.0, "retrieval": 1.0, "planner": 1.0, "answer": 1.0, "service_total": 3.0},
            "stop_reason": "finished", "corpus_version": self.corpus_version,
        }
        self.responses[self.cache_key(question, policy)] = value
        return value


def test_service_health_answer_cache_validation_and_metrics(tmp_path):
    config = tmp_path / "service.json"
    config.write_text(json.dumps({"queueCapacity": 2, "requestTimeoutSeconds": 2}))
    engine = FakeEngine()
    app = create_app(str(config), engine=engine)
    with TestClient(app) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").json()["corpus_version"] == "fixture-v1"
        first = client.post("/v1/answer", json={"question": "Which agency?"})
        second = client.post("/v1/answer", json={"question": "Which agency?"})
        assert first.status_code == 200
        assert first.json()["cache_hit"] is False
        assert second.json()["cache_hit"] is True
        assert engine.calls == 1
        assert client.post("/v1/answer", json={"question": "   "}).status_code == 422
        metrics = client.get("/metrics")
        assert metrics.status_code == 200
        assert "qryeval_http_requests_total" in metrics.text
        assert "qryeval_cache_results_total" in metrics.text
        assert "qryeval_stop_reasons_total" in metrics.text


def test_queue_coalesces_duplicate_requests():
    async def scenario():
        engine = FakeEngine(delay=0.02)
        queue = InferenceQueue(engine, capacity=2)
        await queue.start()
        try:
            first, second = await asyncio.gather(
                queue.submit("same", "adaptive_rewrite"),
                queue.submit("same", "adaptive_rewrite"),
            )
            assert first["answer"] == second["answer"]
            assert engine.calls == 1
        finally:
            await queue.close()

    asyncio.run(scenario())


def test_queue_returns_controlled_overload():
    async def scenario():
        engine = FakeEngine(delay=0.05)
        queue = InferenceQueue(engine, capacity=1)
        await queue.start()
        try:
            first = asyncio.create_task(queue.submit("one", "adaptive_rewrite"))
            await asyncio.sleep(0.005)
            second = asyncio.create_task(queue.submit("two", "adaptive_rewrite"))
            await asyncio.sleep(0)
            with pytest.raises(QueueOverloaded):
                await queue.submit("three", "adaptive_rewrite")
            await asyncio.gather(first, second)
        finally:
            await queue.close()

    asyncio.run(scenario())


def test_optional_bearer_auth_is_enforced(tmp_path, monkeypatch):
    config = tmp_path / "service.json"
    config.write_text(json.dumps({
        "queueCapacity": 2, "requestTimeoutSeconds": 2,
        "authTokenEnv": "TEST_QRYEVAL_TOKEN",
    }))
    monkeypatch.setenv("TEST_QRYEVAL_TOKEN", "secret")
    app = create_app(str(config), engine=FakeEngine())
    with TestClient(app) as client:
        assert client.post("/v1/answer", json={"question": "Which agency?"}).status_code == 401
        response = client.post(
            "/v1/answer",
            json={"question": "Which agency?"},
            headers={"Authorization": "Bearer secret"},
        )
        assert response.status_code == 200


def test_service_maps_overload_and_provider_timeout(tmp_path, monkeypatch):
    config = tmp_path / "service.json"
    config.write_text(json.dumps({"queueCapacity": 1, "requestTimeoutSeconds": 2}))
    app = create_app(str(config), engine=FakeEngine())
    with TestClient(app) as client:
        async def overloaded(*args, **kwargs):
            raise QueueOverloaded("full")

        monkeypatch.setattr(app.state.inference_queue, "submit", overloaded)
        assert client.post("/v1/answer", json={"question": "q"}).status_code == 429

        async def timed_out(*args, **kwargs):
            raise LLMProviderError("provider timeout")

        monkeypatch.setattr(app.state.inference_queue, "submit", timed_out)
        assert client.post("/v1/answer", json={"question": "q"}).status_code == 504
        metrics = client.get("/metrics").text
        assert 'qryeval_service_errors_total{type="queue_full"} 1.0' in metrics
        assert 'qryeval_service_errors_total{type="provider_timeout"} 1.0' in metrics


def test_mock_service_uses_versioned_fixture_corpus(tmp_path):
    from qryeval_plus.service.mock import MockAnswerEngine

    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        json.dumps({
            "doc_id": "d1", "title": "SMERSH",
            "body": "SMERSH means death to spies.", "answer": "SMERSH",
        }) + "\n",
        encoding="utf-8",
    )
    config = tmp_path / "mock.json"
    config.write_text(json.dumps({
        "fixtureCorpusPath": str(corpus), "expectedDocuments": 1,
        "responseCachePath": str(tmp_path / "cache.sqlite3"),
        "corpusVersion": "fixture-test", "mockDelaySeconds": 0,
    }))
    engine = MockAnswerEngine(config)
    engine.start()
    try:
        matched = engine.answer("Which agency means death to spies?", "fixed_bm25")
        missing = engine.answer("zzzz-no-overlap", "fixed_bm25")
        assert matched["answer"] == "SMERSH"
        assert matched["citations"][0]["doc_id"] == "d1"
        assert missing["grounding_status"] == "abstained"
        assert missing["stop_reason"] == "empty_retrieval"
    finally:
        engine.close()
