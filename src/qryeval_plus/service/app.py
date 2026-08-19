"""FastAPI surface, metrics, authentication, and error contracts."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from prometheus_client.exposition import CONTENT_TYPE_LATEST

from qryeval_plus.llm import BudgetExceeded, LLMProviderError
from qryeval_plus.service.engine import AnswerEngine
from qryeval_plus.service.queue import InferenceQueue, QueueOverloaded


LOGGER = logging.getLogger("qryeval.service")


class AnswerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=1000)
    request_id: Optional[str] = Field(default=None, min_length=1, max_length=128)
    policy: Literal["fixed_bm25", "adaptive_rewrite"] = "adaptive_rewrite"

    @field_validator("question")
    @classmethod
    def clean_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question cannot be blank")
        if any(ord(char) < 32 and char not in "\t\n\r" for char in value):
            raise ValueError("question contains unsupported control characters")
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("question contains invalid Unicode") from exc
        return value


class AnswerResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    request_id: str
    answer: str
    citations: list[Dict[str, Any]]
    policy: str
    grounding_status: Literal["answered", "needs_review", "abstained"]
    cache_hit: bool
    usage: Dict[str, Any]
    estimated_cost_usd: float = 0.0
    latency_ms: Dict[str, float]
    stop_reason: str
    corpus_version: str


def create_app(config_path: str, *, engine=None) -> FastAPI:
    path = Path(config_path).expanduser().resolve()
    settings = json.loads(path.read_text(encoding="utf-8"))
    if engine is not None:
        inference = engine
    elif settings.get("mock"):
        from qryeval_plus.service.mock import MockAnswerEngine
        inference = MockAnswerEngine(path)
    else:
        inference = AnswerEngine(path)
    queue = InferenceQueue(inference, capacity=int(settings.get("queueCapacity", 32)))
    timeout = float(settings.get("requestTimeoutSeconds", 180))
    registry = CollectorRegistry()
    request_count = Counter(
        "qryeval_http_requests_total", "HTTP requests", ["path", "status"], registry=registry
    )
    request_latency = Histogram(
        "qryeval_http_request_duration_seconds", "End-to-end HTTP latency", ["path"], registry=registry
    )
    inference_latency = Histogram(
        "qryeval_inference_duration_seconds", "Inference latency", ["policy", "stage"], registry=registry
    )
    cache_count = Counter(
        "qryeval_cache_results_total", "Response cache outcomes", ["result"], registry=registry
    )
    grounding_count = Counter(
        "qryeval_grounding_results_total", "Grounding classifications", ["status"], registry=registry
    )
    token_count = Counter(
        "qryeval_llm_tokens_total", "Reported LLM tokens", ["kind"], registry=registry
    )
    cost_count = Counter(
        "qryeval_llm_estimated_cost_usd_total", "Estimated provider cost in USD", registry=registry
    )
    error_count = Counter(
        "qryeval_service_errors_total", "Controlled service failures", ["type"], registry=registry
    )
    stop_count = Counter(
        "qryeval_stop_reasons_total", "Inference stop reasons", ["reason"], registry=registry
    )
    queue_depth = Gauge(
        "qryeval_inference_queue_depth", "Queued inference requests", registry=registry
    )

    @asynccontextmanager
    async def lifespan(_app):
        await queue.start()
        try:
            yield
        finally:
            await queue.close()

    app = FastAPI(
        title="QryEval_PLUS API",
        version="0.4.0",
        docs_url="/docs",
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.inference_queue = queue
    app.state.engine = inference

    async def authenticate(authorization: Optional[str] = Header(default=None)):
        env_name = str(settings.get("authTokenEnv", "")).strip()
        if not env_name:
            return
        expected = os.environ.get(env_name, "")
        if not expected:
            raise HTTPException(status_code=503, detail="Service authentication is not configured.")
        if authorization != "Bearer " + expected:
            raise HTTPException(status_code=401, detail="Invalid bearer token.")

    @app.middleware("http")
    async def observe(request: Request, call_next):
        started = time.monotonic()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            route = request.url.path
            request_count.labels(path=route, status=str(status)).inc()
            request_latency.labels(path=route).observe(time.monotonic() - started)

    @app.get("/health/live")
    async def live():
        return {"status": "live"}

    @app.get("/health/ready")
    async def ready():
        if not inference.ready:
            raise HTTPException(status_code=503, detail="Inference engine is not ready.")
        return {"status": "ready", "corpus_version": inference.corpus_version}

    @app.get("/metrics")
    async def metrics():
        return Response(content=generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    @app.post("/v1/answer", response_model=AnswerResponse, dependencies=[Depends(authenticate)])
    async def answer(payload: AnswerRequest):
        question_hash = hashlib.sha256(payload.question.encode("utf-8")).hexdigest()[:12]
        queued_at = time.monotonic()
        queue_depth.set(queue.queue.qsize())
        try:
            result = await asyncio.wait_for(
                queue.submit(payload.question, payload.policy, payload.request_id),
                timeout=timeout,
            )
        except QueueOverloaded as exc:
            error_count.labels(type="queue_full").inc()
            _log("overloaded", payload.request_id, question_hash, error=str(exc))
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except BudgetExceeded as exc:
            error_count.labels(type="budget_exhausted").inc()
            _log("budget_exhausted", payload.request_id, question_hash, error=str(exc))
            raise HTTPException(status_code=429, detail=str(exc)) from exc
        except asyncio.TimeoutError as exc:
            error_count.labels(type="request_timeout").inc()
            _log("timeout", payload.request_id, question_hash, error="request timeout")
            raise HTTPException(status_code=504, detail="Inference request timed out.") from exc
        except LLMProviderError as exc:
            status = 504 if "timed out" in str(exc).lower() or "timeout" in str(exc).lower() else 503
            error_count.labels(type="provider_timeout" if status == 504 else "provider_error").inc()
            _log("provider_error", payload.request_id, question_hash, error=str(exc))
            raise HTTPException(status_code=status, detail="Upstream model unavailable.") from exc
        except RuntimeError as exc:
            lowered = str(exc).lower()
            status = 504 if "timeout" in lowered else 503
            error_count.labels(type="inference_timeout" if status == 504 else "inference_error").inc()
            _log("inference_error", payload.request_id, question_hash, error=str(exc))
            raise HTTPException(status_code=status, detail="Inference unavailable.") from exc
        finally:
            queue_depth.set(queue.queue.qsize())

        result["latency_ms"]["http_wait"] = (time.monotonic() - queued_at) * 1000.0
        cache_count.labels(result="hit" if result.get("cache_hit") else "miss").inc()
        grounding_count.labels(status=result.get("grounding_status", "unknown")).inc()
        usage = result.get("usage", {})
        for source, label in (("prompt_tokens", "input"), ("completion_tokens", "output"), ("total_tokens", "total")):
            token_count.labels(kind=label).inc(float(usage.get(source, 0) or 0))
        cost_count.inc(float(result.get("estimated_cost_usd", 0) or 0))
        stop_count.labels(reason=result.get("stop_reason", "unknown")).inc()
        for stage, value in result.get("latency_ms", {}).items():
            inference_latency.labels(policy=payload.policy, stage=stage).observe(float(value) / 1000.0)
        _log(
            "answer_complete", result.get("request_id"), question_hash,
            policy=payload.policy, cache_hit=result.get("cache_hit"),
            grounding_status=result.get("grounding_status"),
        )
        return result

    return app


def _log(event: str, request_id: str | None, question_hash: str, **fields):
    LOGGER.info(json.dumps({
        "event": event,
        "request_id": request_id,
        "question_hash": question_hash,
        **fields,
    }, ensure_ascii=True, sort_keys=True))
