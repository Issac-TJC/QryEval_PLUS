"""Synchronous inference engine owned by the service's single worker."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

from qryeval_plus.agentic.rewrite import RewriteRagAgent
from qryeval_plus.agentic.tools import AgentToolbox
from qryeval_plus.config import ConfigError, load_config, ordered_tasks
from qryeval_plus.grounding import grounding_status
from qryeval_plus.llm import LLMProviderError, create_provider
from qryeval_plus.rag.RagPrompt import RagPrompt
from qryeval_plus.service.cache import ResponseCache


class AnswerEngine:
    """Serve fixed BM25 and the frozen adaptive-rewrite policy."""

    def __init__(self, service_config: str | Path):
        service_path = Path(service_config).expanduser().resolve()
        self.service_path = service_path
        self.settings = json.loads(service_path.read_text(encoding="utf-8"))
        experiment_path = Path(self.settings["experimentConfig"])
        if not experiment_path.is_absolute():
            experiment_path = (service_path.parent / experiment_path).resolve()
        self.experiment_path = experiment_path
        experiment = load_config(experiment_path)
        agent_tasks = [task for _, role, task in ordered_tasks(experiment) if role == "agent"]
        if len(agent_tasks) != 1:
            raise ConfigError("Service experiment must contain exactly one agent task.")
        self.parameters = deepcopy(agent_tasks[0])
        if self.parameters.get("type") != "rewrite_rag" or self.parameters.get("rewrite:policy") != "adaptive":
            raise ConfigError("Service experiment must be the adaptive rewrite_rag configuration.")
        cache_path = Path(self.settings.get("responseCachePath", "../../outputs/service/responses.sqlite3"))
        if not cache_path.is_absolute():
            cache_path = (service_path.parent / cache_path).resolve()
        llm_cache = Path(self.settings.get("llmCachePath", "../../outputs/service/llm.sqlite3"))
        if not llm_cache.is_absolute():
            llm_cache = (service_path.parent / llm_cache).resolve()
        self.parameters["rag:cachePath"] = str(llm_cache)
        self.parameters["rag:cacheNamespace"] = str(self.settings.get("corpusVersion", "clueweb22-v1"))
        self.parameters["rag:inputPricePerMillion"] = float(
            self.settings.get("inputPricePerMillionUsd", 0)
        )
        self.parameters["rag:outputPricePerMillion"] = float(
            self.settings.get("outputPricePerMillionUsd", 0)
        )
        self.parameters["agent:checkpointPath"] = None
        self.parameters["agent:trajectoryPath"] = None
        self.parameters["agent:resume"] = False
        self.index_path = experiment["indexPath"]
        self.corpus_version = str(self.settings.get("corpusVersion", "clueweb22-v1"))
        self.expected_documents = int(self.settings.get("expectedDocuments", 273140))
        self.config_hash = hashlib.sha256(
            json.dumps(self.parameters, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        self.cache = ResponseCache(str(cache_path))
        self.provider = create_provider(self.parameters)
        self.toolbox = AgentToolbox(self.parameters)
        self.adaptive = RewriteRagAgent(
            self.parameters, provider=self.provider, toolbox=self.toolbox
        )
        self.prompt = RagPrompt(self.parameters)
        self.ready = False

    def start(self) -> None:
        from qryeval_plus.core.Idx import Idx

        if not Idx.open(self.index_path):
            raise RuntimeError("Unable to open service index: {}".format(self.index_path))
        documents = int(Idx.getNumDocs())
        if documents != self.expected_documents:
            Idx.close()
            raise RuntimeError(
                "Corpus version mismatch: expected {} documents, found {}.".format(
                    self.expected_documents, documents
                )
            )
        self.ready = True

    def close(self) -> None:
        from qryeval_plus.core.Idx import Idx

        self.ready = False
        if Idx.indexReader is not None:
            Idx.close()
        self.cache.close()
        close = getattr(self.provider, "close", None)
        if close:
            close()

    def cache_key(self, question: str, policy: str) -> str:
        return self.cache.key(
            corpus_version=self.corpus_version,
            config_hash=self.config_hash,
            policy=policy,
            question=question,
        )

    def cached(self, question: str, policy: str) -> Dict[str, Any] | None:
        value = self.cache.get(self.cache_key(question, policy))
        if value:
            value = deepcopy(value)
            value["cache_hit"] = True
            value["usage"] = {}
            value["estimated_cost_usd"] = 0.0
        return value

    def answer(self, question: str, policy: str, request_id: str | None = None) -> Dict[str, Any]:
        if not self.ready:
            raise RuntimeError("Inference engine is not ready.")
        cached = self.cached(question, policy)
        if cached:
            if request_id:
                cached["request_id"] = request_id
            return cached
        request_id = request_id or str(uuid.uuid4())
        started = time.monotonic()
        if policy == "adaptive_rewrite":
            qinfo = {"qstring": question}
            self.adaptive.execute({request_id: qinfo})
            payload = self._from_qinfo(request_id, policy, qinfo)
        elif policy == "fixed_bm25":
            payload = self._fixed_answer(request_id, question)
        else:
            raise ValueError("Unsupported policy: {}".format(policy))
        payload["latency_ms"]["service_total"] = (time.monotonic() - started) * 1000.0
        payload["cache_hit"] = False
        self.cache.put(self.cache_key(question, policy), payload)
        return payload

    def _fixed_answer(self, request_id: str, question: str) -> Dict[str, Any]:
        state = {"qid": request_id, "question": question, "rankings": {}, "ranking_order": []}
        retrieval_started = time.monotonic()
        result = self.toolbox.execute(
            name="search_bm25", arguments={"query": question}, state=state
        )
        retrieval_ms = (time.monotonic() - retrieval_started) * 1000.0
        ranking_id = result["ranking_id"]
        passages, citations = self.toolbox.context_for(state, ranking_id)
        if not state["rankings"][ranking_id]["ranking"] or not any(passages):
            return {
                "request_id": request_id, "answer": "", "citations": [],
                "policy": "fixed_bm25", "grounding_status": "abstained",
                "cache_hit": False, "usage": {},
                "estimated_cost_usd": 0.0,
                "latency_ms": {"queue": 0.0, "retrieval": retrieval_ms, "planner": 0.0, "answer": 0.0},
                "stop_reason": "empty_retrieval", "corpus_version": self.corpus_version,
            }
        answer_started = time.monotonic()
        try:
            response = self.provider.generate(self.prompt.build(question, passages))
        except LLMProviderError:
            raise
        answer_ms = (time.monotonic() - answer_started) * 1000.0
        answer = self.prompt.post_process(response.content)
        return {
            "request_id": request_id, "answer": answer, "citations": citations,
            "policy": "fixed_bm25",
            "grounding_status": grounding_status(answer, passages),
            "cache_hit": False, "usage": response.usage,
            "estimated_cost_usd": response.estimated_cost_usd,
            "latency_ms": {"queue": 0.0, "retrieval": retrieval_ms, "planner": 0.0, "answer": answer_ms},
            "stop_reason": "finished", "corpus_version": self.corpus_version,
        }

    def _from_qinfo(self, request_id, policy, qinfo):
        latency = qinfo.get("latency", {})
        return {
            "request_id": request_id,
            "answer": qinfo.get("answer", ""),
            "citations": qinfo.get("citations", []),
            "policy": policy,
            "grounding_status": qinfo.get("grounding_status", "needs_review"),
            "cache_hit": False,
            "usage": qinfo.get("llm", {}).get("usage", {}),
            "estimated_cost_usd": float(qinfo.get("llm", {}).get("estimated_cost_usd", 0) or 0),
            "latency_ms": {
                "queue": 0.0,
                "retrieval": 1000.0 * float(latency.get("retrieval_seconds", 0)),
                "planner": 1000.0 * float(latency.get("planner_seconds", 0)),
                "answer": 1000.0 * float(latency.get("answer_seconds", 0)),
            },
            "stop_reason": qinfo.get("rewrite", {}).get("stop_reason", "unknown"),
            "corpus_version": self.corpus_version,
        }
