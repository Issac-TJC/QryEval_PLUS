"""Deterministic fixture engine for API and infrastructure load tests."""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from copy import deepcopy
from pathlib import Path

from qryeval_plus.service.cache import ResponseCache


class MockAnswerEngine:
    def __init__(self, service_config):
        config_path = Path(service_config).expanduser().resolve()
        settings = json.loads(config_path.read_text(encoding="utf-8"))
        cache_path = Path(settings.get("responseCachePath", "../../outputs/service/mock.sqlite3"))
        if not cache_path.is_absolute():
            cache_path = (config_path.parent / cache_path).resolve()
        self.cache = ResponseCache(str(cache_path))
        self.corpus_version = str(settings.get("corpusVersion", "fixture-v1"))
        self.config_hash = "mock-" + hashlib.sha256(
            json.dumps(settings, sort_keys=True).encode("utf-8")
        ).hexdigest()
        self.delay = float(settings.get("mockDelaySeconds", 0.01))
        corpus_path = Path(settings["fixtureCorpusPath"])
        if not corpus_path.is_absolute():
            corpus_path = (config_path.parent / corpus_path).resolve()
        self.documents = [
            json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        expected = int(settings.get("expectedDocuments", len(self.documents)))
        if len(self.documents) != expected:
            raise RuntimeError(
                "Fixture corpus mismatch: expected {} documents, found {}.".format(
                    expected, len(self.documents)
                )
            )
        self.ready = False

    def start(self):
        self.ready = True

    def close(self):
        self.ready = False
        self.cache.close()

    def cache_key(self, question, policy):
        return self.cache.key(
            corpus_version=self.corpus_version,
            config_hash=self.config_hash,
            policy=policy,
            question=question,
        )

    def cached(self, question, policy):
        result = self.cache.get(self.cache_key(question, policy))
        if result:
            result = deepcopy(result)
            result["cache_hit"] = True
            result["usage"] = {}
            result["estimated_cost_usd"] = 0.0
        return result

    def answer(self, question, policy, request_id=None):
        cached = self.cached(question, policy)
        if cached:
            return cached
        started = time.monotonic()
        time.sleep(self.delay)
        query_terms = set(re.findall(r"[a-z0-9]+", question.lower()))
        ranked = []
        for document in self.documents:
            text = "{} {}".format(document["title"], document["body"])
            terms = set(re.findall(r"[a-z0-9]+", text.lower()))
            ranked.append((len(query_terms & terms), document))
        score, document = max(ranked, key=lambda item: item[0])
        elapsed = (time.monotonic() - started) * 1000.0
        if score <= 0:
            answer, citations, grounding, stop_reason = "", [], "abstained", "empty_retrieval"
        else:
            answer = document["answer"]
            citations = [{
                "rank": 1, "doc_id": document["doc_id"], "score": float(score),
                "passage": document["body"],
            }]
            grounding, stop_reason = "answered", "mock_fixture"
        result = {
            "request_id": request_id or str(uuid.uuid4()),
            "answer": answer,
            "citations": citations,
            "policy": policy,
            "grounding_status": grounding,
            "cache_hit": False,
            "usage": {"prompt_tokens": 20, "completion_tokens": 2, "total_tokens": 22},
            "estimated_cost_usd": 0.0,
            "latency_ms": {"queue": 0.0, "retrieval": self.delay * 500, "planner": self.delay * 250, "answer": self.delay * 250, "service_total": elapsed},
            "stop_reason": stop_reason,
            "corpus_version": self.corpus_version,
        }
        self.cache.put(self.cache_key(question, policy), result)
        return result
