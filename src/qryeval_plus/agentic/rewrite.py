"""One-decision BM25 rewrite policy used by the controlled ablation."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List

from qryeval_plus.agentic.tools import AgentToolbox
from qryeval_plus.grounding import grounding_status, normalize_text
from qryeval_plus.llm import BudgetExceeded, LLMProviderError, create_provider
from qryeval_plus.rag.RagPrompt import RagPrompt


_REWRITE_TOOL = {
    "type": "function",
    "function": {
        "name": "rewrite_query",
        "description": "Rewrite the question into a concise BM25 keyword query when the initial evidence is insufficient.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "minLength": 1, "maxLength": 300}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

_FINISH_TOOL = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": "Keep the initial BM25 ranking because its evidence is sufficient.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    },
}


class RewriteRagAgent:
    """Run one initial retrieval, one controller decision, and at most one rewrite."""

    def __init__(self, parameters: Dict[str, Any], provider=None, toolbox=None):
        if str(parameters.get("type", "")).lower() != "rewrite_rag":
            raise ValueError("RewriteRagAgent requires type 'rewrite_rag'.")
        self.parameters = parameters
        self.policy = str(parameters.get("rewrite:policy", "")).lower()
        if self.policy not in {"always", "adaptive"}:
            raise ValueError("rewrite:policy must be 'always' or 'adaptive'.")
        self.provider = provider if provider is not None else create_provider(parameters)
        self.toolbox = toolbox if toolbox is not None else AgentToolbox(parameters)
        self.prompt_builder = RagPrompt(parameters)
        self.controller_max_tokens = int(parameters.get("rewrite:maxTokens", 96))
        self.grounding_threshold = float(parameters.get("rag:groundingThreshold", 0.8))
        self.checkpoint_path = parameters.get("agent:checkpointPath")
        self.trajectory_path = parameters.get("agent:trajectoryPath")
        self.resume = _as_bool(parameters.get("agent:resume", False))
        self.completed = self._load_checkpoints() if self.resume else {}
        if not self.resume:
            for value in (self.checkpoint_path, self.trajectory_path):
                if value:
                    path = Path(value)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("", encoding="utf-8")

    def execute(self, batch: Dict[str, Dict[str, Any]]):
        for qid, qinfo in batch.items():
            if str(qid) in self.completed:
                qinfo.update(deepcopy(self.completed[str(qid)]))
                continue
            self._execute_one(str(qid), qinfo)
            self._append(self.checkpoint_path, {"qid": str(qid), "qinfo": qinfo})
            self._append(self.trajectory_path, self._trajectory_record(str(qid), qinfo))
        return batch

    def _execute_one(self, qid: str, qinfo: Dict[str, Any]) -> None:
        question = str(qinfo.get("qstring", "")).strip()
        if not question:
            raise ValueError("Question cannot be empty.")
        state = {
            "qid": qid,
            "question": question,
            "rankings": {},
            "ranking_order": [],
        }
        started = time.monotonic()
        retrieval_started = time.monotonic()
        initial = self.toolbox.execute(
            name="search_bm25", arguments={"query": question}, state=state
        )
        retrieval_seconds = time.monotonic() - retrieval_started
        initial_id = initial["ranking_id"]
        initial_record = state["rankings"][initial_id]
        evidence = initial.get("evidence", [])
        if not initial_record["ranking"]:
            qinfo.update({
                "ranking": [], "answer": "", "citations": [],
                "grounding_status": "abstained",
                "rewrite": {
                    "policy": self.policy, "decision": "abstain", "rewritten_query": None,
                    "retrieval_calls": 1, "stop_reason": "empty_retrieval",
                    "features": _features(question, None, initial_record["ranking"], evidence),
                },
                "llm": {"provider": self.provider.name, "model": self.provider.model, "success": True, "calls": 0, "usage": {}},
                "latency": {"retrieval_seconds": retrieval_seconds, "planner_seconds": 0.0, "answer_seconds": 0.0, "total_seconds": time.monotonic() - started},
            })
            return

        controller_messages = _controller_prompt(question, evidence)
        tools = [_REWRITE_TOOL] + ([_FINISH_TOOL] if self.policy == "adaptive" else [])
        planner_started = time.monotonic()
        try:
            controller = self.provider.complete(
                controller_messages,
                tools=tools,
                tool_choice="required",
                max_tokens=self.controller_max_tokens,
            )
        except BudgetExceeded:
            raise
        except (LLMProviderError, NotImplementedError) as exc:
            raise RuntimeError("Rewrite controller failed for '{}': {}".format(qid, exc)) from exc
        planner_seconds = time.monotonic() - planner_started
        if len(controller.tool_calls) != 1:
            raise RuntimeError("Rewrite controller must return exactly one tool call.")
        call = controller.tool_calls[0]
        selected_id = initial_id
        rewritten_query = None
        decision = call.name
        if call.name == "rewrite_query":
            rewritten_query = str(call.arguments.get("query", "")).strip()
            if not rewritten_query or len(rewritten_query) > 300:
                raise RuntimeError("Rewrite controller returned an invalid query.")
            retrieval_started = time.monotonic()
            rewritten = self.toolbox.execute(
                name="search_bm25", arguments={"query": rewritten_query}, state=state
            )
            retrieval_seconds += time.monotonic() - retrieval_started
            selected_id = rewritten["ranking_id"]
        elif call.name == "finish" and self.policy == "adaptive":
            decision = "finish"
        else:
            raise RuntimeError("Rewrite controller selected disallowed action '{}'.".format(call.name))

        selected = state["rankings"][selected_id]
        passages, citations = self.toolbox.context_for(state, selected_id)
        if not selected["ranking"] or not any(str(item).strip() for item in passages):
            answer = ""
            answer_seconds = 0.0
            answer_response = None
            stop_reason = "empty_retrieval"
        else:
            answer_started = time.monotonic()
            try:
                answer_response = self.provider.generate(
                    self.prompt_builder.build(question, passages)
                )
            except BudgetExceeded:
                raise
            except LLMProviderError as exc:
                raise RuntimeError("Answer generation failed for '{}': {}".format(qid, exc)) from exc
            answer_seconds = time.monotonic() - answer_started
            answer = self.prompt_builder.post_process(answer_response.content)
            if not answer:
                raise RuntimeError("Answer generation returned an empty normalized answer.")
            stop_reason = "rewritten" if rewritten_query else "initial_sufficient"

        usage = _merge_usage(
            controller.usage,
            answer_response.usage if answer_response is not None else {},
        )
        cache_hits = int(bool(controller.cache_hit)) + int(
            bool(answer_response and answer_response.cache_hit)
        )
        qinfo.update({
            "ranking": list(selected["ranking"]),
            "answer": answer,
            "citations": citations,
            "grounding_status": grounding_status(answer, passages, self.grounding_threshold),
            "rewrite": {
                "policy": self.policy,
                "decision": decision,
                "rewritten_query": rewritten_query,
                "retrieval_calls": 2 if rewritten_query else 1,
                "stop_reason": stop_reason,
                "initial_ranking_id": initial_id,
                "selected_ranking_id": selected_id,
                "features": _features(question, rewritten_query, initial_record["ranking"], evidence),
            },
            "agent": {
                "type": "rewrite_rag", "turns": 1, "tool_calls": 1,
                "retrieval_calls": 2 if rewritten_query else 1,
                "rerank_calls": 0, "fusion_calls": 0, "tool_errors": 0,
                "budget_exhaustions": 0,
                "tool_usage": {decision: 1}, "stop_reason": stop_reason, "errors": [],
            },
            "llm": {
                "provider": self.provider.name, "model": self.provider.model,
                "success": True, "calls": 1 + int(answer_response is not None),
                "cache_hits": cache_hits, "duration_seconds": planner_seconds + answer_seconds,
                "usage": usage,
                "estimated_cost_usd": controller.estimated_cost_usd + (
                    answer_response.estimated_cost_usd if answer_response else 0.0
                ),
            },
            "latency": {
                "retrieval_seconds": retrieval_seconds,
                "planner_seconds": planner_seconds,
                "answer_seconds": answer_seconds,
                "total_seconds": time.monotonic() - started,
            },
        })

    def _load_checkpoints(self):
        if not self.checkpoint_path or not Path(self.checkpoint_path).is_file():
            return {}
        completed = {}
        with Path(self.checkpoint_path).open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                    if row["qinfo"].get("llm", {}).get("success") is not False:
                        completed[str(row["qid"])] = row["qinfo"]
                except (KeyError, TypeError, json.JSONDecodeError):
                    continue
        return completed

    @staticmethod
    def _append(path_value, payload):
        if not path_value:
            return
        path = Path(path_value)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    @staticmethod
    def _trajectory_record(qid, qinfo):
        return {
            "qid": qid,
            "trajectory": [
                {"event": "initial_bm25"},
                {"event": "rewrite_decision", **qinfo.get("rewrite", {})},
                {"event": "answer", "grounding_status": qinfo.get("grounding_status")},
            ],
            "llm": qinfo.get("llm", {}),
            "latency": qinfo.get("latency", {}),
        }


def _controller_prompt(question: str, evidence: List[Dict[str, Any]]):
    return [
        {
            "role": "system",
            "content": (
                "You control one cost-bounded BM25 retrieval step. Inspect the initial evidence. "
                "Choose finish when it directly supports a short answer; otherwise call rewrite_query "
                "with a concise keyword query. Make exactly one tool call."
            ),
        },
        {
            "role": "user",
            "content": "Question: {}\nInitial evidence:\n{}".format(
                question, json.dumps(evidence, ensure_ascii=False)
            ),
        },
    ]


def _features(question, rewritten_query, ranking, evidence):
    scores = [float(item[0]) for item in ranking[:2]]
    margin = scores[0] - scores[1] if len(scores) > 1 else (scores[0] if scores else 0.0)
    q_tokens = set(normalize_text(question).split())
    evidence_tokens = set(
        normalize_text(" ".join(str(row.get("passage", "")) for row in evidence)).split()
    )
    coverage = len(q_tokens & evidence_tokens) / len(q_tokens) if q_tokens else 0.0
    return {
        "query_tokens": len(q_tokens),
        "bm25_top_score": scores[0] if scores else None,
        "bm25_score_margin": margin if scores else None,
        "query_term_coverage": coverage,
        "rewrite_similarity": (
            SequenceMatcher(None, normalize_text(question), normalize_text(rewritten_query)).ratio()
            if rewritten_query else None
        ),
    }


def _merge_usage(*items):
    result: Dict[str, int] = {}
    for item in items:
        for key, value in (item or {}).items():
            if isinstance(value, (int, float)):
                result[key] = result.get(key, 0) + int(value)
    return result


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)
