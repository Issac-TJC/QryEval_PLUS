"""LangGraph implementation of a bounded single-agent retrieval loop."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

from pydantic import ValidationError

from qryeval_plus.agentic.state import AgentState, validated
from qryeval_plus.agentic.tools import AgentToolbox, TOOL_SCHEMAS, tool_definitions
from qryeval_plus.llm import LLMProviderError, create_provider
from qryeval_plus.rag.RagPrompt import RagPrompt


DEFAULT_BM25_TOOLS = ["search_bm25", "finish_research"]
DEFAULT_FULL_TOOLS = [
    "search_bm25",
    "search_dense",
    "rerank",
    "fuse_rankings",
    "finish_research",
]


class ToolBudgetExceeded(ValueError):
    """A valid tool request that cannot run within the frozen budget."""


class AgenticRagAgent:
    """Run one independent LangGraph research loop for each query."""

    def __init__(self, parameters: Dict[str, Any], provider=None, toolbox=None):
        if str(parameters.get("type", "")).strip().lower() != "agentic_rag":
            raise ValueError("AgenticRagAgent requires type 'agentic_rag'.")
        self.parameters = parameters
        self.provider = provider if provider is not None else create_provider(parameters)
        self.toolbox = toolbox if toolbox is not None else AgentToolbox(parameters)
        self.max_turns = _positive_int(parameters, "agent:maxTurns", 6)
        self.max_retrieval_calls = _positive_int(
            parameters, "agent:maxRetrievalCalls", 3
        )
        self.max_rerank_calls = _nonnegative_int(
            parameters, "agent:maxRerankCalls", 1
        )
        self.max_fusion_calls = _nonnegative_int(
            parameters, "agent:maxFusionCalls", 1
        )
        self.planner_max_tokens = _positive_int(
            parameters, "agent:plannerMaxTokens", 256
        )
        self.max_planner_retries = _nonnegative_int(
            parameters, "agent:maxPlannerRetries", 1
        )
        allowed = parameters.get("agent:allowedTools", DEFAULT_FULL_TOOLS)
        if not isinstance(allowed, list) or not allowed:
            raise ValueError("agent:allowedTools must be a non-empty list.")
        self.allowed_tools = [str(item) for item in allowed]
        unknown = sorted(set(self.allowed_tools) - set(TOOL_SCHEMAS))
        if unknown:
            raise ValueError("Unknown agent tools: {}".format(", ".join(unknown)))
        if "finish_research" not in self.allowed_tools:
            raise ValueError("agent:allowedTools must include finish_research.")
        self.tools = tool_definitions(self.allowed_tools)
        self.prompt_builder = RagPrompt({"rag:prompt": 1})
        self.trajectory_path = parameters.get("agent:trajectoryPath")
        self.checkpoint_path = parameters.get("agent:checkpointPath")
        self.resume = _as_bool(parameters.get("agent:resume", False), "agent:resume")
        self._prepare_artifacts()
        self._completed = self._load_checkpoints() if self.resume else {}
        self._graph = self._build_graph()

    def execute(self, batch: Dict[str, Dict[str, Any]]):
        for qid, qinfo in batch.items():
            if qid in self._completed:
                qinfo.update(deepcopy(self._completed[qid]))
                continue

            state: AgentState = {
                "qid": str(qid),
                "question": str(qinfo.get("qstring", "")),
                "messages": self._initial_messages(str(qinfo.get("qstring", ""))),
                "rankings": {},
                "ranking_order": [],
                "pending_tool_calls": [],
                "trajectory": [],
                "errors": [],
                "selected_ranking_id": None,
                "stop_reason": None,
                "answer": "",
                "final_prompt": [],
                "turns": 0,
                "retrieval_calls": 0,
                "rerank_calls": 0,
                "fusion_calls": 0,
                "tool_calls": 0,
                "tool_errors": 0,
                "budget_exhaustions": 0,
                "tool_usage": {},
                "llm_calls": 0,
                "llm_duration_seconds": 0.0,
                "usage": {},
                "response_models": [],
            }
            result = validated(self._graph.invoke(state))
            self._apply_result(qinfo, result)
            self._append_trajectory(result)
            self._append_checkpoint(str(qid), qinfo)
        return batch

    def _build_graph(self):
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError as exc:
            raise RuntimeError(
                "agentic_rag requires Python 3.11 and the 'agent' optional "
                "dependencies: pip install -e '.[agent]'"
            ) from exc

        graph = StateGraph(AgentState)
        graph.add_node("planner", self._planner_node)
        graph.add_node("tools", self._tool_node)
        graph.add_node("answer", self._answer_node)
        graph.add_edge(START, "planner")
        graph.add_conditional_edges(
            "planner", self._route_after_planner,
            {"tools": "tools", "answer": "answer"},
        )
        graph.add_conditional_edges(
            "tools", self._route_after_tools,
            {"planner": "planner", "answer": "answer"},
        )
        graph.add_edge("answer", END)
        return graph.compile()

    def _planner_node(self, state: AgentState):
        state = validated(state)
        if state["turns"] >= self.max_turns:
            return self._force_finish(state, "max_steps")
        try:
            response = self._complete_planner_with_retry(state)
            calls = [
                {"id": call.id, "name": call.name, "arguments": call.arguments}
                for call in response.tool_calls
            ]
            assistant_tool_calls = [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": json.dumps(call["arguments"], ensure_ascii=False),
                    },
                }
                for call in calls
            ]
            state["messages"].append({
                "role": "assistant",
                "content": response.content or None,
                "tool_calls": assistant_tool_calls,
            })
            state["pending_tool_calls"] = calls
            state["turns"] += 1
            state["llm_duration_seconds"] += response.duration_seconds
            _merge_usage(state["usage"], response.usage)
            if response.model:
                state["response_models"].append(response.model)
            state["trajectory"].append({
                "event": "planner",
                "turn": state["turns"],
                "tool_calls": calls,
                "finish_reason": response.finish_reason,
                "duration_seconds": response.duration_seconds,
                "usage": response.usage,
            })
            if not calls:
                if state.get("ranking_order"):
                    state["trajectory"].append({
                        "event": "model_finalize",
                        "message": "Planner stopped after obtaining a valid ranking.",
                    })
                    return self._force_finish(state, "model_finalize")
                state["errors"].append(
                    "Planner returned no tool call before obtaining a ranking."
                )
                return self._force_finish(state, "model_stop_no_ranking")
        except (LLMProviderError, NotImplementedError) as exc:
            state["errors"].append(str(exc))
            state["trajectory"].append({"event": "planner_error", "error": str(exc)})
            return self._force_finish(state, "planner_error")
        return state

    def _complete_planner_with_retry(self, state: AgentState):
        for attempt in range(self.max_planner_retries + 1):
            state["llm_calls"] += 1
            try:
                return self.provider.complete(
                    state["messages"],
                    tools=self.tools,
                    tool_choice="auto",
                    max_tokens=self.planner_max_tokens,
                )
            except LLMProviderError as exc:
                if attempt >= self.max_planner_retries:
                    raise
                state["trajectory"].append({
                    "event": "planner_retry",
                    "attempt": attempt + 1,
                    "error": str(exc),
                })
        raise AssertionError("unreachable")

    def _tool_node(self, state: AgentState):
        state = validated(state)
        calls = list(state["pending_tool_calls"])
        state["pending_tool_calls"] = []
        for call in calls:
            name = call["name"]
            state["tool_calls"] += 1
            state["tool_usage"][name] = state["tool_usage"].get(name, 0) + 1
            try:
                self._check_tool_budget(name, state)
                result = self.toolbox.execute(
                    name=name, arguments=call["arguments"], state=state
                )
                self._count_tool(name, state)
                if name == "finish_research":
                    state["selected_ranking_id"] = result["ranking_id"]
                    state["stop_reason"] = "finished"
                content = json.dumps(result, ensure_ascii=False)
                state["trajectory"].append({
                    "event": "tool",
                    "tool_call_id": call["id"],
                    "name": name,
                    "arguments": call["arguments"],
                    "result": result,
                })
            except ToolBudgetExceeded as exc:
                state["budget_exhaustions"] += 1
                content = json.dumps({
                    "status": "budget_exhausted",
                    "message": str(exc),
                    "instruction": "Use existing ranking_ids or finish research.",
                }, ensure_ascii=False)
                state["trajectory"].append({
                    "event": "tool_budget_exhausted",
                    "tool_call_id": call["id"],
                    "name": name,
                    "message": str(exc),
                })
            except (ValueError, ValidationError) as exc:
                content = json.dumps({"error": str(exc)}, ensure_ascii=False)
                state["tool_errors"] += 1
                state["errors"].append("{}: {}".format(name, exc))
                state["trajectory"].append({
                    "event": "tool_error",
                    "tool_call_id": call["id"],
                    "name": name,
                    "error": str(exc),
                })
            state["messages"].append({
                "role": "tool",
                "tool_call_id": call["id"],
                "name": name,
                "content": content,
            })
            if state.get("selected_ranking_id"):
                break
        if state["turns"] >= self.max_turns and not state.get("selected_ranking_id"):
            return self._force_finish(state, "max_steps")
        return state

    def _answer_node(self, state: AgentState):
        state = validated(state)
        ranking_id = state.get("selected_ranking_id")
        if not ranking_id:
            state["answer"] = ""
            state["stop_reason"] = state.get("stop_reason") or "no_ranking"
            return state
        passages, _ = self.toolbox.context_for(state, ranking_id)
        prompt = self.prompt_builder.build(state["question"], [p for p in passages if p])
        state["final_prompt"] = prompt
        try:
            response = self.provider.generate(prompt)
            state["answer"] = self.prompt_builder.post_process(response.content)
            state["llm_calls"] += 1
            state["llm_duration_seconds"] += response.duration_seconds
            _merge_usage(state["usage"], response.usage)
            if response.model:
                state["response_models"].append(response.model)
            state["trajectory"].append({
                "event": "answer",
                "ranking_id": ranking_id,
                "answer": state["answer"],
                "duration_seconds": response.duration_seconds,
                "usage": response.usage,
            })
        except LLMProviderError as exc:
            state["answer"] = ""
            state["errors"].append(str(exc))
            state["stop_reason"] = "answer_error"
            state["trajectory"].append({"event": "answer_error", "error": str(exc)})
        return state

    def _route_after_planner(self, state: AgentState):
        return "tools" if state.get("pending_tool_calls") else "answer"

    def _route_after_tools(self, state: AgentState):
        if state.get("selected_ranking_id"):
            return "answer"
        if state.get("stop_reason"):
            return "answer"
        return "planner"

    def _force_finish(self, state: AgentState, reason: str):
        if not state.get("selected_ranking_id") and state.get("ranking_order"):
            state["selected_ranking_id"] = state["ranking_order"][-1]
        state["stop_reason"] = reason
        state["pending_tool_calls"] = []
        return state

    def _check_tool_budget(self, name: str, state: AgentState):
        if name not in self.allowed_tools:
            raise ValueError("Tool is not allowed: {}".format(name))
        if name in {"search_bm25", "search_dense"}:
            if state["retrieval_calls"] >= self.max_retrieval_calls:
                raise ToolBudgetExceeded("Retrieval budget exhausted.")
        elif name == "rerank" and state["rerank_calls"] >= self.max_rerank_calls:
            raise ToolBudgetExceeded("Rerank budget exhausted.")
        elif name == "fuse_rankings" and state["fusion_calls"] >= self.max_fusion_calls:
            raise ToolBudgetExceeded("Fusion budget exhausted.")

    @staticmethod
    def _count_tool(name: str, state: AgentState):
        if name in {"search_bm25", "search_dense"}:
            state["retrieval_calls"] += 1
        elif name == "rerank":
            state["rerank_calls"] += 1
        elif name == "fuse_rankings":
            state["fusion_calls"] += 1

    def _initial_messages(self, question: str):
        return [
            {
                "role": "system",
                "content": (
                    "You are a bounded retrieval agent. Use tools to find evidence for "
                    "the question. Inspect the returned top-five passages. If evidence "
                    "is weak, rewrite the query or use another allowed retriever. Call "
                    "finish_research with the best ranking_id only when ready. You must "
                    "perform at least one search and must not answer directly. Available "
                    "tools: {}. Budgets: at most {} planner turns, {} retrieval calls, "
                    "{} rerank calls, and {} fusion calls."
                ).format(
                    ", ".join(self.allowed_tools), self.max_turns,
                    self.max_retrieval_calls, self.max_rerank_calls,
                    self.max_fusion_calls,
                ),
            },
            {"role": "user", "content": "Question: {}".format(question)},
        ]

    def _apply_result(self, qinfo: Dict[str, Any], state: AgentState):
        ranking_id = state.get("selected_ranking_id")
        record = state.get("rankings", {}).get(ranking_id or "", {})
        qinfo["ranking"] = record.get("ranking", [])
        qinfo["answer"] = state.get("answer", "")
        qinfo["prompt_rag"] = state.get("final_prompt", [])
        qinfo["agent"] = {
            "runtime": "langgraph",
            "allowed_tools": self.allowed_tools,
            "turns": state["turns"],
            "retrieval_calls": state["retrieval_calls"],
            "rerank_calls": state["rerank_calls"],
            "fusion_calls": state["fusion_calls"],
            "tool_calls": state["tool_calls"],
            "tool_errors": state["tool_errors"],
            "budget_exhaustions": state["budget_exhaustions"],
            "tool_usage": state["tool_usage"],
            "selected_ranking_id": ranking_id,
            "stop_reason": state.get("stop_reason"),
            "errors": state["errors"],
        }
        qinfo["llm"] = {
            "provider": self.provider.name,
            "model": (
                state["response_models"][-1]
                if state["response_models"] else self.provider.model
            ),
            "models": list(dict.fromkeys(state["response_models"])),
            "success": bool(state.get("answer")),
            "fallback_used": False,
            "duration_seconds": state["llm_duration_seconds"],
            "usage": state["usage"],
            "calls": state["llm_calls"],
            "stop_reason": state.get("stop_reason"),
        }

    def _append_trajectory(self, state: AgentState):
        if not self.trajectory_path:
            return
        path = Path(self.trajectory_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "qid": state["qid"],
            "question": state["question"],
            "trajectory": state["trajectory"],
            "agent": {
                "turns": state["turns"],
                "retrieval_calls": state["retrieval_calls"],
                "rerank_calls": state["rerank_calls"],
                "fusion_calls": state["fusion_calls"],
                "tool_calls": state["tool_calls"],
                "tool_errors": state["tool_errors"],
                "budget_exhaustions": state["budget_exhaustions"],
                "tool_usage": state["tool_usage"],
                "selected_ranking_id": state.get("selected_ranking_id"),
                "stop_reason": state.get("stop_reason"),
                "errors": state["errors"],
            },
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _prepare_artifacts(self):
        if self.resume:
            return
        for value in (self.trajectory_path, self.checkpoint_path):
            if not value:
                continue
            path = Path(value)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")

    def _append_checkpoint(self, qid: str, qinfo: Dict[str, Any]):
        if not self.checkpoint_path:
            return
        path = Path(self.checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"qid": qid, "qinfo": qinfo}, ensure_ascii=False) + "\n")

    def _load_checkpoints(self):
        if not self.checkpoint_path or not Path(self.checkpoint_path).is_file():
            return {}
        completed = {}
        with Path(self.checkpoint_path).open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                    qinfo = record["qinfo"]
                    stop_reason = qinfo.get("agent", {}).get("stop_reason")
                    if stop_reason not in {"planner_error", "answer_error"}:
                        completed[str(record["qid"])] = qinfo
                except (KeyError, TypeError, json.JSONDecodeError):
                    continue
        return completed


def _merge_usage(total: Dict[str, float], usage: Dict[str, Any]):
    for key, value in (usage or {}).items():
        if isinstance(value, (int, float)):
            total[key] = total.get(key, 0) + value


def _positive_int(parameters, key, default):
    value = int(parameters.get(key, default))
    if value <= 0:
        raise ValueError("{} must be greater than zero.".format(key))
    return value


def _nonnegative_int(parameters, key, default):
    value = int(parameters.get(key, default))
    if value < 0:
        raise ValueError("{} cannot be negative.".format(key))
    return value


def _as_bool(value, label):
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
