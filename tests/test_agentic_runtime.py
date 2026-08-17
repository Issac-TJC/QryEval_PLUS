import json

import pytest

from qryeval_plus.llm import LLMResponse, LLMToolCall


langgraph = pytest.importorskip("langgraph")


class ScriptedProvider:
    name = "mock"
    model = "scripted"

    def __init__(self, turns, answer="potato"):
        self.turns = list(turns)
        self.answer = answer

    def complete(self, messages, **kwargs):
        call = self.turns.pop(0)
        return LLMResponse(
            content="",
            provider=self.name,
            model=self.model,
            duration_seconds=0.01,
            usage={"total_tokens": 10},
            tool_calls=[LLMToolCall(**call)],
            finish_reason="tool_calls",
        )

    def generate(self, messages):
        return LLMResponse(
            content=self.answer,
            provider=self.name,
            model=self.model,
            duration_seconds=0.01,
            usage={"total_tokens": 5},
        )


class FinalizingProvider(ScriptedProvider):
    def complete(self, messages, **kwargs):
        if self.turns:
            return super().complete(messages, **kwargs)
        return LLMResponse(
            content="Evidence is sufficient.",
            provider=self.name,
            model=self.model,
            duration_seconds=0.01,
            usage={"total_tokens": 4},
            finish_reason="stop",
        )

class FakeToolbox:
    def execute(self, *, name, arguments, state):
        if name in {"search_bm25", "search_dense"}:
            ranking_id = "r{}".format(len(state["ranking_order"]) + 1)
            state["rankings"][ranking_id] = {
                "ranking_id": ranking_id,
                "source": name.removeprefix("search_"),
                "query": arguments["query"],
                "ranking": [(1.0, "doc-{}".format(len(state["ranking_order"]) + 1))],
            }
            state["ranking_order"].append(ranking_id)
            return {"ranking_id": ranking_id, "evidence": "potato"}
        if name == "rerank":
            source = state["rankings"][arguments["ranking_id"]]
            ranking_id = "r{}".format(len(state["ranking_order"]) + 1)
            state["rankings"][ranking_id] = {
                **source,
                "ranking_id": ranking_id,
                "source": "rerank",
                "query": arguments["query"],
            }
            state["ranking_order"].append(ranking_id)
            return {"ranking_id": ranking_id, "evidence": "potato"}
        if name == "fuse_rankings":
            ranking_id = "r{}".format(len(state["ranking_order"]) + 1)
            state["rankings"][ranking_id] = {
                "ranking_id": ranking_id,
                "source": "rrf",
                "query": state["question"],
                "ranking": [(2.0, "doc-fused")],
            }
            state["ranking_order"].append(ranking_id)
            return {"ranking_id": ranking_id, "evidence": "potato"}
        if name == "finish_research":
            if arguments["ranking_id"] not in state["rankings"]:
                raise ValueError("unknown ranking")
            return {"ranking_id": arguments["ranking_id"], "finished": True}
        raise ValueError("unsupported")

    def context_for(self, state, ranking_id):
        return ["The Colorado beetle attacks potato crops."], "potato"


def _parameters(tmp_path, **overrides):
    values = {
        "type": "agentic_rag",
        "agent:allowedTools": ["search_bm25", "finish_research"],
        "agent:maxTurns": 6,
        "agent:maxRetrievalCalls": 3,
        "agent:maxRerankCalls": 0,
        "agent:maxFusionCalls": 0,
        "agent:trajectoryPath": str(tmp_path / "trace.jsonl"),
        "agent:checkpointPath": str(tmp_path / "checkpoint.jsonl"),
    }
    values.update(overrides)
    return values


def test_agent_runs_search_then_finish_and_writes_resume_checkpoint(tmp_path):
    from qryeval_plus.agentic.runtime import AgenticRagAgent

    provider = ScriptedProvider([
        {"id": "c1", "name": "search_bm25", "arguments": {"query": "beetle crop"}},
        {"id": "c2", "name": "finish_research", "arguments": {"ranking_id": "r1"}},
    ])
    agent = AgenticRagAgent(_parameters(tmp_path), provider=provider, toolbox=FakeToolbox())
    batch = agent.execute({"q1": {"qstring": "Which crop?"}})

    assert batch["q1"]["answer"] == "potato"
    assert batch["q1"]["ranking"] == [(1.0, "doc-1")]
    assert batch["q1"]["agent"]["stop_reason"] == "finished"
    assert batch["q1"]["agent"]["retrieval_calls"] == 1
    assert len((tmp_path / "trace.jsonl").read_text().splitlines()) == 1

    resumed_provider = ScriptedProvider([])
    resumed = AgenticRagAgent(
        _parameters(tmp_path, **{"agent:resume": True}),
        provider=resumed_provider,
        toolbox=FakeToolbox(),
    )
    resumed_batch = resumed.execute({"q1": {"qstring": "Which crop?"}})
    assert resumed_batch["q1"]["answer"] == "potato"
    assert resumed_provider.turns == []


def test_retrieval_budget_rejection_is_not_a_tool_error_and_agent_can_finish(tmp_path):
    from qryeval_plus.agentic.runtime import AgenticRagAgent

    provider = ScriptedProvider([
        {"id": "c1", "name": "search_bm25", "arguments": {"query": "beetle crop"}},
        {"id": "c2", "name": "search_bm25", "arguments": {"query": "second search"}},
        {"id": "c3", "name": "finish_research", "arguments": {"ranking_id": "r1"}},
    ])
    agent = AgenticRagAgent(
        _parameters(tmp_path, **{"agent:maxRetrievalCalls": 1}),
        provider=provider,
        toolbox=FakeToolbox(),
    )
    batch = agent.execute({"q1": {"qstring": "Which crop?"}})

    assert batch["q1"]["answer"] == "potato"
    assert batch["q1"]["agent"]["stop_reason"] == "finished"
    assert batch["q1"]["agent"]["retrieval_calls"] == 1
    assert batch["q1"]["agent"]["tool_errors"] == 0
    trajectory = json.loads((tmp_path / "trace.jsonl").read_text())
    assert any(
        event["event"] == "tool_budget_exhausted"
        for event in trajectory["trajectory"]
    )


def test_full_tool_graph_can_search_rerank_fuse_and_finish(tmp_path):
    from qryeval_plus.agentic.runtime import AgenticRagAgent

    provider = ScriptedProvider([
        {"id": "c1", "name": "search_dense", "arguments": {"query": "crop pest"}},
        {"id": "c2", "name": "rerank", "arguments": {"ranking_id": "r1", "query": "crop pest"}},
        {"id": "c3", "name": "search_bm25", "arguments": {"query": "Colorado beetle crop"}},
        {"id": "c4", "name": "fuse_rankings", "arguments": {"ranking_ids": ["r2", "r3"]}},
        {"id": "c5", "name": "finish_research", "arguments": {"ranking_id": "r4"}},
    ])
    parameters = _parameters(
        tmp_path,
        **{
            "agent:allowedTools": [
                "search_bm25", "search_dense", "rerank",
                "fuse_rankings", "finish_research",
            ],
            "agent:maxRerankCalls": 1,
            "agent:maxFusionCalls": 1,
        },
    )
    batch = AgenticRagAgent(
        parameters, provider=provider, toolbox=FakeToolbox()
    ).execute({"q1": {"qstring": "Which crop?"}})

    metadata = batch["q1"]["agent"]
    assert metadata["stop_reason"] == "finished"
    assert metadata["tool_calls"] == 5
    assert metadata["tool_usage"] == {
        "search_dense": 1,
        "rerank": 1,
        "search_bm25": 1,
        "fuse_rankings": 1,
        "finish_research": 1,
    }
    assert batch["q1"]["ranking"] == [(2.0, "doc-fused")]


def test_model_stop_with_valid_ranking_is_a_normal_finalize(tmp_path):
    from qryeval_plus.agentic.runtime import AgenticRagAgent

    provider = FinalizingProvider([
        {"id": "c1", "name": "search_bm25", "arguments": {"query": "crop"}},
    ])
    batch = AgenticRagAgent(
        _parameters(tmp_path), provider=provider, toolbox=FakeToolbox()
    ).execute({"q1": {"qstring": "Which crop?"}})

    assert batch["q1"]["answer"] == "potato"
    assert batch["q1"]["agent"]["stop_reason"] == "model_finalize"
    assert batch["q1"]["agent"]["errors"] == []


def test_invalid_planner_response_is_retried_once(tmp_path):
    from qryeval_plus.agentic.runtime import AgenticRagAgent
    from qryeval_plus.llm import LLMProviderError

    class RetryProvider(ScriptedProvider):
        def __init__(self):
            super().__init__([
                {"id": "c1", "name": "search_bm25", "arguments": {"query": "crop"}},
                {"id": "c2", "name": "finish_research", "arguments": {"ranking_id": "r1"}},
            ])
            self.failed = False

        def complete(self, messages, **kwargs):
            if not self.failed:
                self.failed = True
                raise LLMProviderError("invalid tool call")
            return super().complete(messages, **kwargs)

    batch = AgenticRagAgent(
        _parameters(tmp_path), provider=RetryProvider(), toolbox=FakeToolbox()
    ).execute({"q1": {"qstring": "Which crop?"}})

    assert batch["q1"]["answer"] == "potato"
    assert batch["q1"]["agent"]["errors"] == []
    assert batch["q1"]["llm"]["calls"] == 4
    trajectory = json.loads((tmp_path / "trace.jsonl").read_text())
    assert any(event["event"] == "planner_retry" for event in trajectory["trajectory"])


def test_invalid_finish_at_turn_limit_records_max_steps(tmp_path):
    from qryeval_plus.agentic.runtime import AgenticRagAgent

    provider = ScriptedProvider([
        {"id": "c1", "name": "search_bm25", "arguments": {"query": "crop"}},
        {"id": "c2", "name": "finish_research", "arguments": {"ranking_id": "missing"}},
    ])
    agent = AgenticRagAgent(
        _parameters(tmp_path, **{"agent:maxTurns": 2}),
        provider=provider,
        toolbox=FakeToolbox(),
    )
    batch = agent.execute({"q1": {"qstring": "Which crop?"}})

    assert batch["q1"]["answer"] == "potato"
    assert batch["q1"]["agent"]["stop_reason"] == "max_steps"
    assert batch["q1"]["agent"]["tool_errors"] == 1


def test_no_valid_ranking_at_turn_limit_produces_empty_answer(tmp_path):
    from qryeval_plus.agentic.runtime import AgenticRagAgent

    provider = ScriptedProvider([
        {"id": "c1", "name": "finish_research", "arguments": {"ranking_id": "missing"}},
    ])
    agent = AgenticRagAgent(
        _parameters(tmp_path, **{"agent:maxTurns": 1}),
        provider=provider,
        toolbox=FakeToolbox(),
    )
    batch = agent.execute({"q1": {"qstring": "Which crop?"}})

    assert batch["q1"]["answer"] == ""
    assert batch["q1"]["ranking"] == []
    assert batch["q1"]["agent"]["stop_reason"] == "max_steps"


def test_tool_arguments_reject_unknown_fields():
    from pydantic import ValidationError
    from qryeval_plus.agentic.tools import AgentToolbox

    state = {"qid": "q1", "question": "question", "rankings": {}, "ranking_order": []}
    with pytest.raises(ValidationError):
        AgentToolbox({}).execute(
            name="search_bm25",
            arguments={"query": "valid", "unexpected": True},
            state=state,
        )


def test_unknown_tool_is_rejected_at_configuration_time(tmp_path):
    from qryeval_plus.agentic.runtime import AgenticRagAgent

    with pytest.raises(ValueError, match="Unknown agent tools"):
        AgenticRagAgent(
            _parameters(
                tmp_path,
                **{"agent:allowedTools": ["search_bm25", "web_search", "finish_research"]},
            ),
            provider=ScriptedProvider([]),
            toolbox=FakeToolbox(),
        )


def test_planner_api_failure_is_recorded_without_fallback(tmp_path):
    from qryeval_plus.agentic.runtime import AgenticRagAgent
    from qryeval_plus.llm import LLMProviderError

    class FailingProvider:
        name = "mock"
        model = "failure"

        def complete(self, messages, **kwargs):
            raise LLMProviderError("simulated outage")

        def generate(self, messages):
            raise AssertionError("answer generation must not run without a ranking")

    batch = AgenticRagAgent(
        _parameters(tmp_path), provider=FailingProvider(), toolbox=FakeToolbox()
    ).execute({"q1": {"qstring": "Which crop?"}})

    assert batch["q1"]["answer"] == ""
    assert batch["q1"]["ranking"] == []
    assert batch["q1"]["agent"]["stop_reason"] == "planner_error"
    assert "simulated outage" in batch["q1"]["agent"]["errors"][0]

    resumed_provider = ScriptedProvider([
        {"id": "c1", "name": "search_bm25", "arguments": {"query": "crop"}},
        {"id": "c2", "name": "finish_research", "arguments": {"ranking_id": "r1"}},
    ])
    resumed = AgenticRagAgent(
        _parameters(tmp_path, **{"agent:resume": True}),
        provider=resumed_provider,
        toolbox=FakeToolbox(),
    ).execute({"q1": {"qstring": "Which crop?"}})
    assert resumed["q1"]["answer"] == "potato"
    assert resumed_provider.turns == []
