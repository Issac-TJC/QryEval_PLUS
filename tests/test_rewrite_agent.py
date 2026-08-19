import pytest

pytest.importorskip("langgraph")

from qryeval_plus.agentic.rewrite import RewriteRagAgent
from qryeval_plus.llm import LLMResponse, LLMToolCall


class Controller:
    name = "mock"
    model = "controller"

    def __init__(self, action):
        self.action = action
        self.calls = 0

    def complete(self, messages, **kwargs):
        self.calls += 1
        if self.action == "rewrite_query":
            arguments = {"query": "death spies agency"}
        else:
            arguments = {}
        return LLMResponse(
            content="", provider=self.name, model=self.model, duration_seconds=0.01,
            usage={"total_tokens": 5},
            tool_calls=[LLMToolCall(id="c1", name=self.action, arguments=arguments)],
        )

    def generate(self, messages):
        self.calls += 1
        return LLMResponse(
            content="SMERSH", provider=self.name, model=self.model,
            duration_seconds=0.01, usage={"total_tokens": 3},
        )


class Toolbox:
    def execute(self, *, name, arguments, state):
        ranking_id = "r{}".format(len(state["ranking_order"]) + 1)
        state["ranking_order"].append(ranking_id)
        state["rankings"][ranking_id] = {
            "ranking_id": ranking_id, "source": "bm25", "query": arguments["query"],
            "ranking": [(10.0, "doc-1"), (8.0, "doc-2")],
        }
        return {
            "ranking_id": ranking_id,
            "evidence": [{"rank": 1, "doc_id": "doc-1", "score": 10.0, "passage": "SMERSH means death to spies."}],
        }

    def context_for(self, state, ranking_id):
        evidence = [{"rank": 1, "doc_id": "doc-1", "score": 10.0, "passage": "SMERSH means death to spies."}]
        return [evidence[0]["passage"]], evidence


def parameters(tmp_path, policy):
    return {
        "type": "rewrite_rag", "rewrite:policy": policy,
        "rag:prompt": 1, "agent:checkpointPath": str(tmp_path / "checkpoint.jsonl"),
        "agent:trajectoryPath": str(tmp_path / "trajectory.jsonl"),
    }


def test_adaptive_policy_can_finish_after_one_retrieval(tmp_path):
    provider = Controller("finish")
    batch = RewriteRagAgent(
        parameters(tmp_path, "adaptive"), provider=provider, toolbox=Toolbox()
    ).execute({"q1": {"qstring": "Which agency means death to spies?"}})
    assert batch["q1"]["rewrite"]["decision"] == "finish"
    assert batch["q1"]["rewrite"]["retrieval_calls"] == 1
    assert batch["q1"]["answer"] == "SMERSH"
    assert batch["q1"]["grounding_status"] == "answered"


def test_always_rewrite_uses_exactly_two_retrievals(tmp_path):
    batch = RewriteRagAgent(
        parameters(tmp_path, "always"), provider=Controller("rewrite_query"), toolbox=Toolbox()
    ).execute({"q1": {"qstring": "Which agency means death to spies?"}})
    assert batch["q1"]["rewrite"]["rewritten_query"] == "death spies agency"
    assert batch["q1"]["rewrite"]["retrieval_calls"] == 2


def test_empty_initial_retrieval_abstains_without_model_call(tmp_path):
    class EmptyToolbox(Toolbox):
        def execute(self, *, name, arguments, state):
            state["ranking_order"].append("r1")
            state["rankings"]["r1"] = {
                "ranking_id": "r1", "source": "bm25", "query": arguments["query"],
                "ranking": [],
            }
            return {"ranking_id": "r1", "evidence": []}

    provider = Controller("finish")
    batch = RewriteRagAgent(
        parameters(tmp_path, "adaptive"), provider=provider, toolbox=EmptyToolbox()
    ).execute({"q1": {"qstring": "unknown"}})
    assert batch["q1"]["answer"] == ""
    assert batch["q1"]["grounding_status"] == "abstained"
    assert batch["q1"]["rewrite"]["stop_reason"] == "empty_retrieval"
    assert provider.calls == 0
