import os
import time

import pytest

from conftest import PROJECT_ROOT


pytestmark = pytest.mark.integration


@pytest.mark.skipif(os.environ.get("QRYEVAL_RUN_INTEGRATION") != "1", reason="local integration test is opt-in")
def test_dense_rerank_and_mocked_answer_generation():
    from qryeval_plus.core.Idx import Idx
    from qryeval_plus.llm import LLMResponse
    from qryeval_plus.rag.Agent import Agent
    from qryeval_plus.rerank.Reranker import Reranker
    from qryeval_plus.retrieval.Ranker import Ranker

    data = PROJECT_ROOT / "data"
    assert Idx.open(str(data / "indexes" / "clueweb22-lucene"))
    try:
        assert Idx.getNumDocs() == 273140
        sparse_batch = {"1": {"qstring": "renewable energy storage"}}
        sparse_batch = Ranker({
            "type": "BM25",
            "outputLength": 3,
            "BM25:k_1": 1.2,
            "BM25:b": 0.75,
        }).execute(sparse_batch)
        assert sparse_batch["1"]["ranking"]

        batch = {"1": {"qstring": "what powered the first human made object to enter space"}}
        batch = Ranker({
            "type": "dense",
            "outputLength": 3,
            "dense:indexPath": str(data / "indexes" / "clueweb22-faiss-b300-fp"),
            "dense:modelPath": str(data / "models" / "co-condenser-marco-retriever"),
        }).execute(batch)
        assert len(batch["1"]["ranking"]) == 3
        assert batch["1"]["ranking"][0][1]

        batch = Reranker({
            "type": "bertrr",
            "rerankDepth": 1,
            "bertrr:modelPath": str(data / "models" / "ms-marco-minilm-l6-v2"),
            "bertrr:psgLen": 150,
            "bertrr:psgStride": 140,
            "bertrr:psgCnt": 1,
            "bertrr:scoreAggregation": "maxp",
            "bertrr:maxTitleLength": 15,
        }).execute(batch)
        assert isinstance(batch["1"]["ranking"][0][0], float)

        class StaticProvider:
            name = "mock"
            model = "deterministic-test"

            def generate(self, messages):
                return LLMResponse(
                    content="V-2 rocket",
                    provider=self.name,
                    model=self.model,
                    duration_seconds=0.0,
                )

        agent = Agent({
            "type": "rag",
            "agentDepth": 1,
            "rag:provider": "deepseek",
            "rag:dense:modelPath": str(data / "models" / "co-condenser-marco-retriever"),
            "rag:psgLen": 100,
            "rag:psgStride": 80,
            "rag:psgCnt": 1,
            "rag:maxTitleLength": 15,
            "rag:prompt": 4,
        }, provider=StaticProvider())
        batch = agent.execute(batch)
        assert batch["1"]["answer"] == "V-2 rocket"
        assert batch["1"]["llm"]["provider"] == "mock"
    finally:
        Idx.close()


@pytest.mark.skipif(os.environ.get("QRYEVAL_RUN_INTEGRATION") != "1", reason="local integration test is opt-in")
def test_agentic_bm25_pipeline_with_real_index_and_mock_planner(tmp_path, monkeypatch):
    from copy import deepcopy

    from qryeval_plus.config import load_config
    from qryeval_plus.llm import LLMResponse, LLMToolCall
    from qryeval_plus.pipeline import run_pipeline

    parameters = deepcopy(load_config(PROJECT_ROOT / "configs" / "agent" / "bm25_agent.json"))
    first_query = (
        PROJECT_ROOT / "datasets" / "triviaqa" / "verified_wikipedia_dev.qry"
    ).read_text(encoding="utf-8").splitlines()[0]
    qid, question = first_query.split(":", 1)
    query_path = tmp_path / "query.qry"
    query_path.write_text(first_query + "\n", encoding="utf-8")
    parameters["queryFilePath"] = str(query_path)
    agent_parameters = parameters["task_1:agent"]
    agent_parameters["agent:retrievalDepth"] = 5
    agent_parameters["agent:trajectoryPath"] = str(tmp_path / "trajectory.jsonl")
    agent_parameters["agent:checkpointPath"] = str(tmp_path / "checkpoint.jsonl")
    parameters["task_2:output"]["outputPath"] = str(tmp_path / "result.run")
    parameters["task_3:output"]["outputPath"] = str(tmp_path / "answers.json")
    parameters["task_3:output"]["metadataPath"] = str(tmp_path / "llm.json")

    class Planner:
        name = "mock"
        model = "scripted-tools"

        def __init__(self):
            self.turn = 0

        def complete(self, messages, **kwargs):
            self.turn += 1
            call = (
                LLMToolCall(
                    id="search", name="search_bm25",
                    arguments={"query": question.strip()},
                )
                if self.turn == 1
                else LLMToolCall(
                    id="finish", name="finish_research",
                    arguments={"ranking_id": "r1"},
                )
            )
            return LLMResponse(
                content="", provider=self.name, model=self.model,
                duration_seconds=0.0, usage={"total_tokens": 1},
                tool_calls=[call], finish_reason="tool_calls",
            )

        def generate(self, messages):
            return LLMResponse(
                content="V-2 rocket", provider=self.name, model=self.model,
                duration_seconds=0.0, usage={"total_tokens": 1},
            )

    monkeypatch.setattr(
        "qryeval_plus.agentic.runtime.create_provider", lambda _: Planner()
    )
    batch = run_pipeline(parameters)

    assert batch[qid]["answer"] == "V-2 rocket"
    assert batch[qid]["ranking"]
    assert batch[qid]["agent"]["tool_usage"] == {
        "search_bm25": 1, "finish_research": 1,
    }
    assert (tmp_path / "trajectory.jsonl").is_file()
    assert (tmp_path / "checkpoint.jsonl").is_file()
    assert (tmp_path / "result.run").is_file()
    assert (tmp_path / "answers.json").is_file()
