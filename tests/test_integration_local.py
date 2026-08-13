import os
import time

import pytest

from conftest import PROJECT_ROOT


pytestmark = pytest.mark.integration


@pytest.mark.skipif(os.environ.get("QRYEVAL_RUN_INTEGRATION") != "1", reason="local integration test is opt-in")
def test_dense_rerank_and_mocked_answer_generation():
    from qryeval_plus.core.Idx import Idx
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

        agent = Agent({
            "type": "rag",
            "agentDepth": 1,
            "rag:modelServer": "127.0.0.1/1",
            "rag:dense:modelPath": str(data / "models" / "co-condenser-marco-retriever"),
            "rag:psgLen": 100,
            "rag:psgStride": 80,
            "rag:psgCnt": 1,
            "rag:maxTitleLength": 15,
            "rag:prompt": 4,
            "rag:email": "test@example.invalid",
            "rag:code": "test",
        })
        agent.send_to_llm = lambda address, messages: "V-2 rocket"
        batch = agent.execute(batch)
        assert batch["1"]["answer"] == "V-2 rocket"
    finally:
        Idx.close()
