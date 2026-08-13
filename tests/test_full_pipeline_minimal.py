import json
import os
from copy import deepcopy

import pytest

from conftest import PROJECT_ROOT
from qryeval_plus.config import load_config


pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.environ.get("QRYEVAL_RUN_INTEGRATION") != "1",
    reason="local integration test is opt-in",
)
def test_selected_system_runs_the_complete_pipeline_with_one_query(tmp_path, monkeypatch):
    """Exercise dense retrieval, neural reranking, both outputs, and RAG."""
    from qryeval_plus.rag.Agent import Agent
    from qryeval_plus.pipeline import run_pipeline

    config_path = (
        PROJECT_ROOT
        / "configs"
        / "rag"
        / "systems"
        / "dense_minilm_l6_rerank.json"
    )
    parameters = deepcopy(load_config(config_path))

    source_queries = PROJECT_ROOT / "datasets" / "triviaqa" / "verified_wikipedia_dev.qry"
    first_query = source_queries.read_text(encoding="utf-8").splitlines()[0]
    query_path = tmp_path / "one_query.qry"
    query_path.write_text(first_query + "\n", encoding="utf-8")

    run_path = tmp_path / "selected.run"
    answer_path = tmp_path / "selected.answers.json"
    prompt_path = tmp_path / "selected.prompts.txt"

    parameters["queryFilePath"] = str(query_path)
    parameters["task_1:ranker"]["outputLength"] = 1
    parameters["task_2:reranker"]["rerankDepth"] = 1
    parameters["task_2:reranker"]["bertrr:psgCnt"] = 1
    parameters["task_3:output"]["outputPath"] = str(run_path)
    parameters["task_3:output"]["outputLength"] = 1
    parameters["task_4:agent"]["agentDepth"] = 1
    parameters["task_4:agent"]["rag:psgCnt"] = 1
    parameters["task_4:agent"]["rag:promptPath"] = str(prompt_path)
    parameters["task_5:output"]["outputPath"] = str(answer_path)
    parameters["task_5:output"]["promptPath"] = str(prompt_path)

    monkeypatch.setattr(
        Agent,
        "send_to_llm",
        staticmethod(lambda address, messages: "V-2 rocket"),
    )

    batch = run_pipeline(parameters)

    qid = first_query.split(":", 1)[0]
    assert list(batch) == [qid]
    assert len(batch[qid]["ranking"]) == 1
    assert batch[qid]["ranking"][0][1]
    assert batch[qid]["answer"] == "V-2 rocket"

    run_fields = run_path.read_text(encoding="utf-8").strip().split()
    assert run_fields[0] == qid
    assert run_fields[1] == "Q0"
    assert run_fields[2] == batch[qid]["ranking"][0][1]
    assert json.loads(answer_path.read_text(encoding="utf-8")) == {
        qid: "V-2 rocket"
    }
    prompt_text = prompt_path.read_text(encoding="utf-8")
    assert qid in prompt_text
    assert "Question:" in prompt_text
