import json

from conftest import PROJECT_ROOT
from qryeval_plus.benchmark import (
    _filter_qrels,
    _trec_metrics,
    evaluate_outputs,
    run_benchmark,
)
from qryeval_plus.config import ordered_tasks


def test_evaluator_matches_known_five_question_result(tmp_path):
    source_answers = PROJECT_ROOT / "outputs/rag/systems/dense_minilm_l6_rerank.answers.json"
    source_run = PROJECT_ROOT / "outputs/rag/systems/dense_minilm_l6_rerank.run"
    if not source_answers.is_file() or not source_run.is_file():
        return
    answers = json.loads(source_answers.read_text())
    qids = list(answers)
    qrel_source = PROJECT_ROOT / "data/evaluation/triviaqa/verified-wikipedia-dev.qrel"
    qrel = tmp_path / "qrels"
    qrel.write_text("\n".join(
        line for line in qrel_source.read_text().splitlines()
        if line.split()[0] in set(qids)
    ) + "\n")

    metrics = evaluate_outputs(
        answers_path=source_answers,
        run_path=source_run,
        gold_path=PROJECT_ROOT / "data/evaluation/triviaqa/verified-wikipedia-dev.json",
        qrel_path=qrel,
        trec_eval_path=PROJECT_ROOT / "data/tools/trec_eval",
        qids=qids,
    )

    assert metrics["exact"] == 40.0
    assert round(metrics["f1"], 2) == 69.33
    assert metrics["MRR"] == 0.4567


def test_complete_bm25_run_reproduces_historical_retrieval_metrics(tmp_path):
    query_lines = (
        PROJECT_ROOT / "datasets" / "triviaqa" / "verified_wikipedia_dev.qry"
    ).read_text(encoding="utf-8").splitlines()
    qids = [line.split(":", 1)[0] for line in query_lines]
    qrels = _filter_qrels(
        PROJECT_ROOT / "data" / "evaluation" / "triviaqa" / "verified-wikipedia-dev.qrel",
        tmp_path / "qrels",
        qids,
    )
    metrics, per_query = _trec_metrics(
        PROJECT_ROOT / "data" / "tools" / "trec_eval",
        qrels,
        PROJECT_ROOT / "datasets" / "triviaqa" / "bm25_baseline.run",
        qids,
    )

    assert metrics == {"recip_rank": 0.6643, "P_1": 0.6, "P_5": 0.415}
    assert set(per_query) == set(qids)


def test_benchmark_runner_writes_comparison_report_and_snapshots(tmp_path, monkeypatch):
    reference = PROJECT_ROOT / "benchmarks" / "hw5" / "reference"
    manifest = {
        "queryFilePath": str(PROJECT_ROOT / "datasets" / "triviaqa" / "verified_wikipedia_dev.qry"),
        "qrelPath": str(PROJECT_ROOT / "data" / "evaluation" / "triviaqa" / "verified-wikipedia-dev.qrel"),
        "goldPath": str(PROJECT_ROOT / "data" / "evaluation" / "triviaqa" / "verified-wikipedia-dev.json"),
        "trecEvalPath": str(PROJECT_ROOT / "data" / "tools" / "trec_eval"),
        "historicalReferencePath": str(reference / "HW5_Exp3_CustomExperiments.csv"),
        "historicalProvenancePath": str(reference / "provenance.json"),
        "outputRoot": str(tmp_path / "output"),
        "systems": [
            {
                "name": "historical", "label": "Historical", "historical": True,
                "metrics": {"exact": 40, "f1": 54.17, "MRR": 0.6643, "P@1": 0.6, "P@5": 0.415},
            },
            {
                "name": "fixed", "label": "Fixed", "baseline": True,
                "config": str(PROJECT_ROOT / "configs" / "rag" / "systems" / "bm25_baseline.json"),
            },
            {
                "name": "agent", "label": "Agent",
                "config": str(PROJECT_ROOT / "configs" / "agent" / "bm25_agent.json"),
            },
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    rankings = {}
    for line in (PROJECT_ROOT / "datasets" / "triviaqa" / "bm25_baseline.run").read_text().splitlines():
        fields = line.split()
        rankings.setdefault(fields[0], []).append(line)
    gold_data = json.loads((PROJECT_ROOT / "data" / "evaluation" / "triviaqa" / "verified-wikipedia-dev.json").read_text())
    gold = {item["QuestionId"]: item["Answer"]["NormalizedAliases"][0] for item in gold_data["Data"]}

    def fake_pipeline(parameters):
        qids = [line.split(":", 1)[0] for line in open(parameters["queryFilePath"], encoding="utf-8")]
        agentic = any(
            role == "agent" and task["type"] == "agentic_rag"
            for _, role, task in ordered_tasks(parameters)
        )
        for _, role, task in ordered_tasks(parameters):
            if role != "output":
                continue
            path = task["outputPath"]
            if task["type"] == "trec_eval":
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write("\n".join(line for qid in qids for line in rankings[qid]) + "\n")
            else:
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump({qid: gold[qid] for qid in qids}, handle)
                metadata = {}
                for qid in qids:
                    row = {
                        "provider": "mock", "model": "mock", "success": True,
                        "duration_seconds": 0.1, "usage": {"total_tokens": 10}, "calls": 3,
                    }
                    if agentic:
                        row["agent"] = {
                            "turns": 2, "retrieval_calls": 1, "rerank_calls": 0,
                            "fusion_calls": 0, "tool_calls": 2, "tool_errors": 0,
                            "tool_usage": {"search_bm25": 1, "finish_research": 1},
                            "stop_reason": "finished", "errors": [],
                        }
                    metadata[qid] = row
                with open(task["metadataPath"], "w", encoding="utf-8") as handle:
                    json.dump(metadata, handle)
        return {}

    monkeypatch.setattr("qryeval_plus.benchmark.run_pipeline", fake_pipeline)
    output = run_benchmark(str(manifest_path), limit=5)

    assert (output / "HW5_Agent_Comparison.csv").is_file()
    assert (output / "HW5_AGENT_REPORT.md").is_file()
    assert (output / "per_question_comparison.csv").is_file()
    assert (output / "environment.json").is_file()
    snapshot = json.loads((output / "benchmark_snapshot.json").read_text())
    assert snapshot["limit"] == 5
    assert snapshot["historical_reference"]["sha256"] == "97d3f136d4198ede2bcd31509156029e813eade98ceae833d5b579ea284b0dda"
    agent_metrics = json.loads((output / "agent" / "metrics.json").read_text())
    assert agent_metrics["avg_model_calls"] == 3
    assert agent_metrics["tool_usage"]["search_bm25"] == 5
