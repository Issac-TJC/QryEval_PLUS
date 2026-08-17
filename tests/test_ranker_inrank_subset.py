from qryeval_plus.retrieval.Ranker import Ranker


def test_inrank_file_ignores_queries_outside_active_batch(tmp_path):
    run = tmp_path / "full.run"
    run.write_text(
        "q1 Q0 doc-1 1 2.0 test\n"
        "q2 Q0 doc-2 1 1.0 test\n",
        encoding="utf-8",
    )
    batch = {"q1": {"qstring": "first"}}

    result = Ranker({
        "type": "inRankFile",
        "inRankFile:Path": str(run),
    }).execute(batch)

    assert list(result) == ["q1"]
    assert result["q1"]["ranking"] == [(2.0, "doc-1")]


def test_inrank_file_uses_empty_ranking_for_missing_active_query(tmp_path):
    run = tmp_path / "partial.run"
    run.write_text("q1 Q0 doc-1 1 2.0 test\n", encoding="utf-8")

    result = Ranker({
        "type": "inRankFile",
        "inRankFile:Path": str(run),
    }).execute({"q2": {"qstring": "missing"}})

    assert result["q2"]["ranking"] == []
