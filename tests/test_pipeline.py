import pytest

from qryeval_plus.core import Util
from qryeval_plus.core.Idx import Idx
from qryeval_plus.pipeline import run_pipeline
from qryeval_plus.retrieval.Ranker import Ranker


def test_index_is_closed_when_a_task_fails(tmp_path, monkeypatch):
    index_path = tmp_path / "index"
    index_path.mkdir()
    query_path = tmp_path / "queries.qry"
    query_path.write_text("1: example\n", encoding="utf-8")
    closed = []

    def fake_open(path):
        Idx.indexReader = object()
        return True

    def fake_close():
        closed.append(True)
        Idx.indexReader = None

    monkeypatch.setattr(Idx, "open", fake_open)
    monkeypatch.setattr(Idx, "close", fake_close)
    monkeypatch.setattr(Ranker, "execute", lambda self, batch: (_ for _ in ()).throw(RuntimeError("boom")))

    parameters = {
        "indexPath": str(index_path),
        "queryFilePath": str(query_path),
        "task_1:ranker": {"type": "BM25"},
    }
    with pytest.raises(RuntimeError, match="boom"):
        run_pipeline(parameters)
    assert closed == [True]


def test_partially_opened_index_is_closed_when_open_reports_failure(tmp_path, monkeypatch):
    index_path = tmp_path / "index"
    index_path.mkdir()
    query_path = tmp_path / "queries.qry"
    query_path.write_text("1: example\n", encoding="utf-8")
    closed = []

    def fake_open(path):
        Idx.indexReader = object()
        return False

    def fake_close():
        closed.append(True)
        Idx.indexReader = None

    monkeypatch.setattr(Idx, "open", fake_open)
    monkeypatch.setattr(Idx, "close", fake_close)
    parameters = {
        "indexPath": str(index_path),
        "queryFilePath": str(query_path),
        "task_1:ranker": {"type": "BM25"},
    }
    with pytest.raises(RuntimeError, match="Unable to open index"):
        run_pipeline(parameters)
    assert closed == [True]
