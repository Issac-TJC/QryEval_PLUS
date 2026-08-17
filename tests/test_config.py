import json
from pathlib import Path

import pytest

from qryeval_plus.config import ConfigError, load_config, ordered_tasks, validate_config


def test_paths_resolve_against_config_directory(tmp_path, monkeypatch):
    config_dir = tmp_path / "nested" / "configs"
    config_dir.mkdir(parents=True)
    (tmp_path / "index").mkdir()
    (tmp_path / "queries.qry").write_text("1: example\n", encoding="utf-8")
    config = {
        "indexPath": "../../index",
        "queryFilePath": "../../queries.qry",
        "task_10:output": {"type": "triviaqa_evaluation", "outputPath": "../../out/a.json"},
        "task_2:ranker": {"type": "BM25"},
    }
    config_path = config_dir / "experiment.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.chdir(tmp_path.parent)

    loaded = load_config(config_path)

    assert loaded["indexPath"] == str((tmp_path / "index").resolve())
    assert loaded["queryFilePath"] == str((tmp_path / "queries.qry").resolve())
    assert [role for _, role, _ in ordered_tasks(loaded)] == ["ranker", "output"]


def test_invalid_task_key_is_rejected():
    with pytest.raises(ConfigError, match="Invalid task key"):
        ordered_tasks({"task_first:ranker": {"type": "BM25"}})


def test_validation_reports_missing_assets(tmp_path):
    parameters = {
        "indexPath": str(tmp_path / "missing-index"),
        "queryFilePath": str(tmp_path / "missing.qry"),
        "task_1:ranker": {"type": "BM25"},
    }
    errors = validate_config(parameters)
    assert any("missing-index" in error for error in errors)
    assert any("missing.qry" in error for error in errors)
