import json
import re
from pathlib import Path

from conftest import PROJECT_ROOT
from qryeval_plus.config import load_config, ordered_tasks, validate_config


def test_all_experiment_configs_are_valid_and_unique():
    config_paths = sorted((PROJECT_ROOT / "configs" / "rag").rglob("*.json"))
    assert len(config_paths) == 35

    output_paths = set()
    metadata_paths = set()
    query_paths = set()
    for path in config_paths:
        parameters = load_config(path)
        assert validate_config(parameters) == []
        task_numbers = [int(name.split(":", 1)[0].split("_", 1)[1]) for name, _, _ in ordered_tasks(parameters)]
        assert task_numbers == sorted(task_numbers)
        query_paths.add(parameters["queryFilePath"])
        for _, role, task in ordered_tasks(parameters):
            if role == "output":
                assert task["outputPath"] not in output_paths
                output_paths.add(task["outputPath"])
                if task["type"] == "triviaqa_evaluation":
                    assert task["metadataPath"] not in metadata_paths
                    metadata_paths.add(task["metadataPath"])
            if role == "agent":
                assert task["rag:provider"] == "deepseek"
                assert task["rag:apiKeyEnv"] == "DEEPSEEK_API_KEY"
                assert task["rag:fallback"] is False

    assert query_paths == {str((PROJECT_ROOT / "datasets" / "triviaqa" / "verified_wikipedia_dev.qry").resolve())}
    assert len(output_paths) == 70
    assert len(metadata_paths) == 35


def test_formal_project_files_do_not_use_legacy_course_naming():
    pattern = re.compile(chr(72) + chr(87), re.IGNORECASE)
    roots = [
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "configs",
        PROJECT_ROOT / "tests",
        PROJECT_ROOT / "docs",
        PROJECT_ROOT / "datasets",
    ]
    files = [PROJECT_ROOT / "README.md", PROJECT_ROOT / "pyproject.toml", PROJECT_ROOT / "data" / "README.md"]
    for root in roots:
        files.extend(
            path for path in root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and not any(part.endswith(".egg-info") for part in path.parts)
        )
    # The benchmark compatibility layer intentionally names the historical
    # experiment so its generated CSV can be compared directly with the
    # read-only source table. Product/runtime files remain course-name free.
    allowed = {
        "src/qryeval_plus/benchmark.py",
        "configs/benchmarks/hw5_agent.json",
        "README.md",
        "docs/AGENTIC_RAG.md",
        "docs/PROJECT_UPGRADE_REPORT_ZH.md",
        "tests/test_benchmark.py",
        "tests/test_configs_migrated.py",
    }
    offenders = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        relative = str(path.relative_to(PROJECT_ROOT))
        if (pattern.search(path.name) or pattern.search(text)) and relative not in allowed:
            offenders.append(relative)
    assert offenders == []
