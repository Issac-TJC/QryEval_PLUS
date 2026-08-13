"""Configuration loading, path resolution, and validation."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from qryeval_plus.core import Util


_TASK_RE = re.compile(r"^task_(\d+):([a-z]+)$")
_PATH_KEYS = {
    "indexPath",
    "queryFilePath",
    "inRankFile:Path",
    "outputPath",
    "promptPath",
    "prf:expansionQueryFile",
    "dense:indexPath",
    "dense:modelPath",
    "rag:dense:modelPath",
    "rag:promptPath",
    "bertrr:modelPath",
    "bertrr:topPsgPath",
    "ltr:trainingQueryFile",
    "ltr:trainingQrelsFile",
    "ltr:trainingFeatureVectorsFile",
    "ltr:testingFeatureVectorsFile",
    "ltr:testingDocumentScores",
    "ltr:modelFile",
    "ltr:svmRankLearnPath",
    "ltr:svmRankClassifyPath",
}


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


def _resolve_paths(value: Any, base_dir: Path) -> Any:
    if isinstance(value, dict):
        resolved = {}
        for key, item in value.items():
            if key in _PATH_KEYS and isinstance(item, str) and item.strip():
                path = Path(item).expanduser()
                resolved[key] = str(path if path.is_absolute() else (base_dir / path).resolve())
            else:
                resolved[key] = _resolve_paths(item, base_dir)
        return resolved
    if isinstance(value, list):
        return [_resolve_paths(item, base_dir) for item in value]
    return value


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load JSON and resolve every configured path against its file directory."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"Configuration does not exist: {config_path}")

    try:
        with config_path.open(encoding="utf-8") as handle:
            parameters = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in {config_path}: {exc}") from exc

    if not isinstance(parameters, dict):
        raise ConfigError("Configuration root must be a JSON object.")

    parameters = Util.str_to_num(parameters)
    parameters = _resolve_paths(parameters, config_path.parent)
    parameters["_configPath"] = str(config_path)
    return parameters


def ordered_tasks(parameters: Dict[str, Any]) -> List[Tuple[str, str, Dict[str, Any]]]:
    """Return task definitions sorted by their numeric task prefix."""
    tasks = []
    seen_numbers = set()
    for key, value in parameters.items():
        if not key.startswith("task_"):
            continue
        match = _TASK_RE.fullmatch(key)
        if match is None:
            raise ConfigError(f"Invalid task key: {key}")
        number = int(match.group(1))
        if number in seen_numbers:
            raise ConfigError(f"Duplicate task number: {number}")
        if not isinstance(value, dict):
            raise ConfigError(f"Task {key} must be an object.")
        seen_numbers.add(number)
        tasks.append((number, key, match.group(2), value))
    return [(key, role, value) for _, key, role, value in sorted(tasks)]


def validate_config(parameters: Dict[str, Any], check_assets: bool = True) -> List[str]:
    """Return validation errors without importing the Java bridge."""
    errors: List[str] = []
    for key in ("indexPath", "queryFilePath"):
        if not parameters.get(key):
            errors.append(f"Missing required key: {key}")

    try:
        tasks = ordered_tasks(parameters)
    except ConfigError as exc:
        return errors + [str(exc)]

    if not tasks:
        errors.append("At least one task is required.")

    valid_roles = {"ranker", "rewriter", "reranker", "agent", "output"}
    for task_key, role, task in tasks:
        if role not in valid_roles:
            errors.append(f"Unsupported task role in {task_key}: {role}")
        if not task.get("type"):
            errors.append(f"Missing type in {task_key}")

    if check_assets:
        inputs = [parameters.get("indexPath"), parameters.get("queryFilePath")]
        for _, _, task in tasks:
            for key, value in task.items():
                if key in _PATH_KEYS and key not in {
                    "outputPath", "promptPath", "rag:promptPath",
                    "prf:expansionQueryFile", "bertrr:topPsgPath",
                    "ltr:trainingFeatureVectorsFile", "ltr:testingFeatureVectorsFile",
                    "ltr:testingDocumentScores", "ltr:modelFile",
                }:
                    inputs.append(value)
        for value in inputs:
            if value and not Path(str(value)).exists():
                errors.append(f"Input asset does not exist: {value}")
    return errors


def discover_configs(path: str | Path) -> Iterable[Path]:
    candidate = Path(path).expanduser().resolve()
    if candidate.is_file():
        yield candidate
    elif candidate.is_dir():
        yield from sorted(candidate.rglob("*.json"))
    else:
        raise ConfigError(f"Configuration path does not exist: {candidate}")


def public_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Return parameters without loader metadata."""
    result = deepcopy(parameters)
    result.pop("_configPath", None)
    return result
