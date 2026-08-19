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
    "qrelPath",
    "goldPath",
    "trecEvalPath",
    "historicalReferencePath",
    "historicalProvenancePath",
    "outputRoot",
    "cachePath",
    "corpusManifestPath",
    "fixtureCorpusPath",
    "splitManifestPath",
    "experimentConfig",
    "protocolLockPath",
    "inRankFile:Path",
    "outputPath",
    "promptPath",
    "metadataPath",
    "agent:trajectoryPath",
    "agent:checkpointPath",
    "rag:cachePath",
    "agent:bm25InRankPath",
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
    if "systems" in parameters and "trecEvalPath" in parameters:
        return _validate_benchmark_manifest(parameters, check_assets)
    if "experimentConfig" in parameters:
        return _validate_service_config(parameters, check_assets)
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
        if role == "agent" and str(task.get("type", "")).lower() in {
            "rag", "agentic_rag", "rewrite_rag"
        }:
            try:
                from qryeval_plus.llm import provider_summary
                provider_summary(task)
            except (TypeError, ValueError) as exc:
                errors.append(f"Invalid LLM provider in {task_key}: {exc}")
        if role == "agent" and str(task.get("type", "")).lower() == "agentic_rag":
            errors.extend(_validate_agentic_task(task_key, task))
        if role == "agent" and str(task.get("type", "")).lower() == "rewrite_rag":
            policy = str(task.get("rewrite:policy", "")).lower()
            if policy not in {"always", "adaptive"}:
                errors.append(
                    "{} requires rewrite:policy 'always' or 'adaptive'.".format(task_key)
                )

    if check_assets:
        inputs = [parameters.get("indexPath"), parameters.get("queryFilePath")]
        for _, _, task in tasks:
            for key, value in task.items():
                if key in _PATH_KEYS and key not in {
                    "outputPath", "promptPath", "rag:promptPath",
                    "metadataPath", "agent:trajectoryPath", "agent:checkpointPath",
                    "rag:cachePath", "cachePath", "corpusManifestPath",
                    "prf:expansionQueryFile", "bertrr:topPsgPath",
                    "ltr:trainingFeatureVectorsFile", "ltr:testingFeatureVectorsFile",
                    "ltr:testingDocumentScores", "ltr:modelFile",
                }:
                    inputs.append(value)
        for value in inputs:
            if value and not Path(str(value)).exists():
                errors.append(f"Input asset does not exist: {value}")
    return errors


def _validate_benchmark_manifest(parameters, check_assets):
    errors = []
    systems = parameters.get("systems")
    if not isinstance(systems, list) or not systems:
        errors.append("Benchmark manifest requires a non-empty systems list.")
    else:
        base = Path(parameters.get("_configPath", ".")).parent
        for index, system in enumerate(systems):
            if not isinstance(system, dict) or not system.get("name"):
                errors.append("Benchmark system {} requires a name.".format(index))
                continue
            if system.get("historical"):
                if not isinstance(system.get("metrics"), dict):
                    errors.append("Historical system {} requires metrics.".format(index))
            elif not system.get("config"):
                errors.append("Benchmark system {} requires config.".format(index))
            elif check_assets:
                config_path = Path(str(system["config"])).expanduser()
                config_path = config_path if config_path.is_absolute() else (base / config_path).resolve()
                if not config_path.is_file():
                    errors.append("Input asset does not exist: {}".format(config_path))
    if check_assets:
        for key in (
            "queryFilePath", "qrelPath", "goldPath", "trecEvalPath",
            "historicalReferencePath", "historicalProvenancePath",
            "protocolLockPath",
        ):
            value = parameters.get(key)
            if key.startswith("historical") and not value:
                continue
            if key == "protocolLockPath" and not value:
                continue
            if not value or not Path(str(value)).exists():
                errors.append("Input asset does not exist: {}".format(value or key))
        for name, value in parameters.get("splits", {}).items():
            path = Path(str(value)).expanduser()
            path = path if path.is_absolute() else (Path(parameters.get("_configPath", ".")).parent / path).resolve()
            if not path.is_file():
                errors.append("Input asset does not exist for split {}: {}".format(name, path))
    return errors


def _validate_service_config(parameters, check_assets):
    errors = []
    try:
        if int(parameters.get("queueCapacity", 32)) <= 0:
            raise ValueError
    except (TypeError, ValueError):
        errors.append("Service queueCapacity must be a positive integer.")
    try:
        if float(parameters.get("requestTimeoutSeconds", 180)) <= 0:
            raise ValueError
    except (TypeError, ValueError):
        errors.append("Service requestTimeoutSeconds must be positive.")
    if not str(parameters.get("corpusVersion", "")).strip():
        errors.append("Service corpusVersion is required.")
    if check_assets:
        experiment = Path(str(parameters.get("experimentConfig", "")))
        if not experiment.is_file():
            errors.append("Input asset does not exist: {}".format(experiment))
        fixture = parameters.get("fixtureCorpusPath")
        if parameters.get("mock") and (not fixture or not Path(str(fixture)).is_file()):
            errors.append("Mock service fixture corpus does not exist: {}".format(fixture or "fixtureCorpusPath"))
    return errors


def _validate_agentic_task(task_key, task):
    errors = []
    known = {
        "search_bm25", "search_dense", "rerank",
        "fuse_rankings", "finish_research",
    }
    allowed = task.get("agent:allowedTools")
    if not isinstance(allowed, list) or not allowed:
        errors.append("{} requires a non-empty agent:allowedTools list.".format(task_key))
    else:
        unknown = sorted(set(str(item) for item in allowed) - known)
        if unknown:
            errors.append("{} has unknown tools: {}.".format(task_key, ", ".join(unknown)))
        if "finish_research" not in allowed:
            errors.append("{} must allow finish_research.".format(task_key))
    for key, minimum in (
        ("agent:maxTurns", 1), ("agent:maxRetrievalCalls", 1),
        ("agent:maxRerankCalls", 0), ("agent:maxFusionCalls", 0),
        ("agent:maxPlannerRetries", 0),
        ("agent:maxContextDocs", 1), ("agentDepth", 1),
    ):
        try:
            if int(task.get(key, minimum)) < minimum:
                raise ValueError
        except (TypeError, ValueError):
            errors.append("{} requires {} >= {}.".format(task_key, key, minimum))
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
