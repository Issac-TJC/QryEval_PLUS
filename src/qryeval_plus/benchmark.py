"""Reproducible HW5 fixed-vs-agent benchmark runner."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import platform
import random
import re
import shutil
import string
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

from qryeval_plus.config import ConfigError, load_config, ordered_tasks, validate_config
from qryeval_plus.pipeline import run_pipeline
from qryeval_plus.statistics import (
    classify_error,
    holm_adjust,
    paired_bootstrap,
    paired_randomization_test,
    percentile,
)


def run_benchmark(
    manifest_path: str,
    *,
    limit: int | None = None,
    split: str | None = None,
    resume: bool = False,
):
    manifest_file = Path(manifest_path).expanduser().resolve()
    loaded_manifest = load_config(manifest_file)
    errors = validate_config(loaded_manifest, check_assets=True)
    if errors:
        raise ConfigError("; ".join(errors))
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    base = manifest_file.parent
    _verify_historical_reference(base, manifest)
    if split in {"test", "all"}:
        _verify_protocol_lock(base, manifest)
    query_source, split_label = _query_source(base, manifest, split)
    query_lines = query_source.read_text(encoding="utf-8").splitlines()
    limit = len(query_lines) if limit is None else int(limit)
    if limit <= 0 or limit > len(query_lines):
        raise ConfigError("--limit must be between 1 and {}.".format(len(query_lines)))

    output_label = split_label if split_label and limit == len(query_lines) else "questions-{}".format(limit)
    output_root = _path(base, manifest["outputRoot"]) / output_label
    output_root.mkdir(parents=True, exist_ok=True)
    query_path = output_root / "queries.qry"
    query_path.write_text("\n".join(query_lines[:limit]) + "\n", encoding="utf-8")
    qids = [_qid(line) for line in query_lines[:limit]]
    questions = {_qid(line): line.split(":", 1)[1].strip() for line in query_lines[:limit]}
    qrel_path = _filter_qrels(
        _path(base, manifest["qrelPath"]), output_root / "qrels.filtered", qids
    )

    systems = []
    for system in manifest["systems"]:
        if system.get("historical"):
            systems.append({**system, "metrics": system.get("metrics", {})})
            continue
        result = _run_system(
            system=system,
            base=base,
            output_root=output_root,
            query_path=query_path,
            qids=qids,
            qrel_path=qrel_path,
            gold_path=_path(base, manifest["goldPath"]),
            trec_eval_path=_path(base, manifest["trecEvalPath"]),
            resume=resume,
            pricing=manifest.get("pricing", {}),
        )
        systems.append(result)

    _write_comparison(output_root, systems)
    _write_per_question(output_root, systems, qids, questions)
    _write_error_analysis(output_root, systems, qids)
    _write_report(output_root, systems, qids, questions)
    _write_pareto(output_root, systems, len(qids))
    _write_environment(
        output_root,
        manifest_file,
        manifest,
        systems,
        query_source=query_source,
        split=split_label,
    )
    _write_artifact_checksums(output_root)
    return output_root


def estimate_budget(manifest_path: str, *, split: str | None = None, limit: int | None = None):
    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    query_source, label = _query_source(manifest_file.parent, manifest, split)
    count = len(query_source.read_text(encoding="utf-8").splitlines())
    count = count if limit is None else min(int(limit), count)
    systems = []
    total = 0
    for system in manifest.get("systems", []):
        if system.get("historical"):
            continue
        config = load_config(_path(manifest_file.parent, system["config"]))
        calls = 1
        for _, role, task in ordered_tasks(config):
            if role == "agent" and task.get("type") == "rewrite_rag":
                calls = 2
        maximum = calls * count
        total += maximum
        systems.append({"name": system["name"], "questions": count, "max_requests": maximum})
    return {"split": label, "questions": count, "systems": systems, "max_requests": total}


def publish_artifacts(output_root: str | Path, release_dir: str | Path) -> Path:
    """Copy only scrubbed, lightweight evidence from a completed benchmark."""
    source = Path(output_root).expanduser().resolve()
    target = Path(release_dir).expanduser().resolve()
    required = [
        "HW5_Agent_Comparison.csv", "HW5_AGENT_REPORT.md",
        "per_question_comparison.csv", "error_analysis.csv",
        "environment.json", "benchmark_snapshot.json",
    ]
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise ConfigError("Cannot publish incomplete benchmark: {}".format(", ".join(missing)))
    target.mkdir(parents=True, exist_ok=True)
    for name in required[:4]:
        shutil.copyfile(source / name, target / name)
    for name in ("pareto_points.csv", "pareto_quality_efficiency.svg"):
        if (source / name).is_file():
            shutil.copyfile(source / name, target / name)
    for name in required[4:]:
        payload = _redact_local_paths(json.loads((source / name).read_text(encoding="utf-8")))
        (target / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metrics = {}
    for path in sorted(source.glob("*/metrics.json")):
        metrics[path.parent.name] = json.loads(path.read_text(encoding="utf-8"))
    (target / "system_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_artifact_checksums(target)
    return target


def _redact_local_paths(value):
    if isinstance(value, dict):
        return {key: _redact_local_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_local_paths(item) for item in value]
    if isinstance(value, str) and value.startswith("/"):
        return "<LOCAL_ASSET>/" + Path(value).name
    return value


def _query_source(base, manifest, split):
    if split:
        splits = manifest.get("splits", {})
        if split not in splits:
            raise ConfigError("Unknown benchmark split '{}'.".format(split))
        return _path(base, splits[split]), split
    if not manifest.get("queryFilePath"):
        raise ConfigError("Benchmark requires queryFilePath when --split is omitted.")
    return _path(base, manifest["queryFilePath"]), None


def _verify_protocol_lock(base: Path, manifest: Dict[str, Any]) -> None:
    value = manifest.get("protocolLockPath")
    if not value:
        raise ConfigError("Locked test execution requires protocolLockPath.")
    lock_path = _path(base, value)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    root = (lock_path.parent / lock.get("root", ".")).resolve()
    mismatches = []
    for relative, expected in lock.get("files", {}).items():
        path = root / relative
        actual = _sha256(path) if path.is_file() else "missing"
        if actual != expected:
            mismatches.append(relative)
    if mismatches:
        raise ConfigError(
            "Frozen test protocol changed: {}. Run development again and intentionally regenerate the lock before test.".format(
                ", ".join(mismatches)
            )
        )


def _run_system(
    *, system, base, output_root, query_path, qids, qrel_path,
    gold_path, trec_eval_path, resume, pricing,
):
    name = system["name"]
    system_dir = output_root / name
    system_dir.mkdir(parents=True, exist_ok=True)
    run_path = system_dir / "results.run"
    answers_path = system_dir / "answers.json"
    metadata_path = system_dir / "llm.json"
    runtime_path = system_dir / "runtime.json"

    existing_metadata = _read_json(metadata_path, {})
    complete = (
        resume and run_path.is_file() and answers_path.is_file() and
        metadata_path.is_file() and runtime_path.is_file() and
        _answer_count(answers_path) == len(qids) and
        not _has_retryable_failures(existing_metadata, qids)
    )
    if not complete:
        parameters = load_config(_path(base, system["config"]))
        parameters["queryFilePath"] = str(query_path)
        for _, role, task in ordered_tasks(parameters):
            if role == "agent" and task["type"] in {"rag", "agentic_rag", "rewrite_rag"}:
                task["rag:cachePath"] = str(output_root.parent / "llm_cache.sqlite3")
                task["rag:cacheNamespace"] = output_root.name
                task["rag:inputPricePerMillion"] = float(pricing.get("inputPerMillionUsd", 0))
                task["rag:outputPricePerMillion"] = float(pricing.get("outputPerMillionUsd", 0))
            if role == "output" and task["type"] == "trec_eval":
                task["outputPath"] = str(run_path)
            elif role == "output" and task["type"] == "triviaqa_evaluation":
                task["outputPath"] = str(answers_path)
                task["metadataPath"] = str(metadata_path)
                task["promptPath"] = str(system_dir / "prompts.txt")
            elif role == "agent" and task["type"] in {"agentic_rag", "rewrite_rag"}:
                task["agent:trajectoryPath"] = str(system_dir / "trajectory.jsonl")
                task["agent:checkpointPath"] = str(system_dir / "checkpoint.jsonl")
                task["agent:resume"] = bool(resume)
                task["rag:continueOnError"] = True
            elif role == "agent" and task["type"] == "rag":
                task["rag:promptPath"] = str(system_dir / "prompts.txt")
                task["agent:trajectoryPath"] = str(system_dir / "trajectory.jsonl")
                task["agent:checkpointPath"] = str(system_dir / "checkpoint.jsonl")
                task["agent:resume"] = bool(resume)
        started = time.monotonic()
        run_pipeline(parameters)
        elapsed = time.monotonic() - started
        if resume and runtime_path.is_file():
            elapsed += float(_read_json(runtime_path, {}).get("seconds", 0) or 0)
        runtime_path.write_text(
            json.dumps({"seconds": elapsed}, indent=2), encoding="utf-8"
        )

    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))["seconds"]
    metrics = evaluate_outputs(
        answers_path=answers_path,
        run_path=run_path,
        gold_path=gold_path,
        qrel_path=qrel_path,
        trec_eval_path=trec_eval_path,
        qids=qids,
    )
    metrics["time_seconds"] = float(runtime)
    metadata = _read_json(metadata_path, {})
    metrics.update(_agent_metrics(metadata, qids))
    (system_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        **system,
        "metrics": metrics,
        "metadata": metadata,
        "system_dir": str(system_dir),
    }


def _has_retryable_failures(metadata, qids):
    for qid in qids:
        row = metadata.get(qid, {})
        if row.get("success") is False:
            return True
        if row.get("agent", {}).get("stop_reason") in {
            "planner_error", "answer_error",
        }:
            return True
    return False


def evaluate_outputs(*, answers_path, run_path, gold_path, qrel_path, trec_eval_path, qids):
    answers = json.loads(Path(answers_path).read_text(encoding="utf-8"))
    gold_data = json.loads(Path(gold_path).read_text(encoding="utf-8"))
    gold = {item["QuestionId"]: item["Answer"] for item in gold_data["Data"]}
    per_query = {}
    for qid in qids:
        prediction = str(answers.get(qid, ""))
        aliases = _ground_truths(gold[qid])
        per_query[qid] = {
            "prediction": prediction,
            "exact": float(max(_exact(prediction, item) for item in aliases)),
            "f1": max(_f1(prediction, item) for item in aliases),
        }
    retrieval, retrieval_by_query = _trec_metrics(
        trec_eval_path, qrel_path, run_path, qids
    )
    for qid in qids:
        per_query[qid].update(retrieval_by_query[qid])
    return {
        "exact": 100.0 * sum(item["exact"] for item in per_query.values()) / len(qids),
        "f1": 100.0 * sum(item["f1"] for item in per_query.values()) / len(qids),
        "MRR": retrieval.get("recip_rank", 0.0),
        "P@1": retrieval.get("P_1", 0.0),
        "P@5": retrieval.get("P_5", 0.0),
        "per_query": per_query,
    }


def _trec_metrics(executable, qrels, run, qids):
    result = subprocess.run(
        [
            str(executable), "-c", "-q", "-m", "recip_rank",
            "-m", "P.1,5", str(qrels), str(run),
        ],
        text=True, capture_output=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("trec_eval failed: {}".format(result.stderr.strip()))
    metrics = {}
    per_query = {
        qid: {"MRR": 0.0, "P@1": 0.0, "P@5": 0.0} for qid in qids
    }
    names = {"recip_rank": "MRR", "P_1": "P@1", "P_5": "P@5"}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        if parts[1] == "all":
            metrics[parts[0]] = float(parts[2])
        elif parts[1] in per_query and parts[0] in names:
            per_query[parts[1]][names[parts[0]]] = float(parts[2])
    return metrics, per_query


def _agent_metrics(data: Dict[str, Any], qids: List[str]):
    if not data:
        return {}
    records = [data.get(qid, {}) for qid in qids]
    agent_rows = [row.get("agent") for row in records]
    agent_rows = [row for row in agent_rows if isinstance(row, dict)]
    tool_counts = Counter()
    stop_reasons = Counter()
    for row in agent_rows:
        stop_reasons[str(row.get("stop_reason"))] += 1
        for name, count in row.get("tool_usage", {}).items():
            tool_counts[str(name)] += int(count)
    n = len(qids)
    token_totals = [_usage_total(row.get("usage", {})) for row in records]
    end_to_end = [
        float(row.get("latency", {}).get("total_seconds", row.get("duration_seconds", 0)) or 0)
        for row in records
    ]
    costs = [float(row.get("estimated_cost_usd", 0) or 0) for row in records]
    cache_hits = [int(row.get("cache_hits", 0) or 0) for row in records]
    rewrite_rows = [row.get("rewrite") for row in records if isinstance(row.get("rewrite"), dict)]
    return {
        "avg_turns": sum(int(row.get("turns", 0)) for row in agent_rows) / n,
        "avg_model_calls": sum(
            int(row.get("calls", 1 if row.get("success") is not None else 0))
            for row in records
        ) / n,
        "avg_tool_calls": sum(int(row.get("tool_calls", 0)) for row in agent_rows) / n,
        "avg_retrieval_calls": sum(
            int(row.get("retrieval_calls", 0)) for row in agent_rows
        ) / n,
        "avg_rerank_calls": sum(
            int(row.get("rerank_calls", 0)) for row in agent_rows
        ) / n,
        "avg_fusion_calls": sum(
            int(row.get("fusion_calls", 0)) for row in agent_rows
        ) / n,
        "avg_total_tokens": sum(token_totals) / n,
        "avg_llm_latency_seconds": sum(
            float(row.get("duration_seconds", 0) or 0) for row in records
        ) / n,
        "latency_p50_seconds": percentile(end_to_end, 0.50),
        "latency_p95_seconds": percentile(end_to_end, 0.95),
        "latency_p99_seconds": percentile(end_to_end, 0.99),
        "avg_estimated_cost_usd": sum(costs) / n,
        "total_estimated_cost_usd": sum(costs),
        "cache_hits": sum(cache_hits),
        "grounding_status": dict(Counter(str(row.get("grounding_status", "unknown")) for row in records)),
        "rewrite_rate": (
            sum(bool(row.get("rewritten_query")) for row in rewrite_rows) / len(rewrite_rows)
            if rewrite_rows else 0.0
        ),
        "tool_errors": sum(int(row.get("tool_errors", 0)) for row in agent_rows),
        "tool_error_rate": sum(
            int(row.get("tool_errors", 0)) > 0 for row in agent_rows
        ) / n,
        "budget_exhaustions": sum(
            int(row.get("budget_exhaustions", 0)) for row in agent_rows
        ),
        "budget_exhaustion_rate": sum(
            int(row.get("budget_exhaustions", 0)) > 0 for row in agent_rows
        ) / n,
        "max_step_rate": stop_reasons.get("max_steps", 0) / n,
        "stop_reasons": dict(stop_reasons),
        "tool_usage": dict(sorted(tool_counts.items())),
        "tool_usage_per_question": {
            name: count / n for name, count in sorted(tool_counts.items())
        },
        "empty_answer_rate": sum(
            data.get(qid, {}).get("success") is False
            or data.get(qid, {}).get("grounding_status") == "abstained"
            for qid in qids
        ) / n,
    }


def _usage_total(usage):
    total = usage.get("total_tokens")
    if total is not None:
        return float(total or 0)
    return float(usage.get("prompt_tokens", 0) or 0) + float(
        usage.get("completion_tokens", 0) or 0
    )


def _write_comparison(output_root: Path, systems: List[Dict[str, Any]]):
    path = output_root / "HW5_Agent_Comparison.csv"
    labels = [system["label"] for system in systems]
    rows = []
    for key, label, digits in [
        ("exact", "exact", 2), ("f1", "f1", 2), ("MRR", "MRR", 4),
        ("P@1", "P@1", 4), ("P@5", "P@5", 4),
    ]:
        rows.append([label] + [_fmt(system["metrics"].get(key), digits) for system in systems])
    rows.append(["time"] + [_duration(system["metrics"].get("time_seconds")) for system in systems])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Metric"] + labels)
        writer.writerows(rows)


def _write_per_question(output_root, systems, qids, questions):
    runnable = [system for system in systems if not system.get("historical")]
    baseline = next((item for item in runnable if item.get("baseline")), None)
    if not baseline:
        return
    baseline_rows = baseline["metrics"]["per_query"]
    fieldnames = ["qid", "question"]
    for system in runnable:
        prefix = system["name"]
        fieldnames.extend([
            prefix + "_answer", prefix + "_exact", prefix + "_f1",
            prefix + "_MRR", prefix + "_P@1", prefix + "_P@5",
        ])
        if system is not baseline:
            fieldnames.append(prefix + "_delta_f1")
    path = output_root / "per_question_comparison.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for qid in qids:
            row = {"qid": qid, "question": questions[qid]}
            for system in runnable:
                prefix = system["name"]
                item = system["metrics"]["per_query"][qid]
                row.update({
                    prefix + "_answer": item["prediction"],
                    prefix + "_exact": item["exact"],
                    prefix + "_f1": item["f1"],
                    prefix + "_MRR": item["MRR"],
                    prefix + "_P@1": item["P@1"],
                    prefix + "_P@5": item["P@5"],
                })
                if system is not baseline:
                    row[prefix + "_delta_f1"] = 100.0 * (
                        item["f1"] - baseline_rows[qid]["f1"]
                    )
            writer.writerow(row)


def _write_report(
    output_root: Path,
    systems: List[Dict[str, Any]],
    qids: List[str],
    questions: Dict[str, str],
):
    runnable = [system for system in systems if not system.get("historical")]
    baseline = next((item for item in runnable if item.get("baseline")), None)
    lines = [
        "# Agentic Retrieval comparison",
        "",
        "The historical column is descriptive. Agent lift is attributed only against the contemporaneous fixed baseline.",
        *(
            ["The historical values cover the original 40 questions; they are not a five-question smoke-test score."]
            if len(qids) != 40 else []
        ),
        "",
        "| System | EM | F1 | MRR | P@1 | P@5 | Time |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for system in systems:
        m = system["metrics"]
        lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            system["label"], _fmt(m.get("exact"), 2), _fmt(m.get("f1"), 2),
            _fmt(m.get("MRR"), 4), _fmt(m.get("P@1"), 4), _fmt(m.get("P@5"), 4),
            _duration(m.get("time_seconds")),
        ))
    if baseline:
        lines.extend(["", "## Paired F1 differences", ""])
        base_scores = baseline["metrics"].get("per_query", {})
        raw_pvalues = {}
        for system in runnable:
            if system is baseline:
                continue
            raw_pvalues[system["name"]] = paired_randomization_test(
                [100.0 * base_scores[qid]["f1"] for qid in qids],
                [100.0 * system["metrics"]["per_query"][qid]["f1"] for qid in qids],
            )
        adjusted_pvalues = holm_adjust(raw_pvalues)
        for system in runnable:
            if system is baseline:
                continue
            scores = system["metrics"].get("per_query", {})
            delta, low, high = paired_bootstrap(
                [100.0 * base_scores[qid]["f1"] for qid in qids],
                [100.0 * scores[qid]["f1"] for qid in qids],
            )
            em_delta = system["metrics"]["exact"] - baseline["metrics"]["exact"]
            if low > 0 and adjusted_pvalues[system["name"]] < 0.05:
                conclusion = "statistically supported F1 improvement"
            elif high < 0 and adjusted_pvalues[system["name"]] < 0.05:
                conclusion = "statistically supported F1 degradation"
            elif delta > 0:
                conclusion = "positive point estimate, but evidence is insufficient"
            elif delta < 0:
                conclusion = "negative point estimate, but evidence is insufficient"
            elif em_delta > 0:
                conclusion = "F1 tied; EM is higher"
            elif em_delta < 0:
                conclusion = "F1 tied; EM is lower"
            else:
                conclusion = "F1 and EM tied"
            lines.append(
                "- {}: ΔF1={:.2f}, ΔEM={:.2f}, 95% F1 CI [{:.2f}, {:.2f}], "
                "randomization p={:.4f}, Holm-adjusted p={:.4f} — {}.".format(
                    system["label"], delta, em_delta, low, high,
                    raw_pvalues[system["name"]], adjusted_pvalues[system["name"]], conclusion
                )
            )
        lines.extend(["", "## Baseline difficulty strata", ""])
        strata = {
            "relevant at rank 1": [qid for qid in qids if base_scores[qid]["MRR"] == 1.0],
            "relevant at ranks 2-10": [qid for qid in qids if 0.1 <= base_scores[qid]["MRR"] < 1.0],
            "first relevant below rank 10": [qid for qid in qids if 0.0 < base_scores[qid]["MRR"] < 0.1],
            "not retrieved in judged depth": [qid for qid in qids if base_scores[qid]["MRR"] == 0.0],
        }
        for label, selected in strata.items():
            pieces = []
            for system in runnable:
                if system is baseline or not selected:
                    continue
                scores = system["metrics"]["per_query"]
                delta = 100.0 * sum(
                    scores[qid]["f1"] - base_scores[qid]["f1"] for qid in selected
                ) / len(selected)
                pieces.append("{} ΔF1 {:+.2f}".format(system["label"], delta))
            lines.append("- {} (n={}): {}.".format(label, len(selected), "; ".join(pieces) or "n/a"))
        if len(qids) == 40:
            holdout = qids[5:]
            lines.extend(["", "## 35-question holdout", ""])
            for system in runnable:
                scores = system["metrics"].get("per_query", {})
                values = {
                    "EM": 100.0 * sum(scores[qid]["exact"] for qid in holdout) / len(holdout),
                    "F1": 100.0 * sum(scores[qid]["f1"] for qid in holdout) / len(holdout),
                    "MRR": sum(scores[qid]["MRR"] for qid in holdout) / len(holdout),
                    "P@1": sum(scores[qid]["P@1"] for qid in holdout) / len(holdout),
                    "P@5": sum(scores[qid]["P@5"] for qid in holdout) / len(holdout),
                }
                lines.append(
                    "- {}: EM {:.2f}, F1 {:.2f}, MRR {:.4f}, P@1 {:.4f}, P@5 {:.4f}.".format(
                        system["label"], values["EM"], values["F1"], values["MRR"],
                        values["P@1"], values["P@5"],
                    )
                )

        agents = [item for item in runnable if "avg_model_calls" in item["metrics"]]
        if agents:
            lines.extend([
                "", "## Agent operations", "",
                "| System | Model calls | Retrievals | Tokens | p50 (s) | p95 (s) | p99 (s) | Cost/query | Cache hits | Rewrite rate | Error rate |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ])
            for system in agents:
                m = system["metrics"]
                lines.append(
                    "| {} | {:.2f} | {:.2f} | {:.1f} | {:.2f} | {:.2f} | {:.2f} | ${:.6f} | {} | {:.2%} | {:.2%} |".format(
                        system["label"], m["avg_model_calls"], m["avg_retrieval_calls"],
                        m["avg_total_tokens"], m["latency_p50_seconds"],
                        m["latency_p95_seconds"], m["latency_p99_seconds"],
                        m["avg_estimated_cost_usd"],
                        m["cache_hits"], m["rewrite_rate"], m["tool_error_rate"],
                    )
                )
            lines.extend(["", "Tool-use totals:"])
            for system in agents:
                m = system["metrics"]
                lines.append("- {}: `{}`.".format(
                    system["label"], json.dumps(m["tool_usage"], sort_keys=True)
                ))

        lines.extend(["", "## Per-question gains and regressions", ""])
        base_scores = baseline["metrics"]["per_query"]
        for system in runnable:
            if system is baseline:
                continue
            scores = system["metrics"]["per_query"]
            ranked = sorted(
                qids,
                key=lambda qid: scores[qid]["f1"] - base_scores[qid]["f1"],
                reverse=True,
            )
            gains = [qid for qid in ranked if scores[qid]["f1"] > base_scores[qid]["f1"]][:15]
            losses = [qid for qid in reversed(ranked) if scores[qid]["f1"] < base_scores[qid]["f1"]][:15]
            lines.append("### {}".format(system["label"]))
            lines.append("")
            lines.extend(_comparison_examples("Largest gains", gains, questions, base_scores, scores))
            lines.extend(_comparison_examples("Largest regressions", losses, questions, base_scores, scores))

            failures = []
            metadata = system.get("metadata", {})
            for qid in qids:
                row = metadata.get(qid, {})
                agent = row.get("agent", {})
                if (not row.get("success")) or agent.get("errors"):
                    failures.append((qid, agent.get("stop_reason"), agent.get("errors", [])))
            lines.append("- Recorded failures: {}.".format(len(failures)))
            for qid, reason, errors in failures[:5]:
                detail = "; ".join(str(item) for item in errors)[:240]
                lines.append("  - `{}`: {}{}".format(
                    qid, reason or "unknown", " — " + detail if detail else ""
                ))
    (output_root / "HW5_AGENT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_error_analysis(output_root, systems, qids):
    runnable = [system for system in systems if not system.get("historical")]
    baseline = next((item for item in runnable if item.get("baseline")), None)
    if not baseline:
        return
    rows = []
    base_scores = baseline["metrics"]["per_query"]
    for system in runnable:
        if system is baseline:
            continue
        scores = system["metrics"]["per_query"]
        metadata = system.get("metadata", {})
        for qid in qids:
            row = metadata.get(qid, {})
            rewrite = row.get("rewrite", {})
            rows.append({
                "qid": qid,
                "system": system["name"],
                "category": classify_error(
                    baseline_f1=base_scores[qid]["f1"],
                    candidate_f1=scores[qid]["f1"],
                    baseline_mrr=base_scores[qid]["MRR"],
                    candidate_mrr=scores[qid]["MRR"],
                    rewritten=bool(rewrite.get("rewritten_query")),
                    grounding_status=row.get("grounding_status"),
                ),
                "delta_f1": 100.0 * (scores[qid]["f1"] - base_scores[qid]["f1"]),
                "baseline_mrr": base_scores[qid]["MRR"],
                "candidate_mrr": scores[qid]["MRR"],
                "rewritten": bool(rewrite.get("rewritten_query")),
                "grounding_status": row.get("grounding_status", "unknown"),
            })
    path = output_root / "error_analysis.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["qid"])
        writer.writeheader()
        writer.writerows(rows)


def _write_pareto(output_root: Path, systems: List[Dict[str, Any]], query_count: int):
    """Write dependency-free quality/efficiency evidence for the formal report."""
    points = []
    for system in systems:
        if system.get("historical"):
            continue
        metrics = system.get("metrics", {})
        points.append({
            "system": system["label"],
            "f1": float(metrics.get("f1", 0) or 0),
            "p95_latency_seconds": float(metrics.get("latency_p95_seconds", 0) or 0),
            "tokens_per_query": float(metrics.get("avg_total_tokens", 0) or 0),
            "cost_per_query_usd": float(metrics.get("avg_estimated_cost_usd", 0) or 0),
            "wall_seconds_per_query": float(metrics.get("time_seconds", 0) or 0) / max(query_count, 1),
        })
    if not points:
        return
    csv_path = output_root / "pareto_points.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)

    panels = [
        ("p95_latency_seconds", "p95 recorded latency (s)"),
        ("tokens_per_query", "tokens / query"),
        ("cost_per_query_usd", "estimated USD / query"),
    ]
    width, height, panel_width = 1200, 420, 400
    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">'.format(width, height, width, height),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#172033}.axis{stroke:#64748b;stroke-width:1}.frontier{fill:none;stroke:#0f766e;stroke-width:2}.point{fill:#2563eb;stroke:#fff;stroke-width:2}</style>',
    ]
    y_min = max(0.0, min(point["f1"] for point in points) - 5.0)
    y_max = min(100.0, max(point["f1"] for point in points) + 5.0)
    if y_max <= y_min:
        y_max = y_min + 1.0
    for index, (key, x_label) in enumerate(panels):
        left = index * panel_width + 58
        right = (index + 1) * panel_width - 24
        top, bottom = 42, 342
        x_max = max(point[key] for point in points) or 1.0

        def xy(point):
            x = left + (right - left) * point[key] / x_max
            y = bottom - (bottom - top) * (point["f1"] - y_min) / (y_max - y_min)
            return x, y

        frontier = [
            point for point in points
            if not any(
                other[key] <= point[key] and other["f1"] >= point["f1"]
                and (other[key] < point[key] or other["f1"] > point["f1"])
                for other in points if other is not point
            )
        ]
        frontier.sort(key=lambda point: point[key])
        svg.extend([
            '<text x="{}" y="24" font-size="16" font-weight="bold" text-anchor="middle">Quality vs {}</text>'.format((left + right) / 2, html.escape(x_label)),
            '<line class="axis" x1="{}" y1="{}" x2="{}" y2="{}"/>'.format(left, bottom, right, bottom),
            '<line class="axis" x1="{}" y1="{}" x2="{}" y2="{}"/>'.format(left, top, left, bottom),
            '<text x="{}" y="382" font-size="12" text-anchor="middle">{}</text>'.format((left + right) / 2, html.escape(x_label)),
            '<text x="{}" y="{}" font-size="12" text-anchor="middle" transform="rotate(-90 {} {})">answer F1</text>'.format(left - 42, (top + bottom) / 2, left - 42, (top + bottom) / 2),
            '<text x="{}" y="{}" font-size="10" text-anchor="end">{:.1f}</text>'.format(left - 6, top + 4, y_max),
            '<text x="{}" y="{}" font-size="10" text-anchor="end">{:.1f}</text>'.format(left - 6, bottom + 4, y_min),
            '<text x="{}" y="{}" font-size="10" text-anchor="end">{:.4g}</text>'.format(right, bottom + 16, x_max),
        ])
        if frontier:
            svg.append('<polyline class="frontier" points="{}"/>'.format(
                " ".join("{:.2f},{:.2f}".format(*xy(point)) for point in frontier)
            ))
        for point in points:
            x, y = xy(point)
            svg.append('<circle class="point" cx="{:.2f}" cy="{:.2f}" r="5"><title>{}: F1 {:.2f}, {} {:.6g}</title></circle>'.format(
                x, y, html.escape(point["system"]), point["f1"], html.escape(x_label), point[key]
            ))
            svg.append('<text x="{:.2f}" y="{:.2f}" font-size="10">{}</text>'.format(
                min(x + 7, right - 85), max(y - 7, top + 10), html.escape(point["system"])
            ))
    svg.append("</svg>")
    (output_root / "pareto_quality_efficiency.svg").write_text(
        "\n".join(svg) + "\n", encoding="utf-8"
    )


def _comparison_examples(label, selected, questions, baseline, candidate):
    if not selected:
        return ["- {}: none.".format(label)]
    rows = ["- {}:".format(label)]
    for qid in selected:
        delta = 100.0 * (candidate[qid]["f1"] - baseline[qid]["f1"])
        rows.append(
            "  - `{}` (ΔF1 {:+.2f}) {} — fixed=`{}`, agent=`{}`".format(
                qid, delta, questions[qid].replace("`", "'"),
                baseline[qid]["prediction"].replace("`", "'"),
                candidate[qid]["prediction"].replace("`", "'"),
            )
        )
    return rows


def _bootstrap_delta(baseline, candidate, qids, samples=10000):
    differences = [100.0 * (candidate[qid]["f1"] - baseline[qid]["f1"]) for qid in qids]
    mean = sum(differences) / len(differences)
    randomizer = random.Random(20260814)
    estimates = []
    for _ in range(samples):
        estimates.append(sum(randomizer.choice(differences) for _ in differences) / len(differences))
    estimates.sort()
    return mean, estimates[int(samples * 0.025)], estimates[int(samples * 0.975) - 1]


def _write_environment(
    output_root, manifest_file, manifest, systems, *, query_source=None, split=None
):
    versions = {}
    for module_name in ["numpy", "torch", "transformers", "pydantic", "langgraph"]:
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[module_name] = "missing"
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "git": _git_snapshot(manifest_file.parent),
    }
    (output_root / "environment.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    base = manifest_file.parent
    config_snapshots = []
    model_snapshots = []
    for system in systems:
        if system.get("historical"):
            continue
        config_path = _path(base, system["config"])
        parameters = load_config(config_path)
        config_snapshots.append({
            "system": system["name"],
            "path": str(config_path),
            "sha256": _sha256(config_path),
        })
        for task_name, role, task in ordered_tasks(parameters):
            if role == "agent":
                model_snapshots.append({
                    "system": system["name"],
                    "task": task_name,
                    "provider": task.get("rag:provider"),
                    "model": task.get("rag:model"),
                    "thinking": task.get("rag:thinking"),
                    "temperature": task.get("rag:temperature"),
                })
            for key in ("dense:modelPath", "rag:dense:modelPath", "bertrr:modelPath"):
                if task.get(key):
                    model_snapshots.append(
                        _local_model_snapshot(system["name"], task_name, key, task[key])
                    )
    input_keys = ["qrelPath", "goldPath", "trecEvalPath"]
    resolved_query_source = (
        Path(query_source).expanduser().resolve()
        if query_source is not None
        else _path(base, manifest["queryFilePath"])
    )
    snapshot = {
        "manifest_path": str(manifest_file),
        "manifest_sha256": _sha256(manifest_file),
        "limit": len((output_root / "queries.qry").read_text(encoding="utf-8").splitlines()),
        "split": split,
        "inputs": {
            "queryFilePath": {
                "path": str(resolved_query_source),
                "sha256": _sha256(resolved_query_source),
            },
            **{
                key: {
                    "path": str(_path(base, manifest[key])),
                    "sha256": _sha256(_path(base, manifest[key])),
                }
                for key in input_keys
            },
        },
        "system_configs": config_snapshots,
        "models": model_snapshots,
        "historical_reference": _historical_reference(base, manifest),
    }
    (output_root / "benchmark_snapshot.json").write_text(
        json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8"
    )


def _local_model_snapshot(system, task, key, value):
    path = Path(str(value)).expanduser().resolve()
    config_path = path / "config.json"
    return {
        "system": system,
        "task": task,
        "key": key,
        "path": str(path),
        "config_sha256": _sha256(config_path) if config_path.is_file() else None,
    }


def _historical_reference(base, manifest):
    value = manifest.get("historicalProvenancePath")
    if not value:
        return {}
    provenance = _path(base, value)
    return _read_json(provenance, {})


def _verify_historical_reference(base, manifest):
    reference_value = manifest.get("historicalReferencePath")
    provenance_value = manifest.get("historicalProvenancePath")
    if not reference_value or not provenance_value:
        return
    reference = _path(base, reference_value)
    provenance = _read_json(_path(base, provenance_value), {})
    expected = str(provenance.get("sha256", "")).strip().lower()
    actual = _sha256(reference)
    if not expected or actual != expected:
        raise ConfigError(
            "Historical reference SHA-256 mismatch: expected {}, got {}.".format(
                expected or "<missing>", actual
            )
        )


def _git_snapshot(start):
    try:
        root = subprocess.run(
            ["git", "-C", str(start), "rev-parse", "--show-toplevel"],
            text=True, capture_output=True, check=True,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            text=True, capture_output=True, check=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "-C", root, "status", "--porcelain"],
            text=True, capture_output=True, check=True,
        ).stdout.strip())
        return {"root": root, "commit": commit, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"root": None, "commit": None, "dirty": None}


def _filter_qrels(source: Path, target: Path, qids: Iterable[str]):
    allowed = set(qids)
    lines = [line for line in source.read_text(encoding="utf-8").splitlines()
             if line.split() and line.split()[0] in allowed]
    present = {line.split()[0] for line in lines}
    missing = sorted(allowed - present)
    if missing:
        raise ConfigError(
            "Filtered qrels are missing query ids: {}.".format(", ".join(missing))
        )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _ground_truths(answer):
    return answer["NormalizedAliases"] + [_normalize(item) for item in answer.get("HumanAnswers", [])]


def _normalize(value):
    value = str(value).replace("_", " ").lower()
    punctuation = set(string.punctuation + "‘’´`")
    value = "".join(char if char not in punctuation else " " for char in value)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split()).strip()


def _exact(prediction, truth):
    return _normalize(prediction) == _normalize(truth)


def _f1(prediction, truth):
    predicted = _normalize(prediction).split()
    expected = _normalize(truth).split()
    common = Counter(predicted) & Counter(expected)
    same = sum(common.values())
    if same == 0 or not predicted or not expected:
        return 0.0
    precision = same / len(predicted)
    recall = same / len(expected)
    return 2 * precision * recall / (precision + recall)


def _path(base, value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _qid(line):
    return line.split(":", 1)[0].strip()


def _answer_count(path):
    try:
        return len(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return 0


def _read_json(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return default


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_artifact_checksums(output_root):
    target = output_root / "artifact_checksums.sha256"
    files = sorted(
        path for path in output_root.rglob("*")
        if path.is_file() and path != target
    )
    lines = [
        "{}  {}".format(_sha256(path), path.relative_to(output_root))
        for path in files
    ]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value, digits):
    return "" if value is None else ("{:.%df}" % digits).format(float(value))


def _duration(value):
    if value is None:
        return ""
    seconds = int(round(float(value)))
    return "{:02d}:{:02d}".format(seconds // 60, seconds % 60)
