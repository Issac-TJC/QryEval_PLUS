"""Dependency-free paired statistics and latency summaries."""

from __future__ import annotations

import math
import random
from typing import Dict, Iterable, List, Sequence, Tuple


def paired_bootstrap(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    samples: int = 10_000,
    seed: int = 20260814,
) -> Tuple[float, float, float]:
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("Paired samples must be non-empty and equal length.")
    differences = [float(right) - float(left) for left, right in zip(baseline, candidate)]
    mean = sum(differences) / len(differences)
    rng = random.Random(seed)
    estimates = sorted(
        sum(rng.choice(differences) for _ in differences) / len(differences)
        for _ in range(samples)
    )
    return mean, estimates[int(samples * 0.025)], estimates[int(samples * 0.975) - 1]


def paired_randomization_test(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    samples: int = 10_000,
    seed: int = 20260814,
) -> float:
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("Paired samples must be non-empty and equal length.")
    differences = [float(right) - float(left) for left, right in zip(baseline, candidate)]
    observed = abs(sum(differences) / len(differences))
    rng = random.Random(seed)
    extreme = 0
    for _ in range(samples):
        estimate = abs(sum(value if rng.random() < 0.5 else -value for value in differences) / len(differences))
        extreme += estimate >= observed
    return (extreme + 1) / (samples + 1)


def holm_adjust(pvalues: Dict[str, float]) -> Dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    count = len(ordered)
    adjusted: Dict[str, float] = {}
    running = 0.0
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * float(value)))
        adjusted[name] = running
    return adjusted


def percentile(values: Iterable[float], quantile: float) -> float:
    rows = sorted(float(value) for value in values)
    if not rows:
        return 0.0
    if not 0 <= quantile <= 1:
        raise ValueError("Quantile must be between zero and one.")
    position = (len(rows) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return rows[lower]
    fraction = position - lower
    return rows[lower] * (1 - fraction) + rows[upper] * fraction


def classify_error(
    *,
    baseline_f1: float,
    candidate_f1: float,
    baseline_mrr: float,
    candidate_mrr: float,
    rewritten: bool,
    grounding_status: str | None,
) -> str:
    if candidate_mrr == 0:
        return "retrieval_miss"
    if candidate_mrr > 0 and candidate_f1 == 0:
        return "generation_or_passage_error"
    if grounding_status == "needs_review":
        return "unsupported_generation"
    if (
        grounding_status == "answered"
        and candidate_f1 < baseline_f1
        and candidate_mrr >= baseline_mrr
    ):
        return "possible_gold_alias_or_time_sensitive"
    if rewritten and candidate_f1 > baseline_f1:
        return "beneficial_rewrite"
    if rewritten and candidate_f1 < baseline_f1:
        return "harmful_rewrite"
    if rewritten and candidate_f1 == baseline_f1:
        return "ineffective_rewrite"
    if candidate_mrr > baseline_mrr and candidate_f1 < baseline_f1:
        return "retrieval_generation_mismatch"
    return "unchanged_or_other"
