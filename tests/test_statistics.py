import pytest

from qryeval_plus.statistics import (
    classify_error,
    holm_adjust,
    paired_bootstrap,
    paired_randomization_test,
    percentile,
)


def test_paired_statistics_are_deterministic_and_holm_is_monotonic():
    baseline = [0, 0, 1, 0, 1]
    candidate = [1, 1, 1, 1, 1]
    first = paired_bootstrap(baseline, candidate, samples=1000, seed=7)
    second = paired_bootstrap(baseline, candidate, samples=1000, seed=7)
    assert first == second
    assert first[0] == pytest.approx(0.6)
    assert 0 <= paired_randomization_test(baseline, candidate, samples=1000) <= 1
    adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.2})
    assert adjusted["a"] <= adjusted["b"] <= adjusted["c"]


def test_percentiles_and_error_taxonomy():
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert classify_error(
        baseline_f1=0, candidate_f1=1, baseline_mrr=0,
        candidate_mrr=1, rewritten=True, grounding_status="answered",
    ) == "beneficial_rewrite"
    assert classify_error(
        baseline_f1=1, candidate_f1=0, baseline_mrr=1,
        candidate_mrr=0, rewritten=True, grounding_status="needs_review",
    ) == "retrieval_miss"
    assert classify_error(
        baseline_f1=1, candidate_f1=0.5, baseline_mrr=1,
        candidate_mrr=1, rewritten=True, grounding_status="answered",
    ) == "possible_gold_alias_or_time_sensitive"
