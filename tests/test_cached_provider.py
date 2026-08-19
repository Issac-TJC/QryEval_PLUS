import pytest

from qryeval_plus.llm import BudgetExceeded, CachedBudgetProvider, LLMResponse


class CountingProvider:
    name = "mock"
    model = "counting"

    def __init__(self):
        self.calls = 0

    def complete(self, messages, **kwargs):
        self.calls += 1
        return LLMResponse(
            content="answer",
            provider=self.name,
            model=self.model,
            duration_seconds=0.01,
            usage={"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10},
        )


def test_cached_provider_reuses_exact_request_and_persists_budget(tmp_path):
    source = CountingProvider()
    cache_path = tmp_path / "llm.sqlite3"
    provider = CachedBudgetProvider(
        source,
        cache_path=str(cache_path),
        namespace="test",
        max_requests=2,
        max_tokens=100,
        input_price_per_million=1.0,
        output_price_per_million=2.0,
    )
    first = provider.generate([{"role": "user", "content": "question"}])
    second = provider.generate([{"role": "user", "content": "question"}])
    assert source.calls == 1
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.estimated_cost_usd == 0.0
    assert provider.budget_snapshot()["requests"] == 1
    assert provider.budget_snapshot()["tokens"] == 10
    assert first.estimated_cost_usd == pytest.approx(0.000012)


def test_cached_provider_stops_before_request_limit(tmp_path):
    source = CountingProvider()
    provider = CachedBudgetProvider(
        source, cache_path=str(tmp_path / "llm.sqlite3"), namespace="test",
        max_requests=1, max_tokens=100,
    )
    provider.generate([{"role": "user", "content": "one"}])
    with pytest.raises(BudgetExceeded, match="request budget"):
        provider.generate([{"role": "user", "content": "two"}])
    assert source.calls == 1


def test_corrupt_cached_response_is_not_silently_ignored(tmp_path):
    source = CountingProvider()
    provider = CachedBudgetProvider(
        source, cache_path=str(tmp_path / "llm.sqlite3"), namespace="test",
    )
    provider.generate([{"role": "user", "content": "one"}])
    provider._db.execute("UPDATE llm_cache SET response_json = 'not-json'")
    provider._db.commit()
    with pytest.raises(Exception, match="corrupt"):
        provider.generate([{"role": "user", "content": "one"}])
