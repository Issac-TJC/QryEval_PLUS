import pytest

from qryeval_plus.service.cache import ResponseCache


def test_response_cache_key_changes_with_corpus_config_and_policy(tmp_path):
    cache = ResponseCache(str(tmp_path / "responses.sqlite3"))
    base = dict(corpus_version="v1", config_hash="c1", policy="fixed_bm25", question="Question?")
    first = cache.key(**base)
    assert first != cache.key(**{**base, "corpus_version": "v2"})
    assert first != cache.key(**{**base, "config_hash": "c2"})
    assert first != cache.key(**{**base, "policy": "adaptive_rewrite"})
    cache.put(first, {"answer": "one"})
    assert cache.get(first) == {"answer": "one"}


def test_corrupt_response_cache_fails_closed(tmp_path):
    cache = ResponseCache(str(tmp_path / "responses.sqlite3"))
    cache._db.execute(
        "INSERT INTO responses(cache_key, response_json) VALUES ('broken', 'not-json')"
    )
    cache._db.commit()
    with pytest.raises(RuntimeError, match="corrupt"):
        cache.get("broken")
