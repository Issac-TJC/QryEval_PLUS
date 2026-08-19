# Local production-like service

## Architecture and concurrency

FastAPI accepts concurrent requests and places cache misses in a bounded queue. One dedicated executor thread owns Lucene, the JVM-backed ranker, passage encoder, and response generation. This avoids unsafe concurrent access to the framework's global `Idx` state. Identical in-flight requests share one future; completed responses are cached in SQLite WAL mode.

For public, no-secret reproducibility, `configs/service/mock.json` uses the committed three-document `datasets/fixtures/corpus.jsonl` fixture and a deterministic mock answer engine. It exercises the same HTTP admission, queue, single-flight, cache, metrics, and shutdown surfaces without loading Lucene or calling an external model.

This design demonstrates admission control and observability on the available machine. It is not horizontal inference scaling. Multiple Uvicorn workers must not be enabled against the same in-process index state.

## API contract

`POST /v1/answer` accepts a 1–1000 character UTF-8 question, optional request ID, and `fixed_bm25` or `adaptive_rewrite` policy. Unknown fields and control characters are rejected. Responses contain the short answer, document/passage citations, grounding status, cache outcome, usage, per-stage latency, stop reason, and corpus version.

- Empty retrieval: HTTP 200 with an empty answer and `abstained`.
- Lexically unsupported answer: HTTP 200 with `needs_review` and citations.
- Queue or API budget exhausted: 429.
- Missing index/model or provider failure: 503.
- Upstream/request timeout: 504.
- No hidden answer or model fallback is permitted.

## Security and data handling

- The service binds to loopback by default and Docker publishes only loopback ports.
- Optional bearer authentication is configured by an environment-variable name; no token is stored in JSON.
- CORS is not enabled.
- Logs contain request ID and a 12-character question hash, not the question, passages, answer, or credentials.
- Lucene and model directories are mounted read-only; output caches use a separate writable mount.
- Plain HTTP model endpoints remain restricted to loopback by the provider implementation.

## Data updates and cache invalidation

Indexes are immutable snapshots. `qryeval corpus inspect` records document/vector counts, field statistics, byte sizes, and content hashes. Deployment configuration names the corpus version and expected document count. A new snapshot is activated by changing the mounted version and restarting the service. Corpus version and experiment hash are part of the response-cache key, so old entries cannot be served under a new corpus.

## Metrics and load-test interpretation

Prometheus records HTTP status, latency histograms, inference stage latency, queue depth, cache outcomes, grounding outcomes, and reported tokens. Grafana provisioning is in `deploy/grafana`.

Report three workloads separately:

1. Mock-provider infrastructure test at concurrency 1/4/8/16.
2. A 200-request hot-cache test for API/cache/single-flight throughput.
3. A small uncached real-provider test, explicitly identifying external model latency and cost.

Normal-capacity tests should produce no 5xx responses. Overload tests should produce controlled 429 responses rather than crashes. Never present hot-cache or mock throughput as uncached RAG throughput.

## Fault-injection checklist

- Provider timeout, 429, 500, and malformed response.
- Empty ranking and missing document fields.
- Wrong expected document count at startup.
- Corrupt SQLite response and LLM cache entries.
- Corpus version change and cache miss.
- Full queue, client cancellation, graceful shutdown, and benchmark checkpoint resume.

Automated tests cover the deterministic contracts. Docker and real-provider measurements require the local Docker runtime and credential and must be attached to the release report after execution.

The versioned [`mock_loadtest_report.json`](../benchmarks/triviaqa318/mock_loadtest_report.json) records the current local FastAPI/Uvicorn measurements. It labels the deterministic 10 ms mock engine, cache state, concurrency, status counts, QPS, p50/p95/p99, and the fact that Docker and real-provider tests were not run.
