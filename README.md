# QryEval_PLUS

QryEval_PLUS is a cost-aware, reproducible Agentic Retrieval evaluation and local serving platform for TriviaQA. Its frozen research question is: **when is one additional BM25 query rewrite worth its latency and token cost?** The repository combines controlled Fixed BM25, RM3 PRF, Always Rewrite, and Adaptive Rewrite experiments with a production-like FastAPI service.

## Evidence at a glance

| Evidence | Verified value |
| --- | ---: |
| TriviaQA protocol | 318 questions: 40 development + 278 locked test |
| Lucene corpus | 273,140 documents, 1.78 GB |
| Dense index | 273,140 vectors, 839 MB |
| Passage policy | Up to 6 dynamic candidates per retrieved document; no precomputed chunk index |
| Locked test quality | Fixed BM25 F1 82.38; Adaptive Rewrite F1 84.18, ΔF1 +1.79, 95% CI `[-0.30, 4.03]` |
| Adaptive efficiency | 44.96% rewrite rate; 7.89s p50 / 13.86s p95; 3,072 tokens/query; $0.000294/query |
| Mock uncached service | 54.3 QPS / 20.1 ms p95 at c=1; 68.0 QPS / 233.8 ms p95 at c=16 |
| Mock hot-cache service | 2,311 QPS / 1.8 ms p95 at c=4 |
| Test integrity | 278/278 questions in every system; 0 recorded failures; paired bootstrap/randomization with 10,000 samples |

The locked test result supports a quality–cost claim, not a significance claim: Adaptive Rewrite has nearly the same point-estimate F1 as Always Rewrite (84.18 vs. 84.17) while using fewer second retrievals, lower p95 latency (13.86s vs. 16.07s), and 32.5% lower estimated cost/query. Its improvement over Fixed BM25 is not statistically significant after paired testing and Holm correction. Fixed/PRF latency below measures the recorded answer stage; Rewrite latency is end-to-end controller, retrieval, and answer time, so those columns are not a direct retrieval-overhead comparison. Load figures use a deterministic 10 ms mock inference engine and are not real RAG throughput. Versioned test evidence is in [`benchmarks/triviaqa318/test-release`](benchmarks/triviaqa318/test-release).

Locked-test cases make the mechanism and failure boundary concrete:

| Outcome | Question | Fixed answer | Agent answer |
| --- | --- | --- | --- |
| Gain, +100 F1 | Cathedral known as “The Ship of the Fens” | Peterborough | Ely |
| Gain, +100 F1 | Edinburgh dog that watched its owner's grave | No answer in context | Greyfriars Bobby |
| Regression, −100 F1 | Choreographer of Hot Gossip | Arlene Phillips | Flick Colby |
| Regression, −42.9 F1 | Only team to win the Premier League exactly once | Blackburn Rovers | Leicester City |

The complete automatically generated gain/regression lists, paired statistics, difficulty strata, and per-question metrics are in the release report. These examples are diagnostic rather than cherry-picked evidence for statistical significance.

```text
question -> BM25 -> controller: finish ---------> grounded short answer
                         `-> rewrite -> BM25 ----^  (maximum two retrievals)
```

## Project layout

```text
src/qryeval_plus/       Python package and command-line interface
configs/rag/            Reproducible passage, prompt, and system experiments
configs/agent/          BM25-only and full-tools Agentic RAG experiments
configs/triviaqa318/    Fixed/PRF/Always/Adaptive controlled ablations
configs/service/        Local API runtime configuration
configs/benchmarks/     Fixed-vs-agent benchmark manifests
benchmarks/hw5/         Read-only historical reference snapshot and provenance
benchmarks/triviaqa318/ Versioned lightweight evidence and corpus manifest
datasets/triviaqa/      Versioned queries and the small BM25 baseline run
datasets/fixtures/      Three-document public fixture corpus for API/load tests
data/                   Local indexes, models, evaluation data, and tools
outputs/                Generated runs, prompts, answers, and logs
tests/                  Unit and local integration tests
docs/                   Security and executability analysis
MIGRATION.md            Source-to-current migration ledger
```

Large local assets under `data/` and generated files under `outputs/` are not committed. The six required Lucene and RankLib archives are packaged under `src/qryeval_plus/vendor/java/`.
The mock service reads the committed three-document fixture corpus, so API, queue, cache, metrics, and load-test workflows run without private indexes or paid model calls. Full experiments mount and checksum the local ClueWeb22, FAISS, and model assets.

## Installation

The validated local environment is `11x42-26S-a`, with Python 3.9 and its JVM installed inside the environment. Activate it and install this project in editable mode:

```sh
conda activate 11x42-26S-a
python -m pip install -e .
qryeval doctor
```

The runtime uses `JAVA_HOME` when it is set. Otherwise it automatically detects a conda JVM at `sys.prefix/lib/jvm`, so a system-wide Java installation is not required.

The Agentic RAG and HTTP service use a separate Python 3.11 environment. Creating it does not modify the historical Python 3.9 environment:

```sh
conda env create -f environment.agent.yml
conda activate qryeval-agent-py311
qryeval doctor --config configs/agent/full_tools_agent.json
```

The environment installs the `agent`, `serve`, and `test` optional dependencies.

## Commands

Run an experiment from any working directory:

```sh
qryeval run /absolute/path/to/configs/rag/systems/dense_baseline.json
qryeval run /absolute/path/to/configs/agent/bm25_agent.json
```

All relative paths inside a configuration are resolved against that configuration file, not the shell working directory.

Validate one configuration or the complete configuration tree:

```sh
qryeval validate configs/rag
qryeval validate configs/rag --no-assets
```

Inspect the environment and referenced assets:

```sh
qryeval doctor --config configs/rag/systems/dense_minilm_l6_rerank.json
```

Preview or run the compact demo:

```sh
qryeval demo --dry-run
qryeval demo --questions 1
```

Every command is also available through `python -m qryeval_plus`.

Run the frozen HW5 comparison first as a five-question engineering gate, then as the complete 40-question experiment:

```sh
export DEEPSEEK_API_KEY="your-api-key"
qryeval benchmark configs/benchmarks/hw5_agent.json --limit 5
qryeval benchmark configs/benchmarks/hw5_agent.json --limit 40 --resume
```

`--resume` reuses completed Agent checkpoints after an interrupted API run. The key is read only from the environment and is never written to configs, prompts, trajectories, logs, or benchmark snapshots.

Prepare and validate the 318-question protocol, then inspect the mounted corpus:

```sh
qryeval dataset validate \
  --manifest datasets/triviaqa/protocol/split_manifest.json \
  --gold data/evaluation/triviaqa/verified-wikipedia-dev.json \
  --qrels data/evaluation/triviaqa/verified-wikipedia-dev.qrel
qryeval corpus inspect --config configs/agent/full_tools_agent.json
```

Estimate paid requests without calling an API, tune only on development data, and run the locked test once:

```sh
qryeval benchmark configs/benchmarks/triviaqa318.json --split dev --dry-run-budget
qryeval benchmark configs/benchmarks/triviaqa318.json --split dev --resume
qryeval benchmark configs/benchmarks/triviaqa318.json --split test --resume
qryeval benchmark configs/benchmarks/triviaqa318.json --split test --resume \
  --publish-to benchmarks/triviaqa318/test-release
```

The benchmark enforces 2,000 uncached requests and 10,000,000 tokens by default. Exact requests are cached in SQLite; hitting a limit stops the run without switching models or generating fallback answers. The manifest stores a dated provider-price snapshot, and cost accounting conservatively applies the cache-miss input price to every external request.
`--publish-to` copies only lightweight reports, metrics, error labels, scrubbed environment metadata, and checksums; prompts, caches, checkpoints, and full rankings remain ignored.

## Local service

Start the API directly or with Docker Compose. Large indexes and models are mounted read-only and are never copied into the image.

```sh
export DEEPSEEK_API_KEY="your-api-key"
qryeval serve --config configs/service/local.json

curl -s http://127.0.0.1:8000/v1/answer \
  -H 'Content-Type: application/json' \
  -d '{"question":"Which agency had a name meaning death to spies?","policy":"adaptive_rewrite"}'
```

The service exposes `/health/live`, `/health/ready`, `/metrics`, and `/v1/answer`. It uses concurrent HTTP admission, a bounded queue, one JVM/index inference worker, SQLite response caching, duplicate-request coalescing, explicit abstention, evidence citations, and `answered`/`needs_review`/`abstained` grounding status. Queue overflow returns 429; upstream timeouts return 504; unavailable indexes or providers return 503.

```sh
docker compose up --build
qryeval loadtest --url http://127.0.0.1:8000 \
  --questions datasets/triviaqa/protocol/dev.qry \
  --concurrency 1 4 8 16 --requests 200 \
  --output outputs/service/loadtest.json
```

Prometheus is bound to `127.0.0.1:9090` and the provisioned Grafana dashboard to `127.0.0.1:3000`. See [`docs/PRODUCTION_SERVICE.md`](docs/PRODUCTION_SERVICE.md) for security, overload, cache-versioning, and fault-injection behavior.

## LLM providers

The answer-generation stage uses HTTPS Chat Completions providers. The 35
versioned experiments are configured for DeepSeek and read the credential only
from the environment:

```sh
export DEEPSEEK_API_KEY="your-api-key"
qryeval doctor --config configs/examples/deepseek_grounded_rag.json
qryeval demo --config configs/examples/deepseek_grounded_rag.json --questions 1
```

The default experiment settings use `deepseek-v4-flash`, disable thinking for
short TriviaQA answers, set temperature to zero, limit generation to 64 tokens,
and fail the experiment if the API call fails. No credential is stored in a JSON
file or prompt artifact.
Each run also writes a `.llm.json` sidecar containing non-secret provider,
model, request, latency, token-usage, success, and fallback metadata.

Other OpenAI-compatible services can be selected in an agent task:

```json
{
  "rag:provider": "openai-compatible",
  "rag:baseUrl": "http://127.0.0.1:11434/v1",
  "rag:model": "local-model",
  "rag:apiKeyRequired": false
}
```

See [docs/LLM_PROVIDERS.md](docs/LLM_PROVIDERS.md) for the full configuration
contract and validation procedure. See
[docs/DEEPSEEK_VALIDATION.md](docs/DEEPSEEK_VALIDATION.md) for the first real
API reliability test and its evidence-quality diagnosis.

## Pipeline

Configurations contain numbered `task_<number>:<role>` objects. Tasks are executed by numeric order and exchange a batch of query records:

```text
queries
  -> ranker (inRank / Boolean / BM25 / dense FAISS)
  -> optional PRF rewriter and second ranker
  -> optional LTR or BERT reranker
  -> TREC run output
  -> optional RAG agent
  -> answer and prompt output
```

The 35 migrated RAG configurations are organized by intent:

- `configs/rag/passages/`: retriever, passage length, and passage-selection experiments.
- `configs/rag/prompts/`: six prompt variants for sparse and dense retrieval.
- `configs/rag/systems/`: baseline, neural reranking, agent-depth, and grounded-prompt systems.

The fixed pipeline answers from one predetermined ranking. The new `agentic_rag` task lets the planner inspect evidence and choose bounded query rewrites, BM25 or dense retrieval, MiniLM reranking, reciprocal-rank fusion, and a final ranking. Both systems use the same Prompt 1 answer generator, questions, indexes, gold answers, qrels, and evaluation code, so the contemporaneous comparison changes retrieval control rather than the answer protocol. See [docs/AGENTIC_RAG.md](docs/AGENTIC_RAG.md) for the purpose, graph, budgets, artifacts, and attribution rules.

## Development and verification

```sh
python -m pytest
python -m pip wheel --no-build-isolation --no-deps .
```

Local model/index integration tests are opt-in:

```sh
QRYEVAL_RUN_INTEGRATION=1 python -m pytest -m integration
```

See [MIGRATION.md](MIGRATION.md) for provenance and [docs/SECURITY_AND_EXECUTABILITY.md](docs/SECURITY_AND_EXECUTABILITY.md) for known risks and runtime limitations.

## Claim boundaries

This is a single-dataset, single-machine evaluation and serving project. It has no evidence of real users, online A/B testing, cross-domain generalization, horizontal inference scaling, or cloud production operation. Mock-provider and hot-cache load tests must be reported separately from uncached real-provider measurements. These boundaries are deliberate and must not be removed from portfolio descriptions.

## License

See [LICENSE](LICENSE).
