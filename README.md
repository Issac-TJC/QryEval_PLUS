# QryEval_PLUS

QryEval_PLUS is a configurable information-retrieval experimentation framework. It combines Lucene-based sparse retrieval, FAISS dense retrieval, pseudo-relevance feedback, learning-to-rank, cross-encoder reranking, and retrieval-augmented answer generation behind one ordered pipeline. It now includes both the original fixed RAG pipeline and a bounded LangGraph single-agent retrieval loop.

## Project layout

```text
src/qryeval_plus/       Python package and command-line interface
configs/rag/            Reproducible passage, prompt, and system experiments
configs/agent/          BM25-only and full-tools Agentic RAG experiments
configs/benchmarks/     Fixed-vs-agent benchmark manifests
benchmarks/hw5/         Read-only historical reference snapshot and provenance
datasets/triviaqa/      Versioned queries and the small BM25 baseline run
data/                   Local indexes, models, evaluation data, and tools
outputs/                Generated runs, prompts, answers, and logs
tests/                  Unit and local integration tests
docs/                   Security and executability analysis
MIGRATION.md            Source-to-current migration ledger
```

Large local assets under `data/` and generated files under `outputs/` are not committed. The six required Lucene and RankLib archives are packaged under `src/qryeval_plus/vendor/java/`.

## Installation

The validated local environment is `11x42-26S-a`, with Python 3.9 and its JVM installed inside the environment. Activate it and install this project in editable mode:

```sh
conda activate 11x42-26S-a
python -m pip install -e .
qryeval doctor
```

The runtime uses `JAVA_HOME` when it is set. Otherwise it automatically detects a conda JVM at `sys.prefix/lib/jvm`, so a system-wide Java installation is not required.

The Agentic RAG runtime uses a separate Python 3.11 environment. Creating it does not modify the historical Python 3.9 environment:

```sh
conda env create -f environment.agent.yml
conda activate qryeval-agent-py311
qryeval doctor --config configs/agent/full_tools_agent.json
```

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

## License

See [LICENSE](LICENSE).
