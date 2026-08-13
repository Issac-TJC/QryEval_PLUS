# QryEval_PLUS

QryEval_PLUS is a configurable information-retrieval experimentation framework. It combines Lucene-based sparse retrieval, FAISS dense retrieval, pseudo-relevance feedback, learning-to-rank, cross-encoder reranking, and retrieval-augmented answer generation behind one ordered pipeline.

## Project layout

```text
src/qryeval_plus/       Python package and command-line interface
configs/rag/            Reproducible passage, prompt, and system experiments
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
qryeval doctor --config configs/rag/systems/dense_baseline.json
```

The runtime uses `JAVA_HOME` when it is set. Otherwise it automatically detects a conda JVM at `sys.prefix/lib/jvm`, so a system-wide Java installation is not required.

## Commands

Run an experiment from any working directory:

```sh
qryeval run /absolute/path/to/configs/rag/systems/dense_baseline.json
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
