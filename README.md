# QryEval_PLUS

QryEval_PLUS is a modular retrieval, reranking, and RAG QA framework. It connects traditional information retrieval, neural retrieval, learning-to-rank, BERT reranking, and LLM-based answer generation in a configurable experiment pipeline.

## Pipeline

The project runs experiments as ordered stages declared in JSON `.param` files:

```text
ranker -> rewriter -> reranker -> agent -> output
```

- `ranker`: builds an initial ranking from BM25, Boolean retrieval, existing run files, or FAISS dense retrieval.
- `rewriter`: expands queries with pseudo relevance feedback, including RM3-style weighted queries.
- `reranker`: rescores top documents with Learning-to-Rank or BERT cross-encoder models.
- `agent`: performs retrieval-augmented answer generation by selecting evidence passages and calling an LLM server.
- `output`: writes TREC evaluation runs or TriviaQA-style answer files.

## Project Highlights

- Implements sparse retrieval with Boolean models and BM25 over Lucene indexes.
- Adds pseudo relevance feedback for query expansion.
- Provides a hot-pluggable Learning-to-Rank reranker with RankLib/SVMRank backends and 20 ranking features.
- Integrates HuggingFace dense encoders with FAISS for neural first-stage retrieval.
- Supports BERT cross-encoder reranking with passage windows and aggregation strategies.
- Includes a RAG agent stage that retrieves documents, selects passages, builds grounded prompts, and produces QA answers.

## Directory Structure

```text
qryeval_plus/
  core/       Index access, Lucene bridge, utilities, timers, rankings
  query/      Query parser and structured query operators
  retrieval/  First-stage sparse and dense rankers
  rewrite/    Query rewriting and pseudo relevance feedback
  rerank/     LTR and BERT reranking stages
  rag/        RAG agent, prompt builder, passage builder
  io/         TREC and QA output writers
  scripts/    Utility and demo scripts
```

Root-level `QryEval.py` and `helloHW5.py` are compatibility wrappers so the original commands still work.

## Usage

Run an experiment from the project root:

```sh
python QryEval.py <paramFile>
```

For example:

```sh
python QryEval.py HW-backup/HW5-Exp-1.1a.param
```

The HW5 demo wrapper is also preserved:

```sh
python helloHW5.py
```

## Input Assets

Large indexes, models, evaluation data, and external tools are intentionally kept out of git. See [INPUT_DIR/INPUT_README.md](INPUT_DIR/INPUT_README.md) for the expected input layout and download instructions.

## Resume Framing

This project can be presented as a retrieval-augmented QA system and evaluation framework. It demonstrates AI agent-relevant skills across retrieval orchestration, tool-style pipeline stages, evidence selection, prompt construction, ranking evaluation, and grounded answer generation.
