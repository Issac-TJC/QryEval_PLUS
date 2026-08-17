# Agentic RAG upgrade and HW5 comparison

## Purpose before and after the upgrade

The original QryEval_PLUS project is an information-retrieval experiment runner. A JSON configuration fixes an ordered sequence such as `BM25 ranking → passage selection → Prompt 1 → answer`. Its purpose is controlled comparison of retrievers, passage settings, rerankers, and answer prompts. Once a run starts, every question follows the same predetermined path.

The upgraded project keeps that purpose and adds a second research question: can a bounded agent improve retrieval and answer quality by choosing what to do per question? The `agentic_rag` path may inspect the current evidence, rewrite the query, choose sparse or dense retrieval, rerank, fuse rankings, and select the evidence used for the answer. It is intentionally a single agent without long-term memory, web search, or human approval so that dynamic retrieval control is the principal changed variable.

The upgrade is additive. Existing `type: rag` configurations and the Python 3.9 environment remain valid; Agentic RAG uses the separate `qryeval-agent-py311` environment.

## Why the same experiment is valid

Both paths use the same 40 ordered TriviaQA questions, ClueWeb22 Lucene and FAISS assets, versioned BM25 run, local models, gold aliases, qrels, Prompt 1, output normalization, and metrics. This makes EM, token-overlap F1, MRR, P@1, P@5, and elapsed time directly comparable.

There are four reported systems:

1. Historical HW5 best baseline. This is a read-only descriptive reference from the original CSV.
2. A newly run fixed BM25 RAG baseline. This controls the current model endpoint, execution date, network, and post-processing.
3. A BM25-only Agent. This isolates planning and query rewriting without introducing a different retriever.
4. A full-tools Agent using BM25, CoCondenser + FAISS, MiniLM-L6 max-passage reranking, and RRF.

Only differences against system 2 can be attributed to the Agent in this experiment. Differences against the historical CSV are reported separately because they may also reflect model-service or runtime drift. The primary selection rule is F1 first and EM second. Paired bootstrap uses 10,000 samples and seed `20260814`; if the 95% interval crosses zero, the report calls the point change inconclusive.

The first five ordered questions form an engineering smoke test. They may be used only to repair protocol or runtime errors, not to tune against correct answers. The 40-question report also shows the remaining 35-question holdout separately.

## Bounded graph and tools

The LangGraph `StateGraph` follows this loop:

```text
planner → validated tool execution → evidence returned to planner
   ↑                                      ↓
   └──────── continue research ───────────┘
                         ↓
              finish or bounded stop
                         ↓
       Prompt 1 answer from selected ranking
```

The planner uses DeepSeek native function calling with thinking disabled and temperature zero. Pydantic rejects unknown fields and malformed arguments. The state records messages, every ranking, trajectory events, budgets, selected ranking, answer, usage, latency, errors, and stop reason.

Available tools are:

- `search_bm25(query)`: reuses the versioned HW5 BM25 run when the query is unchanged; rewritten queries use BM25 k1 1.2 and b 0.75 against Lucene.
- `search_dense(query)`: searches the existing CoCondenser FAISS index.
- `rerank(ranking_id, query)`: MiniLM-L6, depth 100, max-passage aggregation.
- `fuse_rankings(ranking_ids)`: reciprocal-rank fusion with k=60.
- `finish_research(ranking_id)`: selects the ranking used by the fixed Prompt 1 answer step.

Default limits are six planner turns, three retrievals, one rerank, one fusion, and five evidence passages per tool response. Passage selection is fixed at length 150, stride 140, six candidates per document, and five final documents. At a step limit, the last valid ranking is used and `max_steps` is recorded. Without a valid ranking, the answer and run are empty for that question; there is no hidden fixed-pipeline fallback.

## Running the experiment

From the repository root:

```sh
conda env create -f environment.agent.yml
conda activate qryeval-agent-py311
export DEEPSEEK_API_KEY="your-api-key"

qryeval validate configs --no-assets
qryeval doctor --config configs/agent/full_tools_agent.json
qryeval benchmark configs/benchmarks/hw5_agent.json --limit 5
qryeval benchmark configs/benchmarks/hw5_agent.json --limit 40 --resume
```

Do not put the API key in a JSON file or shell history that will be committed. An interrupted question is recovered from `checkpoint.jsonl`; completed questions are not sent to the API again, while questions recorded with a provider or answer error are retried by the next `--resume` run.

Each benchmark size writes under `outputs/benchmarks/hw5-agent/questions-N/`:

- `HW5_Agent_Comparison.csv` and `HW5_AGENT_REPORT.md`
- `per_question_comparison.csv`
- filtered qrels and the exact query subset
- per-system `results.run`, `answers.json`, `metrics.json`, `llm.json`, prompts, runtime, checkpoint, and `trajectory.jsonl`; both the fixed and Agent paths checkpoint after every completed question
- `environment.json` and `benchmark_snapshot.json` with non-secret versions, config hashes, input hashes, model identifiers, local model config hashes, and historical provenance

Agent metrics include model/tool/retrieval call averages, token usage, LLM latency, total elapsed time, tool/schema errors, max-step and empty-answer rates, stop reasons, and per-tool frequencies.

## Verification

Run all unit tests and the opt-in local integration suite:

```sh
python -m pytest
QRYEVAL_RUN_INTEGRATION=1 python -m pytest -m integration
```

The integration suite loads the real Lucene and FAISS indexes, CoCondenser and MiniLM checkpoints, and runs an Agentic BM25 graph with a deterministic mock LLM. The complete historical BM25 run is also checked against the filtered 40-query qrels and must reproduce MRR 0.6643, P@1 0.6000, and P@5 0.4150.
