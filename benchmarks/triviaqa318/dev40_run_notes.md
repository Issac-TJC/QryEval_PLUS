# HW5 Agentic RAG run notes

Date: 2026-08-14

## Protocol incidents and recovery

1. The first five-question smoke attempt exposed an `inRankFile` subset bug: the 40-query BM25 run attempted to populate qids outside the five-query batch. The ranker was changed to populate only active qids and to use an empty ranking for a missing active qid. No LLM request had been made when this attempt stopped.
2. The next smoke attempt hit the local network sandbox before the first answer response. It was rerun with approved external network access. No experimental answer from the failed attempt was retained.
3. The smoke trajectory showed that a parallel legal tool request reaching a frozen call budget was being classified as a tool error and prematurely finalizing. Budget rejection was separated from schema/tool errors, and the graph was allowed to use existing rankings for rerank, fusion, or finish. A model stop with a valid ranking was likewise classified as normal `model_finalize`; the model's direct text was ignored and the frozen Prompt 1 answer step remained authoritative.
4. During the formal BM25-Agent run, qid `qz_1032` received one malformed provider tool-call response. The run was stopped before the full-tools system. A single planner-response retry and retry-aware benchmark completion check were added. `--resume` reused the 39 valid BM25-Agent questions and reran only `qz_1032`, which completed successfully. The full-tools system then ran from the frozen configuration.

These changes repaired execution and classification behavior only. No prompts, answers, gold labels, retrieval parameters, tool budgets, evaluation metrics, or selection rules were tuned against correctness.

## Final acceptance

- All three contemporaneous systems contain 40 answers, 40 metadata records, and all 40 qids in their TREC runs.
- Empty-answer, provider-failure, Agent-error, schema/tool-error, and max-step rates are all zero.
- The fixed BM25 run reproduces historical retrieval metrics: MRR 0.6643, P@1 0.6000, and P@5 0.4150.
- No API-key pattern is present in the benchmark artifacts.
- `artifact_checksums.sha256` records the final result-file hashes.
