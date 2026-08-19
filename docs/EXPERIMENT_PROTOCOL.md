# TriviaQA-318 frozen experiment protocol

## Research question

With corpus, BM25 parameters, answer model, prompt, context depth, and request budget fixed, does evidence-conditioned selective query rewriting produce a better quality-cost trade-off than Fixed BM25, RM3 PRF, or unconditional LLM rewriting?

## Data and leakage policy

- The protocol contains 318 TriviaQA questions with gold aliases and qrels.
- The 40 previously inspected questions are development-only.
- The remaining 278 questions are the locked test set.
- Controller wording and all retrieval, passage, grounding, and budget parameters must be frozen after the development run.
- The test benchmark is run once. A failed infrastructure attempt may resume from checkpoints, but test answers must not be used for tuning.
- `benchmarks/triviaqa318/protocol_lock.json` freezes the four configs, split files, controller, retrieval, prompt, benchmark, and statistics code. Test/all execution fails closed if any listed hash changes.

Run `qryeval dataset validate` before every formal experiment. Query files and their hashes live in `datasets/triviaqa/protocol`; corpus and FAISS hashes live in `benchmarks/triviaqa318/corpus_manifest.json`.

## Systems and controlled variables

| System | Retrieval | Controller calls | Answer calls |
| --- | --- | ---: | ---: |
| Fixed BM25 | Original query once | 0 | 1 |
| BM25 + RM3 PRF | BM25, RM3 expansion, BM25 | 0 | 1 |
| Always Rewrite | Initial BM25, mandatory rewrite, second BM25 | 1 | 1 |
| Adaptive Rewrite | Initial BM25; finish or one rewrite and second BM25 | 1 | 1 |

Always and Adaptive use the same controller prompt. Adaptive's only extra action is `finish`; after an optional second retrieval both systems answer immediately. No system may silently fall back to another model or ranking.

The earlier LangGraph full-tools experiment remains development-only supplementary evidence. It is not part of the 278-question primary comparison.

## Metrics and decision rule

- Primary: answer token-overlap F1.
- Secondary: EM, MRR, P@1, P@5, p50/p95/p99 end-to-end latency, tokens/query, and estimated cost/query.
- Inference: 10,000-sample paired bootstrap 95% CI; 10,000-sample paired randomization tests; Holm correction across comparisons.
- Mechanism analysis: baseline relevant document at rank 1, ranks 2–10, or absent; query length, score margin, term coverage, rewrite similarity, and rewrite rate.
- Error labels: retrieval miss, generation/passage error, unsupported generation, beneficial/harmful/ineffective rewrite, retrieval-generation mismatch, and a conservative `possible_gold_alias_or_time_sensitive` review flag when a grounded answer regresses without worse retrieval.

The report may call a quality gain statistically supported only if the paired F1 interval excludes zero and the Holm-adjusted p-value is below 0.05. Otherwise it must say that the point estimate is inconclusive.

## Cost and artifacts

The shared SQLite request cache keys provider, model, exact messages, tools, tool choice, token limit, and corpus namespace. Formal runs default to at most 2,000 uncached requests and 10,000,000 reported tokens. Prices are supplied by the dated benchmark manifest rather than embedded in code.

Raw prompts, checkpoints, response caches, and full rankings remain under ignored `outputs/`. Versioned release evidence contains aggregate metrics, per-question scores, error labels, environment metadata, checksums, and no credentials.
