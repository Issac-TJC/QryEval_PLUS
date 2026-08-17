# DeepSeek end-to-end validation

Validation date: 2026-08-14. The test used the local `11x42-26S-a`
environment, the migrated Lucene and FAISS indexes, MiniLM-L6 reranking, and
the official DeepSeek Chat Completions API. The credential was supplied only
through the process environment and was not written to the repository or an
output artifact.

## Scope

The first five TriviaQA queries were run through:

```text
dense retrieval -> MiniLM-L6 reranking -> passage selection
-> RAG prompt -> deepseek-v4-flash -> TriviaQA evaluation
```

Two prompt policies were compared with thinking disabled, a 64-token answer
limit, fallback disabled, and temperature zero for the final stability check.
This is a small diagnostic sample, not a statistically sufficient benchmark.

## Results

| Policy | Exact Match | F1 | Semantic answers | Evidence-grounded behavior |
| --- | ---: | ---: | ---: | ---: |
| Open short-answer prompt | 40.00 | 69.33 | 4/5 acceptable | 2/5 answers explicitly supported |
| Strict grounded prompt | 20.00 | 33.33 | 2/5 answers plus 3 abstentions | 5/5 respected available evidence |

All 13 paid API requests made during smoke, batch, grounded, and repeatability
checks succeeded. No pipeline query used fallback. The two five-query batches
averaged approximately 0.95 seconds of LLM latency per query. A repeated
temperature-zero request produced identical responses twice; this is useful
evidence of improved determinism but is not a broad stability guarantee.

The open prompt returned these five answers:

```text
Quelch.
Potato crops.
The Abwehr.
Formula One.
Martin Ruane did.
```

The strict grounded prompt returned:

```text
Not enough information.
Potato crops.
Not enough information.
Formula One.
Not enough information.
```

## Diagnosis

The API transport and response parsing are operational. Answer reliability is
currently limited primarily by evidence quality:

- Only three of five queries had a judged-relevant document in the top five.
- The first query had no judged-relevant document in the produced run.
- The SMERSH query's first relevant document appeared at rank 30, outside the
  agent depth of five; the open prompt selected the distractor `Abwehr`.
- The Giant Haystacks query had a relevant document at rank four, but the
  selected passage did not contain `Martin Ruane`.
- The open prompt sometimes answered from model knowledge when the supplied
  context lacked the answer. Before temperature was fixed, the same first-query
  prompt produced both `Suter` and `Quelch` on separate calls.
- Strict Exact Match also penalized semantically correct additions such as
  `Potato crops.` and `Martin Ruane did.`. Prompt and normalization policy must
  therefore be held constant when comparing systems.

For trustworthy evaluation, use
`configs/examples/deepseek_grounded_rag.json`. It combines the strongest tested
retrieval/reranking path with prompt 3, which abstains when evidence is missing.
Improve retrieval recall and passage selection before interpreting a larger
LLM as an answer-quality fix.
