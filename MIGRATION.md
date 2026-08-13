# Migration ledger

## Provenance and policy

- Read-only source: `/Users/issactjc/dev/11642/QryEval`
- Migration target: `/Users/issactjc/dev/QryEval_PLUS`
- Migration date: 2026-08-13
- Scope: execution framework and 35 HW5 retrieval/RAG configurations
- Excluded from the product structure: HW1–HW4 orchestration scripts, generated reports, result tables, office documents, and historical outputs
- Legacy names appear only in this ledger so that the current project can be audited against the source precisely.

The source tree was inspected but never written. Before deleting the target-side `HW-backup/`, it contained 1,193 files and occupied approximately 281 MB. The original `/Users/issactjc/dev/11642` tree remains the recovery source.

## Runtime source mapping

| Source relationship | Current relationship |
| --- | --- |
| `QryEval.py` | `src/qryeval_plus/pipeline.py`, called by `src/qryeval_plus/cli.py` |
| `Idx.py`, `PyLu.py`, `InvList.py`, `Ranking.py`, `ScoreList.py`, `Timer.py`, `Util.py` | `src/qryeval_plus/core/` |
| `Qry*.py` | `src/qryeval_plus/query/` |
| `Ranker.py`, `DenseRanker.py`, `DenseEncoder.py`, `RetrievalModel*.py` | `src/qryeval_plus/retrieval/` |
| `Rewriter.py`, `RewriteWithPrf.py` | `src/qryeval_plus/rewrite/` |
| `Reranker.py`, `RerankWithLtr.py`, `RerankWithBERT*.py` | `src/qryeval_plus/rerank/` |
| `Agent.py`, `PassageBuilder.py`, `RagPrompt.py` | `src/qryeval_plus/rag/` |
| `Output.py`, `TeIn.py` | `src/qryeval_plus/io/` |
| Root-level script execution | Installed `qryeval` command or `python -m qryeval_plus` |
| Paths relative to shell cwd | Paths relative to the owning JSON configuration |
| `LIB_DIR/*.jar` | Packaged resources in `src/qryeval_plus/vendor/java/` |

Most source modules were package-qualified during the earlier reframing. This migration preserves those functional changes and adds configuration loading, deterministic task ordering, JVM discovery, output-directory creation, and guaranteed index cleanup.

## RAG configuration mapping

All source query files named `HW5-Exp-*.qry` were byte-identical. They map to `datasets/triviaqa/verified_wikipedia_dev.qry`.

### Passage experiments

| Source | Current configuration |
| --- | --- |
| `HW5-Exp-1.1a.param` | `configs/rag/passages/bm25_first_passage_050.json` |
| `HW5-Exp-1.1b.param` | `configs/rag/passages/bm25_first_passage_100.json` |
| `HW5-Exp-1.1c.param` | `configs/rag/passages/bm25_first_passage_150.json` |
| `HW5-Exp-1.1d.param` | `configs/rag/passages/bm25_first_passage_200.json` |
| `HW5-Exp-1.2a.param` | `configs/rag/passages/bm25_best6_passage_050.json` |
| `HW5-Exp-1.2b.param` | `configs/rag/passages/bm25_best6_passage_100.json` |
| `HW5-Exp-1.2c.param` | `configs/rag/passages/bm25_best6_passage_150.json` |
| `HW5-Exp-1.2d.param` | `configs/rag/passages/bm25_best6_passage_200.json` |
| `HW5-Exp-1.3a.param` | `configs/rag/passages/dense_first_passage_050.json` |
| `HW5-Exp-1.3b.param` | `configs/rag/passages/dense_first_passage_100.json` |
| `HW5-Exp-1.3c.param` | `configs/rag/passages/dense_first_passage_150.json` |
| `HW5-Exp-1.3d.param` | `configs/rag/passages/dense_first_passage_200.json` |
| `HW5-Exp-1.4a.param` | `configs/rag/passages/dense_best6_passage_050.json` |
| `HW5-Exp-1.4b.param` | `configs/rag/passages/dense_best6_passage_100.json` |
| `HW5-Exp-1.4c.param` | `configs/rag/passages/dense_best6_passage_150.json` |
| `HW5-Exp-1.4d.param` | `configs/rag/passages/dense_best6_passage_200.json` |

### Prompt experiments

| Source | Current configuration |
| --- | --- |
| `HW5-Exp-2.1a.param` | `configs/rag/prompts/bm25_prompt_01.json` |
| `HW5-Exp-2.1b.param` | `configs/rag/prompts/bm25_prompt_02.json` |
| `HW5-Exp-2.1c.param` | `configs/rag/prompts/bm25_prompt_03.json` |
| `HW5-Exp-2.1d.param` | `configs/rag/prompts/bm25_prompt_04.json` |
| `HW5-Exp-2.1e.param` | `configs/rag/prompts/bm25_prompt_05.json` |
| `HW5-Exp-2.1f.param` | `configs/rag/prompts/bm25_prompt_06.json` |
| `HW5-Exp-2.2a.param` | `configs/rag/prompts/dense_prompt_01.json` |
| `HW5-Exp-2.2b.param` | `configs/rag/prompts/dense_prompt_02.json` |
| `HW5-Exp-2.2c.param` | `configs/rag/prompts/dense_prompt_03.json` |
| `HW5-Exp-2.2d.param` | `configs/rag/prompts/dense_prompt_04.json` |
| `HW5-Exp-2.2e.param` | `configs/rag/prompts/dense_prompt_05.json` |
| `HW5-Exp-2.2f.param` | `configs/rag/prompts/dense_prompt_06.json` |

### System experiments

| Source | Current configuration |
| --- | --- |
| `HW5-Exp-3.1a.param` | `configs/rag/systems/bm25_baseline.json` |
| `HW5-Exp-3.1b.param` | `configs/rag/systems/dense_baseline.json` |
| `HW5-Exp-3.1c.param` | `configs/rag/systems/bm25_minilm_l6_rerank.json` |
| `HW5-Exp-3.1d.param` | `configs/rag/systems/dense_minilm_l6_rerank.json` |
| `HW5-Exp-3.1e.param` | `configs/rag/systems/dense_minilm_l12_rerank.json` |
| `HW5-Exp-3.1f.param` | `configs/rag/systems/dense_agent_depth_08.json` |
| `HW5-Exp-3.1g.param` | `configs/rag/systems/dense_grounded_prompt.json` |

## Asset mapping

| Source | Current path | Version-control policy |
| --- | --- | --- |
| `INPUT_DIR/index-cw22b-wp` | `data/indexes/clueweb22-lucene` | Local only |
| `INPUT_DIR/index-cw22b-wp-faiss-b300-Fp` | `data/indexes/clueweb22-faiss-b300-fp` | Local only |
| `INPUT_DIR/index-cw09` | `data/indexes/clueweb09-lucene` | Local only |
| `INPUT_DIR/co-condenser-marco-retriever` | `data/models/co-condenser-marco-retriever` | Local only |
| `INPUT_DIR/ms-marco-MiniLM-L-6-v2` | `data/models/ms-marco-minilm-l6-v2` | Local only |
| `INPUT_DIR/ms-marco-MiniLM-L-12-v2` | `data/models/ms-marco-minilm-l12-v2` | Local only |
| `INPUT_DIR/triviaqa_evaluation` and verified qrel | `data/evaluation/triviaqa` | Local only |
| `INPUT_DIR/trec_eval-9.0.4` | `data/tools/trec_eval` | Local only |
| `INPUT_DIR/svm_rank_*` | `data/tools/svm_rank_*` | Local only |
| `HW5/HW5-Exp-BM25.inRank` | `datasets/triviaqa/bm25_baseline.run` | Versioned fixture |

## Excluded historical material

- HW1 and HW2 shell/Python experiment drivers and their generated TREC output were not migrated; their retrieval capabilities remain in the core package.
- HW3 LTR experiment generators, CSV summaries, and report inputs were not migrated; RankLib/SVMRank support remains in `qryeval_plus.rerank`.
- HW4 neural-reranking experiment generators, reports, and CSV summaries were not migrated; the current BERT reranker and both local checkpoints remain available.
- HW5 report builders, office documents, HTML exports, shell runners, and generated result tables were not migrated. Their executable configurations are represented by the 35 JSON files above.

## Integrity evidence

| Source file | SHA-256 before migration |
| --- | --- |
| `QryEval.py` | `27aee9b25ce6690a58b8323fbb15f848006869becbea1386057c75d8845e33a1` |
| `Ranker.py` | `86ac91d0f3253f98f6bcc3fa1806baa9e29a10b95c8fd65f426abb3c4c72a5f8` |
| `DenseRanker.py` | `1c36398e2be5b27226e445e926b34958c9b1ec6a9c41fe9c1b66e4e1897865c3` |
| `RerankWithBERT.py` | `40e6245d86a92e839724370e077689ad257183430fb6e3a05159ee22cf9e7a59` |
| `Agent.py` | `6fae9b2f4cf408ae448a8443abe69fd4e56c636076d73d4b2a4fdadfe0822916` |
| `HW5/HW5-Exp-BM25.inRank` | `d801f5dfc2fc1cf7104445d82fe9872933d0482604f0422b55bd54976ff232a0` |
| `HW5-Exp-1.1a.qry` | `4dac3d3dad428a5b05be30d63cc83a4e27d2ee90c42c1f9c45e45931b95d0212` |

The two versioned dataset files retain the source hashes for the baseline and query set.

Verification commands:

```sh
qryeval validate configs/rag
qryeval doctor --config configs/rag/systems/dense_baseline.json
python -m pytest
QRYEVAL_RUN_INTEGRATION=1 python -m pytest -m integration
python -m pip wheel --no-build-isolation --no-deps .
```
