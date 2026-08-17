# Local data assets

This directory stores large or platform-specific assets that are intentionally excluded from version control. Experiment configurations refer to these semantic paths:

```text
data/
  indexes/
    clueweb22-lucene/
    clueweb22-faiss-b300-fp
    clueweb09-lucene/
  models/
    co-condenser-marco-retriever/
    ms-marco-minilm-l6-v2/
    ms-marco-minilm-l12-v2/
  evaluation/triviaqa/
    verified-wikipedia-dev.json
    verified-wikipedia-dev.qrel
    triviaqa_evaluation.py
  qrels/
    clueweb09-adhoc-1-200.qrel
  tools/
    trec_eval
    svm_rank_learn
    svm_rank_classify
```

The Lucene and FAISS indexes must describe the same document collection because FAISS internal IDs are converted to external IDs through Lucene. Model directories must be valid Hugging Face checkpoints. Evaluation and tool binaries are local dependencies and may be platform-specific.

Downloadable model sources:

- `Luyu/co-condenser-marco-retriever`
- `cross-encoder/ms-marco-MiniLM-L6-v2`
- `cross-encoder/ms-marco-MiniLM-L12-v2`

The index artifacts are collection-specific and must be copied or rebuilt separately. Run `qryeval doctor --config <config>` after restoring assets.
