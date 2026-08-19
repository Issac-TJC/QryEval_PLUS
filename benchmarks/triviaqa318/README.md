# TriviaQA-318 evidence bundle

This directory is intentionally versioned while raw experiment output is ignored.

- `corpus_manifest.json`: real mounted Lucene/FAISS counts, sizes, field statistics, and content hashes.
- `dev40_agentic_report.md`: the existing 40-question fixed-vs-multi-step-Agent development report.
- `dev40_comparison.csv`: aggregate development metrics.
- `dev40_per_question.csv`: per-question development metrics and answers.
- `mock_loadtest_report.json`: actual FastAPI/Uvicorn mock-provider uncached and hot-cache measurements, explicitly separated from real-provider evidence.
- `protocol_lock.json`: hashes that must match before the locked test or all split can execute.
- `artifact_checksums.sha256`: integrity hashes for every versioned evidence file in this directory.
- `test-release/`: scrubbed locked-test report, aggregate and per-question metrics, error labels, Pareto plot, environment/input snapshots, and its own checksums.

The 40-question Agent is development evidence, not the new Adaptive Rewrite ablation. The frozen 278-question test completed on 2026-08-19: all four systems produced 278 unique results with no recorded failures. Adaptive Rewrite reached F1 84.18 versus Fixed BM25 82.38, but its paired 95% CI `[-0.30, 4.03]` crosses zero; the defensible result is a positive point estimate and a better quality–cost tradeoff than unconditional rewriting, not statistically proven superiority. The release intentionally excludes API responses, prompt logs, SQLite caches, credentials, local asset paths, and full ranking files.
