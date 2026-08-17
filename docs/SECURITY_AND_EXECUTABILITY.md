# Security and executability review

Review date: 2026-08-13. This document records the current state after the
structural migration and provider modernization.

## Security findings

| Severity | Finding and evidence | Impact | Recommended remediation | Status |
| --- | --- | --- | --- | --- |
| High | Questions and retrieved passages are sent to the configured external LLM provider. | Private queries or document text may leave the local machine and be retained under the provider's policy. | Use only approved providers, classify the corpus before use, and use a local endpoint for sensitive data. | Open; operational control required. |
| High | `openai-compatible` permits a user-supplied base URL and can optionally use HTTP or omit authentication for local services. | A malicious or mistakenly remote endpoint could receive prompts or API credentials. | Treat configurations as trusted input; permit keyless HTTP only on loopback addresses and use HTTPS with authentication remotely. | Open; required for local-provider flexibility. |
| High | Configurations can select output paths and LTR executable paths; `rerank/RerankWithLtr.py` invokes configured external binaries. | An untrusted configuration can overwrite writable files or execute a supplied binary. | Treat configurations as trusted code, constrain writes to an approved output root, and allow-list or package executable tools. | Open; not changed. |
| Medium | Indexes, model checkpoints, Java archives, evaluation code, and native tools are loaded without provenance or checksum enforcement. | Replaced local artifacts can execute code or silently alter results. | Publish an asset manifest with hashes/signatures and validate it in `doctor`. | Open; not changed. |
| Medium | API credentials are read from a configurable environment-variable name. | Credentials can still leak through shell history, process debugging, CI logs, or an unsafe local environment. | Use a secret manager in CI, never echo credentials, and rotate a key after suspected exposure. | Open; operational control required. |
| Low | Prompt and answer artifacts may contain retrieved document text and user questions. | Generated outputs can retain sensitive or licensed text. | Define retention rules, restrict file permissions, and add optional prompt redaction. | Open; not changed. |

The previous embedded authorization values, authorization prompt message,
plaintext custom transport, and implicit server address were removed from the
product code. LLM calls now have configurable timeouts and bounded retries.
Fallback is disabled by default and every returned query record carries
structured provider status.

## Executability findings

### Validated local environment

- Environment: `11x42-26S-a` (the originally supplied spelling did not match the case/order of the installed environment name).
- Python: 3.9.23.
- NumPy 1.26.4, PyTorch 2.0.0, Transformers 4.12.5, Hugging Face Hub 0.34.4, Pyjnius 1.6.1, FAISS 1.8.0, pytest 7.4.4.
- OpenJDK 20 is installed under the environment's `lib/jvm`, but its `java` executable is not linked into the environment's `bin` directory.
- Runtime discovery now checks `JAVA_HOME`, then `sys.prefix/lib/jvm`, then `PATH`. With the conda JVM, Pyjnius loads successfully.

### Verified behavior

- The packaged Java classpath loads all six required archives.
- The local Lucene collection opens successfully and reports 273,140 documents.
- A one-query dense search returns external document identifiers from FAISS/Lucene.
- The MiniLM-L6 cross-encoder reranks a retrieved document.
- The RAG stage builds standard chat messages and returns a normalized answer
  with structured provider metadata under a mocked HTTPS response.
- DeepSeek and generic OpenAI-compatible provider configuration is validated;
  API credentials are referenced by environment-variable name only.
- A manual DeepSeek end-to-end run completed successfully with no fallback.
  The five-query diagnostic and its retrieval-quality limitations are recorded
  in `docs/DEEPSEEK_VALIDATION.md`.
- All 35 migrated configurations resolve independently of the caller's working directory.
- Unit tests and the opt-in local integration test pass in the environment above.

### Remaining limitations

- A real paid LLM request is not part of the automated test suite. The manual
  validation confirms the tested account and model at one point in time, but
  future availability, latency, cost, and answer quality remain external state.
- `trec_eval` and SVMRank executables are platform-specific local binaries; portability to Linux, Windows, or another CPU architecture is not guaranteed.
- Neural stages currently optimize for correctness over throughput. Dense encoding is performed one text at a time, and CPU-only cross-encoder runs can be slow for deep rankings.
- Dependency lower bounds in `pyproject.toml` describe compatibility intent but do not provide a fully locked environment. Exact reproducibility still depends on the recorded local versions or a future lock file.
- Large indexes and checkpoints are intentionally excluded from version control. A fresh clone requires restoring the asset layout described in `data/README.md`.

## Operational recommendation

Use `qryeval doctor --config <config>` before each experiment family. Treat all configurations and local assets as trusted input. Do not use the real answer-generation service or distribute prompt artifacts until the Critical and High findings are remediated.
