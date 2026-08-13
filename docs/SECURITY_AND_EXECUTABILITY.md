# Security and executability review

Review date: 2026-08-13. This document records findings after structural migration. Findings are intentionally not remediated in this migration unless a change was strictly necessary for path-independent execution.

## Security findings

| Severity | Finding and evidence | Impact | Recommended remediation | Status |
| --- | --- | --- | --- | --- |
| Critical | `rag/RagPrompt.py` contains fallback authorization values in source. The values are redacted here. Authorization messages are also written to configured prompt files. | Repository readers or prompt-artifact readers may obtain reusable credentials. | Revoke the exposed values, require a secret provider or environment variables, and redact authorization messages before persistence. | Open; not changed in this migration. |
| High | `rag/Agent.py::send_to_llm` sends authorization and prompt content through an unauthenticated plaintext TCP socket. | Credentials, questions, and retrieved passages can be intercepted or modified. | Replace the custom protocol with authenticated TLS and server identity verification. | Open; not changed. |
| High | Configurations can select output paths and LTR executable paths; `rerank/RerankWithLtr.py` invokes configured external binaries. | An untrusted configuration can overwrite writable files or execute a supplied binary. | Treat configurations as trusted code, constrain writes to an approved output root, and allow-list or package executable tools. | Open; not changed. |
| Medium | The LLM socket has no explicit connect/read timeout or maximum total operation duration. | A stalled endpoint can block an experiment indefinitely. | Add configurable connect and read timeouts plus a bounded retry policy. | Open; not changed. |
| Medium | Indexes, model checkpoints, Java archives, evaluation code, and native tools are loaded without provenance or checksum enforcement. | Replaced local artifacts can execute code or silently alter results. | Publish an asset manifest with hashes/signatures and validate it in `doctor`. | Open; not changed. |
| Medium | The answer agent converts transport and authorization failures into heuristic answers. | Infrastructure failures can appear as valid experiment output and corrupt evaluation. | Record structured stage status, fail closed by default, and make heuristic fallback explicitly opt-in. | Open; not changed. |
| Low | Prompt and answer artifacts may contain retrieved document text and user questions. | Generated outputs can retain sensitive or licensed text. | Define retention rules, restrict file permissions, and add optional prompt redaction. | Open; not changed. |

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
- The RAG stage builds passages and prompts and returns an answer with a mocked LLM response.
- All 35 migrated configurations resolve independently of the caller's working directory.
- Unit tests and the opt-in local integration test pass in the environment above.

### Remaining limitations

- The real external LLM endpoint was deliberately not contacted. Authentication, availability, response compatibility, latency, and answer quality remain unverified.
- `trec_eval` and SVMRank executables are platform-specific local binaries; portability to Linux, Windows, or another CPU architecture is not guaranteed.
- Neural stages currently optimize for correctness over throughput. Dense encoding is performed one text at a time, and CPU-only cross-encoder runs can be slow for deep rankings.
- Dependency lower bounds in `pyproject.toml` describe compatibility intent but do not provide a fully locked environment. Exact reproducibility still depends on the recorded local versions or a future lock file.
- Large indexes and checkpoints are intentionally excluded from version control. A fresh clone requires restoring the asset layout described in `data/README.md`.

## Operational recommendation

Use `qryeval doctor --config <config>` before each experiment family. Treat all configurations and local assets as trusted input. Do not use the real answer-generation service or distribute prompt artifacts until the Critical and High findings are remediated.
