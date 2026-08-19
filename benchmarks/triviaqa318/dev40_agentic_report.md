# HW5 Agentic RAG comparison

The historical column is descriptive. Agent lift is attributed only against the contemporaneous fixed baseline.

| System | EM | F1 | MRR | P@1 | P@5 | Time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Historical HW5 Baseline | 40.00 | 54.17 | 0.6643 | 0.6000 | 0.4150 | 08:35 |
| Reproduced Fixed BM25 RAG | 67.50 | 80.50 | 0.6643 | 0.6000 | 0.4150 | 02:21 |
| LangGraph BM25 Agent | 72.50 | 85.24 | 0.7951 | 0.7500 | 0.5100 | 08:26 |
| LangGraph Full-tools Agent | 72.50 | 83.89 | 0.7080 | 0.6500 | 0.4400 | 15:09 |

## Paired F1 differences

- LangGraph BM25 Agent: ΔF1=4.74, ΔEM=5.00, 95% F1 CI [-3.58, 13.75] — positive point estimate, but evidence is insufficient.
- LangGraph Full-tools Agent: ΔF1=3.39, ΔEM=5.00, 95% F1 CI [-5.36, 12.50] — positive point estimate, but evidence is insufficient.

## 35-question holdout

- Reproduced Fixed BM25 RAG: EM 68.57, F1 81.53, MRR 0.6711, P@1 0.6000, P@5 0.4171.
- LangGraph BM25 Agent: EM 71.43, F1 84.08, MRR 0.7944, P@1 0.7429, P@5 0.5143.
- LangGraph Full-tools Agent: EM 71.43, F1 82.54, MRR 0.7186, P@1 0.6571, P@5 0.4571.

## Agent operations

| System | Model calls | Tool calls | Retrievals | Tokens | LLM latency (s) | Tool error rate | Budget-limit rate | Max-step rate | Empty rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LangGraph BM25 Agent | 3.95 | 2.83 | 1.85 | 8194.2 | 6.07 | 0.00% | 10.00% | 0.00% | 0.00% |
| LangGraph Full-tools Agent | 4.40 | 4.47 | 2.48 | 15019.5 | 8.47 | 0.00% | 32.50% | 0.00% | 0.00% |

Tool-use totals:
- LangGraph BM25 Agent: `{"finish_research": 35, "search_bm25": 78}`.
- LangGraph Full-tools Agent: `{"finish_research": 32, "fuse_rankings": 16, "rerank": 19, "search_bm25": 60, "search_dense": 52}`.

## Per-question gains and regressions

### LangGraph BM25 Agent

- Largest gains:
  - `dpql_2585` (ΔF1 +100.00) Which counter-intelligence agency had a name meaning 'death to spies' in English? — fixed=`The Counter Intelligence Corps (CIC).`, agent=`SMERSH`
  - `sfq_21572` (ΔF1 +100.00) Which cricketing nation was first granted Test status in 2000? — fixed=`Ireland`, agent=`Bangladesh.`
  - `dpql_5685` (ΔF1 +50.00) Which aperitif is named for the Paris chemist who created it in 1846? — fixed=`The aperitif is Dubonnet.`, agent=`Dubonnet`
- Largest regressions:
  - `odql_921` (ΔF1 -91.30) Which British daily newspaper is published in the Berliner format? — fixed=`The Guardian.`, agent=`The Observer is a British Sunday newspaper, not a daily. No British daily newspaper is mentioned in the passage as being published in the Berliner format.`
- Recorded failures: 0.
### LangGraph Full-tools Agent

- Largest gains:
  - `dpql_2585` (ΔF1 +100.00) Which counter-intelligence agency had a name meaning 'death to spies' in English? — fixed=`The Counter Intelligence Corps (CIC).`, agent=`SMERSH`
  - `sfq_21572` (ΔF1 +100.00) Which cricketing nation was first granted Test status in 2000? — fixed=`Ireland`, agent=`Bangladesh.`
  - `dpql_5685` (ΔF1 +50.00) Which aperitif is named for the Paris chemist who created it in 1846? — fixed=`The aperitif is Dubonnet.`, agent=`Dubonnet`
- Largest regressions:
  - `odql_921` (ΔF1 -100.00) Which British daily newspaper is published in the Berliner format? — fixed=`The Guardian.`, agent=`The Independent.`
  - `sfq_6038` (ΔF1 -14.48) Nigel Farage is the leader of which political party? — fixed=`UK Independence Party (UKIP) and Brexit Party (Reform UK).`, agent=`Nigel Farage is the leader of the Brexit Party (renamed Reform UK in 2021).`
- Recorded failures: 0.
