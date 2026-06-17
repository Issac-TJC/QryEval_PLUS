# QryEval_PLUS

English | [中文](#qryeval_plus-中文)

QryEval_PLUS is a modular retrieval, reranking, and retrieval-augmented generation (RAG) question-answering framework. It organizes classic IR methods, neural retrieval, learning-to-rank, BERT reranking, and LLM-based answer generation into a configurable experiment pipeline.

## Features

- Configurable multi-stage pipeline: `ranker -> rewriter -> reranker -> agent -> output`
- Sparse retrieval over Lucene indexes, including Boolean models and BM25
- Pseudo relevance feedback (PRF) query rewriting with RM3-style expansion
- Learning-to-Rank reranking with RankLib/SVMRank-style feature vectors
- Dense retrieval with Hugging Face encoders and FAISS indexes
- BERT cross-encoder reranking with passage windows and score aggregation
- RAG agent stage for evidence selection, prompt construction, and TriviaQA-style answer output

## Project Structure

```text
QryEval_PLUS/
  QryEval.py              Compatibility entry point for running experiments
  helloHW5.py             Compatibility wrapper for the HW5 dense retrieval + RAG demo
  requirements.txt        Python package dependencies
  qryeval_plus/
    core/                 Index access, Lucene bridge, utilities, timers, ranking containers
    query/                Query parser and structured query operators
    retrieval/            First-stage sparse and dense rankers
    rewrite/              Query rewriting and pseudo relevance feedback
    rerank/               LTR and BERT reranking stages
    rag/                  RAG agent, prompt builder, passage builder
    io/                   TREC and QA output writers
    scripts/              Utility and demo scripts
  INPUT_DIR/              Local indexes, models, qrels, evaluation tools, and other large assets
  LIB_DIR/                Java/Lucene/RankLib jar dependencies
```

## Requirements

Recommended runtime:

- Python 3.9+
- Java JDK 8+ with `JAVA_HOME` configured
- Python packages listed in `requirements.txt`
- Java jars already provided under `LIB_DIR/`
- Local input assets under `INPUT_DIR/`; see [INPUT_DIR/INPUT_README.md](INPUT_DIR/INPUT_README.md)

Install Python dependencies in a virtual environment or conda environment:

```sh
python -m pip install -r requirements.txt
```

If FAISS installation through pip does not work on your platform, install it through conda-forge instead:

```sh
conda install -c conda-forge faiss-cpu
```

## Input Assets

Large indexes, model checkpoints, evaluation data, and external binaries are not committed to this repository. Recreate or download them according to [INPUT_DIR/INPUT_README.md](INPUT_DIR/INPUT_README.md).

At minimum, typical HW5/RAG experiments expect:

```text
INPUT_DIR/index-cw22b-wp/
INPUT_DIR/index-cw22b-wp-faiss-b300-Fp
INPUT_DIR/co-condenser-marco-retriever/
INPUT_DIR/triviaqa_evaluation/
```

Some experiments may also require `trec_eval`, SVMrank binaries, qrels, or additional reranker checkpoints.

## Usage

Run from the repository root:

```sh
python QryEval.py <paramFile>
```

Example:

```sh
python QryEval.py HW-backup/HW5-Exp-1.1a.param
```

The original HW5 demo command is preserved:

```sh
python helloHW5.py
```

Experiment behavior is controlled by JSON `.param` files. A typical configuration declares an index path, query file, and ordered `task_*` stages such as `ranker`, `rewriter`, `reranker`, `agent`, and `output`.

## Reproducibility Notes

- Run commands from the repository root so relative paths such as `INPUT_DIR/`, `OUTPUT_DIR/`, and `LIB_DIR/` resolve correctly.
- The Lucene bridge uses Pyjnius by default and loads Java classes from `LIB_DIR/*`.
- Dense retrieval and neural reranking require local Hugging Face model checkpoints or internet access during model download.
- Results can vary if model checkpoints, indexes, query sets, or reranker parameters differ from the original experiment setup.

## License

See [LICENSE](LICENSE).

---

# QryEval_PLUS 中文

QryEval_PLUS 是一个模块化的信息检索、重排序与检索增强生成（RAG）问答框架。项目将传统 IR 方法、神经检索、Learning-to-Rank、BERT 重排序和基于 LLM 的答案生成组织成可配置的实验 pipeline。

## 功能特性

- 可配置多阶段 pipeline：`ranker -> rewriter -> reranker -> agent -> output`
- 基于 Lucene 索引的稀疏检索，包括 Boolean 模型和 BM25
- 基于伪相关反馈（PRF）的查询改写，支持 RM3 风格扩展
- 基于 RankLib/SVMRank 风格特征向量的 Learning-to-Rank 重排序
- 基于 Hugging Face encoder 与 FAISS 索引的 dense retrieval
- 支持 passage window 和分数聚合的 BERT cross-encoder 重排序
- RAG agent 阶段，用于证据选择、prompt 构造和 TriviaQA 风格答案输出

## 项目结构

```text
QryEval_PLUS/
  QryEval.py              实验运行入口，保留原命令兼容性
  helloHW5.py             HW5 dense retrieval + RAG demo 兼容入口
  requirements.txt        Python 依赖列表
  qryeval_plus/
    core/                 索引访问、Lucene bridge、工具函数、计时器、ranking 容器
    query/                查询解析器和结构化查询算子
    retrieval/            第一阶段稀疏/稠密检索器
    rewrite/              查询改写和伪相关反馈
    rerank/               LTR 与 BERT 重排序
    rag/                  RAG agent、prompt 构造、passage 构造
    io/                   TREC 与 QA 输出模块
    scripts/              工具脚本和 demo 脚本
  INPUT_DIR/              本地索引、模型、qrels、评测工具和其他大文件资产
  LIB_DIR/                Java/Lucene/RankLib jar 依赖
```

## 环境要求

推荐运行环境：

- Python 3.9+
- Java JDK 8+，并正确配置 `JAVA_HOME`
- `requirements.txt` 中列出的 Python 包
- `LIB_DIR/` 中的 Java jar 依赖
- `INPUT_DIR/` 中的本地输入资产，详见 [INPUT_DIR/INPUT_README.md](INPUT_DIR/INPUT_README.md)

可以使用 virtualenv 或 conda 创建干净环境后安装依赖：

```sh
python -m pip install -r requirements.txt
```

如果你的平台无法通过 pip 安装 FAISS，建议使用 conda-forge：

```sh
conda install -c conda-forge faiss-cpu
```

## 输入资产

大型索引、模型 checkpoint、评测数据和外部二进制工具不会提交到仓库。请按照 [INPUT_DIR/INPUT_README.md](INPUT_DIR/INPUT_README.md) 下载或重新生成。

典型 HW5/RAG 实验至少需要：

```text
INPUT_DIR/index-cw22b-wp/
INPUT_DIR/index-cw22b-wp-faiss-b300-Fp
INPUT_DIR/co-condenser-marco-retriever/
INPUT_DIR/triviaqa_evaluation/
```

部分实验还可能需要 `trec_eval`、SVMrank 二进制文件、qrels 或额外 reranker checkpoint。

## 运行方式

请在仓库根目录运行：

```sh
python QryEval.py <paramFile>
```

示例：

```sh
python QryEval.py HW-backup/HW5-Exp-1.1a.param
```

原 HW5 demo 命令也保留：

```sh
python helloHW5.py
```

实验行为由 JSON `.param` 文件控制。通常配置包括索引路径、查询文件，以及按顺序执行的 `task_*` 阶段，例如 `ranker`、`rewriter`、`reranker`、`agent` 和 `output`。

## 复现说明

- 请从仓库根目录运行命令，确保 `INPUT_DIR/`、`OUTPUT_DIR/` 和 `LIB_DIR/` 等相对路径正确解析。
- Lucene bridge 默认使用 Pyjnius，并从 `LIB_DIR/*` 加载 Java class。
- Dense retrieval 和神经重排序需要本地 Hugging Face 模型 checkpoint，或在下载阶段具备网络访问。
- 如果模型 checkpoint、索引、查询集或 reranker 参数与原实验不同，结果可能存在差异。

## 许可证

请见 [LICENSE](LICENSE)。
