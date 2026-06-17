# QryEval_PLUS

This repository expects several large or externally maintained assets under
`INPUT_DIR/`. Those files are intentionally not committed: they are either
downloadable from upstream projects, course-provided data, or generated indexes.

Create the directory after cloning:

```sh
mkdir -p INPUT_DIR
```

## Required input layout

`helloHW5.py` currently expects these paths:

```text
INPUT_DIR/index-cw22b-wp/
INPUT_DIR/index-cw22b-wp-faiss-b300-Fp
INPUT_DIR/co-condenser-marco-retriever/
```

`index-cw22b-wp/` is a Lucene index and `index-cw22b-wp-faiss-b300-Fp` is a
matching FAISS dense-vector index. These appear to be course/local generated
artifacts rather than public model checkpoints. Recreate them from the course
corpus and indexing pipeline, or copy them from the course-provided input
package, keeping the same filenames.

## Downloadable models

Install the Hugging Face CLI if needed:

```sh
python -m pip install -U huggingface_hub
```

Download the dense retriever used by `helloHW5.py`:

```sh
huggingface-cli download Luyu/co-condenser-marco-retriever \
  --local-dir INPUT_DIR/co-condenser-marco-retriever \
  --local-dir-use-symlinks False
```

Source: https://huggingface.co/Luyu/co-condenser-marco-retriever

Optional reranker checkpoints that have appeared in local `INPUT_DIR` copies:

```sh
huggingface-cli download cross-encoder/ms-marco-MiniLM-L6-v2 \
  --local-dir INPUT_DIR/ms-marco-MiniLM-L-6-v2 \
  --local-dir-use-symlinks False

huggingface-cli download cross-encoder/ms-marco-MiniLM-L12-v2 \
  --local-dir INPUT_DIR/ms-marco-MiniLM-L-12-v2 \
  --local-dir-use-symlinks False
```

Sources:

- https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2
- https://huggingface.co/cross-encoder/ms-marco-MiniLM-L12-v2

## Evaluation data and tools

TriviaQA data and evaluation code:

- Official data page: https://nlp.cs.washington.edu/triviaqa/
- Official code: https://github.com/mandarjoshi90/triviaqa

If an experiment expects these local files, download TriviaQA v1.0 from the
official page and copy or generate the needed files under:

```text
INPUT_DIR/triviaqa_evaluation/
INPUT_DIR/verified-wikipedia-dev.qrel
```

`verified-wikipedia-dev.qrel` is not part of the standard TriviaQA download; it
is a qrel-style conversion used by this project/course setup. Regenerate it from
the verified Wikipedia dev split if your experiment needs it.

TREC evaluation:

- Source: https://github.com/usnistgov/trec_eval

Build from source and place the compiled executable where legacy parameters
expect it:

```sh
git clone https://github.com/usnistgov/trec_eval /tmp/trec_eval
make -C /tmp/trec_eval
cp /tmp/trec_eval/trec_eval INPUT_DIR/trec_eval-9.0.4
```

SVMrank binaries:

- Source: https://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html

Download or build platform-appropriate binaries, then place them at:

```text
INPUT_DIR/svm_rank_learn
INPUT_DIR/svm_rank_classify
```

## Ignored local assets

The repo ignores `INPUT_DIR/*` so large downloaded files, generated indexes,
compiled tools, and local course data do not get committed. Keep only source
code, parameter files, and documentation in git.
