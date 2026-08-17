"""
Rerank documents with a BERT cross-encoder.
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

import os
import re

from qryeval_plus.core.Idx import Idx
from qryeval_plus.query.QryParser import QryParser


class _BertRuntime:
    """
    Cache tokenizer/model instances and provide batched scoring.
    """

    _cache = {}
    _threads_initialized = False
    _cpu_threads = max(1, min(4, os.cpu_count() or 1))

    def __init__(self, model_path, max_seq_length):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "BERT reranking requires the Python packages `transformers` and `torch`."
            ) from exc

        if not _BertRuntime._threads_initialized:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            torch.set_num_threads(self._cpu_threads)
            _BertRuntime._threads_initialized = True

        cache_key = (str(model_path), int(max_seq_length))
        cached = self._cache.get(cache_key)

        if cached is None:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=1,
            )
            model.eval()
            cached = (tokenizer, model, torch)
            self._cache[cache_key] = cached

        self._tokenizer, self._model, self._torch = cached
        self._max_seq_length = int(max_seq_length)

    def score_pairs(self, query_text, passages):
        if not passages:
            return []

        encoded = self._tokenizer(
            [query_text] * len(passages),
            list(passages),
            add_special_tokens=True,
            max_length=self._max_seq_length,
            truncation="only_second",
            padding=True,
            return_tensors="pt",
        )

        with self._torch.no_grad():
            outputs = self._model(**encoded)

        return outputs.logits.squeeze(-1).detach().cpu().tolist()


class RerankWithBERT:
    """
    Re-rank an existing ranking with a HuggingFace sequence classifier.
    """

    _VALID_AGGREGATIONS = {"firstp", "avgp", "maxp"}
    _KNOWN_FIELDS = {"body", "title", "url", "inlink", "keywords"}
    _OPERATOR_RE = re.compile(r"#[A-Za-z0-9_]+")
    _WEIGHT_RE = re.compile(r"(?<![A-Za-z])[-+]?\d*\.?\d+(?![A-Za-z])")

    def __init__(self, parameters):
        self._model_path = parameters.get("bertrr:modelPath")
        self._max_seq_length = int(parameters.get("bertrr:maxSeqLength", 512))
        self._psg_len = int(parameters.get("bertrr:psgLen", 0))
        self._psg_cnt = int(parameters.get("bertrr:psgCnt", 1))
        self._psg_stride = int(parameters.get("bertrr:psgStride", self._psg_len))
        self._max_title_length = int(parameters.get("bertrr:maxTitleLength", 0))
        self._score_aggregation = str(
            parameters.get("bertrr:scoreAggregation", "firstp")
        ).lower()
        self._batch_size = int(parameters.get("bertrr:batchSize", 16))
        self._top_psg_path = parameters.get("bertrr:topPsgPath")

        self._validate_parameters()
        self._runtime = _BertRuntime(self._model_path, self._max_seq_length)

    def _validate_parameters(self):
        if not self._model_path:
            raise AttributeError("Missing parameter: bertrr:modelPath")
        if self._psg_len <= 0:
            raise AttributeError("bertrr:psgLen must be > 0")
        if self._max_seq_length <= 0:
            raise AttributeError("bertrr:maxSeqLength must be > 0")
        if self._psg_stride <= 0:
            raise AttributeError("bertrr:psgStride must be > 0")
        if self._psg_cnt <= 0:
            raise AttributeError("bertrr:psgCnt must be > 0")
        if self._max_title_length < 0:
            raise AttributeError("bertrr:maxTitleLength must be >= 0")
        if self._batch_size <= 0:
            raise AttributeError("bertrr:batchSize must be > 0")
        if self._score_aggregation not in self._VALID_AGGREGATIONS:
            raise AttributeError(
                "bertrr:scoreAggregation must be one of {firstp, avgp, maxp}"
            )

    @staticmethod
    def _normalize_spaces(text):
        return " ".join(str(text).split())

    @classmethod
    def _remove_field_suffixes(cls, text):
        cleaned = []

        for token in str(text).split():
            head, sep, tail = token.rpartition(".")
            if sep and tail.lower() in cls._KNOWN_FIELDS:
                token = head
            cleaned.append(token)

        return " ".join(cleaned)

    @classmethod
    def _query_to_text(cls, qstring):
        query_text = str(qstring).strip()

        if "#" in query_text:
            try:
                query_text = QryParser.bowQuery(query_text)
            except Exception:
                query_text = cls._OPERATOR_RE.sub(" ", query_text)
                query_text = query_text.replace("(", " ").replace(")", " ")
                if "#wsum" in str(qstring).lower():
                    query_text = cls._WEIGHT_RE.sub(" ", query_text)

        query_text = cls._remove_field_suffixes(query_text)
        return cls._normalize_spaces(query_text)

    @staticmethod
    def _text_to_tokens(text, limit=None):
        if text is None:
            return []

        tokens = str(text).split()
        if limit is not None and limit >= 0:
            return tokens[:limit]
        return tokens

    @staticmethod
    def _aggregate(scores, method):
        if not scores:
            return float("-inf")
        if method == "firstp":
            return scores[0]
        if method == "avgp":
            return sum(scores) / float(len(scores))
        if method == "maxp":
            return max(scores)
        raise ValueError("Unknown aggregation method: {}".format(method))

    def _body_windows(self, body_tokens):
        if not body_tokens:
            return []

        windows = []
        start = 0

        while start < len(body_tokens) and len(windows) < self._psg_cnt:
            end = min(start + self._psg_len, len(body_tokens))
            windows.append(body_tokens[start:end])
            if end >= len(body_tokens):
                break
            start += self._psg_stride

        return windows

    def _build_passages(self, internal_docid):
        title_tokens = []
        if self._max_title_length > 0:
            title_tokens = self._text_to_tokens(
                Idx.getAttribute("title-string", internal_docid),
                self._max_title_length,
            )

        body_tokens = self._text_to_tokens(
            Idx.getAttribute("body-string", internal_docid),
            None,
        )
        body_windows = self._body_windows(body_tokens)

        if not body_windows:
            if not title_tokens:
                return []
            return [self._normalize_spaces(" ".join(title_tokens))]

        title_prefix = " ".join(title_tokens)
        passages = []

        for window in body_windows:
            body_text = " ".join(window)
            if title_prefix and body_text:
                passage = "{} {}".format(title_prefix, body_text)
            elif title_prefix:
                passage = title_prefix
            else:
                passage = body_text
            passages.append(self._normalize_spaces(passage))

        return passages

    def _score_passages(self, query_text, passages):
        scored = []

        for start in range(0, len(passages), self._batch_size):
            batch = passages[start:start + self._batch_size]
            scores = self._runtime.score_pairs(query_text, batch)
            scored.extend(zip(scores, batch))

        return scored

    def _score_document(self, query_text, external_docid):
        internal_docid = Idx.getInternalDocid(external_docid)
        if internal_docid is None:
            return float("-inf"), ""

        passages = self._build_passages(internal_docid)
        if not passages:
            return float("-inf"), ""

        scored_passages = self._score_passages(query_text, passages)
        if not scored_passages:
            return float("-inf"), ""

        scores = [score for score, _ in scored_passages]
        if self._score_aggregation == "firstp":
            best_passage = passages[0]
        else:
            best_passage = max(scored_passages, key=lambda item: item[0])[1]

        return self._aggregate(scores, self._score_aggregation), best_passage

    def rerank(self, batch):
        """
        Re-score the current ranking for each query in the batch.
        """
        top_psg_lines = []

        for qid, qinfo in batch.items():
            query_text = self._query_to_text(qinfo.get("qstring", ""))
            reranked = []

            for _, external_docid in qinfo.get("ranking", []):
                score, best_passage = self._score_document(query_text, external_docid)
                reranked.append((score, external_docid))

                if self._top_psg_path and best_passage:
                    top_psg_lines.append("{}.0 {}".format(external_docid, best_passage))

            reranked.sort(key=lambda item: (-item[0], item[1]))
            qinfo["ranking"] = reranked

        if self._top_psg_path:
            try:
                with open(self._top_psg_path, "w") as handle:
                    for line in top_psg_lines:
                        handle.write("{}\n".format(line))
            except OSError as exc:
                print(
                    "Warning: Cannot write bertrr:topPsgPath {}: {}".format(
                        self._top_psg_path, exc
                    ),
                    flush=True,
                )

        return batch

    def close(self):
        self._runtime = None
