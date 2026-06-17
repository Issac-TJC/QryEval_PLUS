"""
Rerank documents with a BERT cross-encoder.
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

import re
import sys

from Idx import Idx
from QryParser import QryParser


class RerankWithBERT:
    """
    Re-rank an existing ranking with a HuggingFace sequence classifier.
    """

    _VALID_AGGREGATIONS = {"firstp", "avgp", "maxp"}

    def __init__(self, parameters):
        self._parameters = parameters

        self._model_path = self._require_string("bertrr:modelPath")
        self._psg_len = self._require_positive_int("bertrr:psgLen")
        self._psg_stride = self._require_positive_int(
            "bertrr:psgStride", default=self._psg_len)
        self._psg_cnt = self._optional_positive_int(
            "bertrr:psgCnt", default=sys.maxsize)
        self._max_title_length = self._require_nonnegative_int(
            "bertrr:maxTitleLength", default=0)
        self._max_seq_length = self._require_positive_int(
            "bertrr:maxSeqLength", default=512)
        self._top_psg_path = parameters.get("bertrr:topPsgPath")

        self._score_aggregation = self._require_string(
            "bertrr:scoreAggregation").lower()
        if self._score_aggregation not in self._VALID_AGGREGATIONS:
            raise ValueError(
                "Invalid bertrr:scoreAggregation '{}'. Expected one of {}.".format(
                    self._score_aggregation,
                    sorted(self._VALID_AGGREGATIONS)))

        try:
            import torch
            from transformers import AutoModelForSequenceClassification
            from transformers import AutoTokenizer
        except Exception as exc:
            raise ImportError(
                "bertrr requires torch and transformers to be installed."
            ) from exc

        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self._model_path,
                num_labels=1)
        except Exception as exc:
            raise RuntimeError(
                "Cannot load BERT reranker model from '{}'.".format(
                    self._model_path)
            ) from exc

        self._model.to(self._device)
        self._model.eval()

    def rerank(self, batch):
        """
        Re-score the current ranking for each query in the batch.
        """
        top_psg_lines = []

        for qid, qinfo in batch.items():
            query_text = self._get_query_text(qinfo.get("qstring", ""))
            original_ranking = qinfo.get("ranking", [])
            updated_ranking = []

            for _, external_id in original_ranking:
                new_score = float("-inf")
                best_passage = ""

                try:
                    new_score, best_passage = self._score_document(
                        query_text, external_id)
                except Exception:
                    new_score = float("-inf")
                    best_passage = ""

                updated_ranking.append((float(new_score), external_id))
                if self._top_psg_path and best_passage:
                    top_psg_lines.append(
                        "{}.0 {}".format(external_id, best_passage))

            updated_ranking.sort(key=lambda item: (-item[0], item[1]))
            batch[qid]["ranking"] = updated_ranking

        if self._top_psg_path:
            try:
                with open(self._top_psg_path, "w") as handle:
                    for line in top_psg_lines:
                        handle.write("{}\n".format(line))
            except OSError as exc:
                print(
                    "Warning: Cannot write bertrr:topPsgPath {}: {}".format(
                        self._top_psg_path, exc),
                    flush=True)

        return batch

    def _require_string(self, key):
        value = self._parameters.get(key)
        if value is None:
            raise ValueError("Missing parameter '{}'.".format(key))

        value = str(value).strip()
        if value == "":
            raise ValueError("Parameter '{}' cannot be empty.".format(key))

        return value

    def _require_positive_int(self, key, default=None):
        value = self._parameters.get(key, default)
        if value is None:
            raise ValueError("Missing parameter '{}'.".format(key))

        try:
            value = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Parameter '{}' must be an integer.".format(key)
            ) from exc

        if value <= 0:
            raise ValueError(
                "Parameter '{}' must be greater than 0.".format(key))

        return value

    def _optional_positive_int(self, key, default=None):
        value = self._parameters.get(key, default)
        if value is None:
            return None

        try:
            value = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Parameter '{}' must be an integer.".format(key)
            ) from exc

        if value <= 0:
            raise ValueError(
                "Parameter '{}' must be greater than 0.".format(key))

        return value

    def _require_nonnegative_int(self, key, default=0):
        value = self._parameters.get(key, default)

        try:
            value = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Parameter '{}' must be an integer.".format(key)
            ) from exc

        if value < 0:
            raise ValueError(
                "Parameter '{}' must be greater than or equal to 0.".format(key))

        return value

    def _get_query_text(self, qstring):
        qstring = "" if qstring is None else str(qstring).strip()

        if "#" not in qstring and "(" not in qstring and ")" not in qstring:
            query_text = qstring
        else:
            try:
                query_text = QryParser.bowQuery(qstring)
            except Exception:
                query_text = re.sub(r"#[A-Za-z0-9_/]+", " ", qstring)
                query_text = query_text.replace("(", " ").replace(")", " ")

        query_text = re.sub(
            r"\b([A-Za-z0-9_]+)\.[A-Za-z0-9_]+\b", r"\1", query_text)
        query_text = re.sub(r"\s+", " ", query_text).strip()
        return query_text

    def _get_document_strings(self, internal_docid):
        title_text = ""
        body_text = ""

        if self._max_title_length > 0:
            title_val = Idx.getAttribute("title-string", internal_docid)
            if title_val is not None:
                title_text = str(title_val)

        body_val = Idx.getAttribute("body-string", internal_docid)
        if body_val is not None:
            body_text = str(body_val)

        return title_text, body_text

    def _truncate_tokens(self, text, max_tokens=None):
        if text is None:
            return []

        tokens = str(text).split()
        if max_tokens is not None:
            tokens = tokens[:max_tokens]
        return tokens

    def _build_passages(self, title_text, body_text):
        title_tokens = []
        if self._max_title_length > 0:
            title_tokens = self._truncate_tokens(title_text, self._max_title_length)

        body_tokens = self._truncate_tokens(body_text)
        body_passages = []

        if body_tokens:
            start = 0
            previous_end = -1

            while start < len(body_tokens) and len(body_passages) < self._psg_cnt:
                end = min(start + self._psg_len, len(body_tokens))

                # Skip windows fully covered by the previous passage.
                if end <= previous_end:
                    break

                body_passages.append(body_tokens[start:end])
                previous_end = end
                start += self._psg_stride
        elif title_tokens:
            body_passages = [[]]

        passages = []
        title_prefix = " ".join(title_tokens).strip()

        for body_passage in body_passages[:self._psg_cnt]:
            body_part = " ".join(body_passage).strip()
            if title_prefix and body_part:
                passages.append("{} {}".format(title_prefix, body_part))
            elif title_prefix:
                passages.append(title_prefix)
            elif body_part:
                passages.append(body_part)

        return passages

    def _score_document(self, query_text, external_docid):
        internal_docid = Idx.getInternalDocid(external_docid)
        if internal_docid is None:
            return float("-inf"), ""

        title_text, body_text = self._get_document_strings(internal_docid)
        passages = self._build_passages(title_text, body_text)
        if not passages:
            return float("-inf"), ""

        scored_passages = self._score_passages(query_text, passages)
        if not scored_passages:
            return float("-inf"), ""

        passage_scores = [score for score, _ in scored_passages]

        if self._score_aggregation == "firstp":
            best_passage = passages[0]
        else:
            best_passage = max(scored_passages, key=lambda item: item[0])[1]

        return self._aggregate_scores(passage_scores), best_passage

    def _score_passages(self, query_text, passages):
        if not passages:
            return []

        queries = [query_text] * len(passages)
        encoded = self._tokenizer(
            queries,
            passages,
            padding=True,
            truncation=True,
            max_length=self._max_seq_length,
            return_tensors="pt")

        encoded = {key: value.to(self._device) for key, value in encoded.items()}

        with self._torch.no_grad():
            outputs = self._model(**encoded)

        logits = outputs.logits
        if len(logits.shape) == 1:
            scores = [float(score) for score in logits.detach().cpu().tolist()]
        else:
            scores = [
                float(score)
                for score in logits.squeeze(-1).detach().cpu().tolist()
            ]

        return list(zip(scores, passages))

    def _aggregate_scores(self, passage_scores):
        if not passage_scores:
            raise ValueError("Cannot aggregate an empty list of passage scores.")

        if self._score_aggregation == "firstp":
            return float(passage_scores[0])
        if self._score_aggregation == "avgp":
            return float(sum(passage_scores) / len(passage_scores))
        if self._score_aggregation == "maxp":
            return float(max(passage_scores))

        raise ValueError(
            "Unsupported score aggregation '{}'.".format(
                self._score_aggregation))
