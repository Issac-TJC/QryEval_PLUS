"""
Access and manage a BERT-based reranker.
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.
import os
import re

from qryeval_plus.core.Idx import Idx
from qryeval_plus.query.QryParser import QryParser


class BERT:
    """A thin wrapper around HuggingFace Transformers."""

    _instance_cache = {}
    _torch_configured = False
    _cpu_threads = max(1, min(4, os.cpu_count() or 1))

    def __init__(self, model_path, max_sequence_length):
        try:
            import torch
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise ImportError(
                'BERT reranking requires the Python packages '
                '`transformers` and `torch`.'
            ) from exc

        if not self._torch_configured:
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            torch.set_num_threads(self._cpu_threads)
            BERT._torch_configured = True

        cache_key = (model_path, max_sequence_length)
        cached = self._instance_cache.get(cache_key)
        if cached is None:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=1,
            )
            model.eval()
            cached = (tokenizer, model)
            self._instance_cache[cache_key] = cached

        self._torch = torch
        self._max_sequence_length = max_sequence_length
        self.tokenizer, self.model = cached

    def encode_q_psg(self, q_str, psg_str):
        """Encode a (query, passage) pair for the classifier."""
        return self.tokenizer.encode_plus(
            [q_str, psg_str],
            add_special_tokens=True,
            max_length=self._max_sequence_length,
            truncation='only_second',
            return_tensors='pt',
        )

    def encode_batch_q_psg(self, q_str, psg_strs):
        """Encode a batch of (query, passage) pairs."""
        return self.tokenizer(
            [q_str] * len(psg_strs),
            list(psg_strs),
            add_special_tokens=True,
            max_length=self._max_sequence_length,
            truncation='only_second',
            padding=True,
            return_tensors='pt',
        )

    def score_sequence(self, tensors_dict):
        """Score an encoded (query, passage) pair."""
        with self._torch.no_grad():
            outputs = self.model(**tensors_dict)
            return outputs.logits.data.item()

    def score_batch(self, tensors_dict):
        """Score a batch of encoded (query, passage) pairs."""
        with self._torch.no_grad():
            outputs = self.model(**tensors_dict)
            return outputs.logits.squeeze(-1).detach().cpu().tolist()


class RerankWithBERT:
    """
    Access and manage a BERT-based reranker.
    """

    _operator_regex = re.compile(r'#[A-Za-z0-9_]+')
    _weight_regex = re.compile(r'(?<![A-Za-z])[-+]?\d*\.?\d+(?![A-Za-z])')
    _known_fields = {'body', 'title', 'url', 'inlink', 'keywords'}

    # -------------- Methods (alphabetical) ---------------- #

    def __init__(self, parameters):
        self._model_path = parameters.get('bertrr:modelPath')
        self._bert_max_sequence_length = int(
            parameters.get('bertrr:maxSeqLength', 512)
        )
        self._psg_len = int(parameters.get('bertrr:psgLen', 0))
        self._psg_cnt = int(parameters.get('bertrr:psgCnt', 1))
        self._psg_stride = int(parameters.get('bertrr:psgStride', self._psg_len))
        self._max_title_length = int(parameters.get('bertrr:maxTitleLength', 0))
        self._score_aggregation = str(
            parameters.get('bertrr:scoreAggregation', 'firstp')
        ).lower()
        self._batch_size = int(parameters.get('bertrr:batchSize', 16))
        self._top_psg_path = parameters.get('bertrr:topPsgPath')

        if not self._model_path:
            raise AttributeError('Missing parameter: bertrr:modelPath')
        if self._psg_len <= 0:
            raise AttributeError('bertrr:psgLen must be > 0')
        if self._bert_max_sequence_length <= 0:
            raise AttributeError('bertrr:maxSeqLength must be > 0')
        if self._psg_stride <= 0:
            raise AttributeError('bertrr:psgStride must be > 0')
        if self._psg_cnt <= 0:
            raise AttributeError('bertrr:psgCnt must be > 0')
        if self._max_title_length < 0:
            raise AttributeError('bertrr:maxTitleLength must be >= 0')
        if self._batch_size <= 0:
            raise AttributeError('bertrr:batchSize must be > 0')
        if self._score_aggregation not in {'firstp', 'avgp', 'maxp'}:
            raise AttributeError(
                'bertrr:scoreAggregation must be one of {firstp, avgp, maxp}'
            )

        self._bert = BERT(self._model_path, self._bert_max_sequence_length)

    @staticmethod
    def _aggregate_scores(scores, method):
        """Aggregate passage scores into a document score."""
        if len(scores) == 0:
            return float('-inf')
        if method == 'firstp':
            return scores[0]
        if method == 'avgp':
            return sum(scores) / float(len(scores))
        if method == 'maxp':
            return max(scores)

        raise ValueError(f'Unknown aggregation method: {method}')

    @staticmethod
    def _collapse_whitespace(text):
        """Normalize whitespace for passage strings."""
        return ' '.join(str(text).split())

    @classmethod
    def _strip_field_suffixes(cls, text):
        """Convert tokens like apple.body into apple."""
        stripped_tokens = []

        for token in str(text).split():
            parts = token.rsplit('.', 1)
            if len(parts) == 2 and parts[1].lower() in cls._known_fields:
                token = parts[0]
            stripped_tokens.append(token)

        return ' '.join(stripped_tokens)

    @classmethod
    def _query_to_text(cls, qstring):
        """Convert a query string into text suitable for BERT."""
        query_text = str(qstring).strip()
        was_structured = '#' in query_text

        if was_structured:
            try:
                query_text = QryParser.bowQuery(query_text)
            except Exception:
                query_text = cls._operator_regex.sub(' ', query_text)
                query_text = query_text.replace('(', ' ')
                query_text = query_text.replace(')', ' ')
                if '#wsum' in str(qstring).lower():
                    query_text = cls._weight_regex.sub(' ', query_text)

        query_text = cls._strip_field_suffixes(query_text)
        return cls._collapse_whitespace(query_text)

    @staticmethod
    def _slice_text_tokens(text, max_tokens):
        """Convert text to a token list, optionally truncating it."""
        if text is None:
            return []

        tokens = str(text).split()
        if max_tokens is not None and max_tokens >= 0:
            return tokens[:max_tokens]
        return tokens

    @classmethod
    def _build_body_passages(cls, body_tokens, psg_len, psg_stride, psg_cnt):
        """Create overlapping body passages using length and stride."""
        if len(body_tokens) == 0:
            return []

        passages = []
        start = 0
        while start < len(body_tokens) and len(passages) < psg_cnt:
            end = min(start + psg_len, len(body_tokens))
            passages.append(body_tokens[start:end])
            if end >= len(body_tokens):
                break
            start += psg_stride

        return passages

    @classmethod
    def _combine_title_and_body(cls, title_tokens, body_passages):
        """Attach the title prefix to every body passage."""
        if len(body_passages) == 0:
            if len(title_tokens) == 0:
                return []
            return [cls._collapse_whitespace(' '.join(title_tokens))]

        passages = []
        title_prefix = ' '.join(title_tokens)

        for body_tokens in body_passages:
            body_text = ' '.join(body_tokens)
            if title_prefix and body_text:
                passages.append(f'{title_prefix} {body_text}')
            elif title_prefix:
                passages.append(title_prefix)
            else:
                passages.append(body_text)

        return [cls._collapse_whitespace(p) for p in passages]

    def _build_passages(self, internal_docid):
        """Build the list of passage strings for a document."""
        title_tokens = []
        if self._max_title_length > 0:
            title_tokens = self._slice_text_tokens(
                Idx.getAttribute('title-string', internal_docid),
                self._max_title_length,
            )

        body_tokens = self._slice_text_tokens(
            Idx.getAttribute('body-string', internal_docid),
            None,
        )
        body_passages = self._build_body_passages(
            body_tokens,
            self._psg_len,
            self._psg_stride,
            self._psg_cnt,
        )

        return self._combine_title_and_body(title_tokens, body_passages)

    def _score_document(self, query_text, external_docid):
        """Score a document by encoding and scoring its passages."""
        internal_docid = Idx.getInternalDocid(external_docid)
        if internal_docid is None:
            return float('-inf'), ''

        passages = self._build_passages(internal_docid)
        if len(passages) == 0:
            return float('-inf'), ''

        scored_passages = self._score_passages(query_text, passages)
        passage_scores = [score for score, _ in scored_passages]
        best_passage = passages[0]

        if self._score_aggregation == 'maxp':
            best_passage = max(scored_passages, key=lambda item: item[0])[1]
        elif self._score_aggregation == 'avgp':
            best_passage = max(scored_passages, key=lambda item: item[0])[1]

        return self._aggregate_scores(passage_scores, self._score_aggregation), best_passage

    def _score_passages(self, query_text, passages):
        """Score passages in batches for better throughput."""
        scored_passages = []

        for start in range(0, len(passages), self._batch_size):
            batch_passages = passages[start:start + self._batch_size]
            encoded = self._bert.encode_batch_q_psg(query_text, batch_passages)
            scores = self._bert.score_batch(encoded)
            scored_passages.extend(zip(scores, batch_passages))

        return scored_passages

    def rerank(self, batch):
        """Update the results for a set of queries with BERT scores."""
        top_psg_lines = []

        for qid, query_data in batch.items():
            query_text = self._query_to_text(query_data['qstring'])
            reranked = []

            for _, external_docid in query_data.get('ranking', []):
                score, best_passage = self._score_document(
                    query_text,
                    external_docid,
                )
                reranked.append((score, external_docid))
                if self._top_psg_path and best_passage:
                    top_psg_lines.append(
                        f'{external_docid}.0 {best_passage}'
                    )

            reranked.sort(key=lambda item: (-item[0], item[1]))
            batch[qid]['ranking'] = reranked

        if self._top_psg_path:
            try:
                with open(self._top_psg_path, 'w') as handle:
                    for line in top_psg_lines:
                        handle.write(f'{line}\n')
            except OSError as exc:
                print(
                    'Warning: Cannot write bertrr:topPsgPath '
                    f'{self._top_psg_path}: {exc}',
                    flush=True,
                )

        return batch

    def close(self):
        """Release cached BERT model resources."""
        self._bert = None
