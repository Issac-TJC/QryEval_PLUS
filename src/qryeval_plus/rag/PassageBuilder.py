"""
Build passages from title and body strings.
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.


class PassageBuilder:
    """
    Shared passage construction logic for RAG and rerankers.
    """

    def __init__(self, psg_len, psg_stride, psg_cnt, max_title_length=0):
        self._psg_len = int(psg_len)
        self._psg_stride = int(psg_stride)
        self._psg_cnt = int(psg_cnt)
        self._max_title_length = int(max_title_length)

    def _truncate_tokens(self, text, max_tokens=None):
        if text is None:
            return []

        tokens = str(text).split()
        if max_tokens is not None:
            tokens = tokens[:max_tokens]
        return tokens

    def build(self, title_text, body_text):
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
