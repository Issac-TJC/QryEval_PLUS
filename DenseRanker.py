"""
Dense first-stage ranker backed by FAISS.
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

import numpy

from Idx import Idx

from DenseEncoder import DenseEncoder


class DenseRanker:
    """
    Produce rankings with a dense encoder and FAISS index.
    """

    def __init__(self, parameters):
        self._parameters = parameters
        self._max_results = int(parameters.get("outputLength", 1000))
        self._index_path = self._require_string("dense:indexPath")
        self._model_path = self._require_string("dense:modelPath")

        try:
            import faiss
        except Exception as exc:
            raise ImportError(
                "dense retrieval requires faiss to be installed."
            ) from exc

        try:
            self._faiss_index = faiss.read_index(self._index_path)
        except Exception as exc:
            raise RuntimeError(
                "Cannot load dense FAISS index from '{}'.".format(self._index_path)
            ) from exc

        self._encoder = DenseEncoder.get(self._model_path)

    def _require_string(self, key):
        value = self._parameters.get(key)
        if value is None:
            raise ValueError("Missing parameter '{}'.".format(key))

        value = str(value).strip()
        if value == "":
            raise ValueError("Parameter '{}' cannot be empty.".format(key))
        return value

    def rerank_batch(self, batch):
        qids = list(batch.keys())
        if not qids:
            return batch

        query_texts = []

        for qid in qids:
            query_text = batch[qid]["qstring"]
            print(f'{qid}: {query_text}', flush=True)
            query_texts.append(query_text)

        query_vectors = self._encoder.encode_texts(query_texts)
        query_matrix = numpy.array(query_vectors, dtype="float32")
        all_scores, all_docids = self._faiss_index.search(
            query_matrix, self._max_results)

        for idx, qid in enumerate(qids):
            ranking = []
            for score, internal_docid in zip(all_scores[idx], all_docids[idx]):
                if int(internal_docid) < 0:
                    continue
                ranking.append(
                    (float(score), Idx.getExternalDocid(int(internal_docid))))

            ranking.sort(key=lambda item: (-item[0], item[1]))
            batch[qid]["ranking"] = ranking

        return batch
