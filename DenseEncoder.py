"""
Shared dense encoding utilities for HW5 dense retrieval and RAG.
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

import numpy


class DenseEncoder:
    """
    Cache dense tokenizer/model instances by model path so multiple
    components can share them in one process.
    """

    _instances = {}
    _model_max_sequence_length = 512

    @classmethod
    def get(cls, model_path):
        model_path = str(model_path).strip()
        if model_path == "":
            raise ValueError("Dense model path cannot be empty.")

        if model_path not in cls._instances:
            cls._instances[model_path] = cls(model_path)
        return cls._instances[model_path]

    def __init__(self, model_path):
        self._model_path = model_path

        try:
            import torch
            from transformers import AutoModel
            from transformers import AutoTokenizer
        except Exception as exc:
            raise ImportError(
                "dense retrieval requires torch and transformers to be installed."
            ) from exc

        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_path)
            self._model = AutoModel.from_pretrained(model_path)
        except Exception as exc:
            raise RuntimeError(
                "Cannot load dense encoder model from '{}'.".format(model_path)
            ) from exc

        self._model.to(self._device)
        self._model.eval()

    def _tokenize_string(self, text):
        return self._tokenizer.encode_plus(
            "" if text is None else str(text),
            max_length=self._model_max_sequence_length,
            truncation=True,
            return_tensors="pt")

    def encode_text(self, text):
        encoded = self._tokenize_string(text)
        encoded = {key: value.to(self._device) for key, value in encoded.items()}

        with self._torch.no_grad():
            outputs = self._model(**encoded)
            rep = outputs.last_hidden_state[:, 0]
            rep = rep.squeeze()
            rep = rep.detach().cpu().numpy().astype("float32")

        return rep

    def encode_texts(self, texts):
        return [self.encode_text(text) for text in texts]

    def dot_score(self, text_a, text_b):
        vec_a = self.encode_text(text_a)
        vec_b = self.encode_text(text_b)
        return float(numpy.dot(vec_a, vec_b))

    def search_faiss(self, query_text, faiss_index, topk):
        query_vector = self.encode_text(query_text)
        query_vector = numpy.array([query_vector], dtype="float32")
        scores, docids = faiss_index.search(query_vector, int(topk))
        return scores[0], docids[0]
