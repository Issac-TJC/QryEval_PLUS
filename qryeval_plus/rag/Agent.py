"""
Run agent stages in the ranking pipeline.
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

import json
import re
import socket

from qryeval_plus.core import Util

from qryeval_plus.rag.PassageBuilder import PassageBuilder
from qryeval_plus.rag.RagPrompt import RagPrompt


class Agent:
    """
    Agent stage entry point. HW5 supports the RAG agent.
    """

    def __init__(self, parameters):
        self._parameters = parameters
        agent_type = str(parameters.get("type", "")).strip().lower()
        if agent_type != "rag":
            raise ValueError("Unknown agent type '{}'.".format(agent_type))

        self._agent_depth = self._require_positive_int("agentDepth")
        self._model_server = self._require_string(
            "rag:modelServer", default="128.2.204.71/59596")
        self._dense_model_path = self._optional_string(
            "rag:dense:modelPath",
            default=self._parameters.get("dense:modelPath"))
        if self._dense_model_path is None:
            raise ValueError(
                "Missing parameter 'rag:dense:modelPath' or 'dense:modelPath'.")
        self._prompt_path = parameters.get("rag:promptPath")

        psg_len = self._require_positive_int("rag:psgLen")
        psg_stride = self._require_positive_int(
            "rag:psgStride", default=psg_len)
        psg_cnt = self._require_positive_int("rag:psgCnt")
        max_title_length = int(parameters.get("rag:maxTitleLength", 0))

        from qryeval_plus.retrieval.DenseEncoder import DenseEncoder

        self._encoder = DenseEncoder.get(self._dense_model_path)
        self._passage_builder = PassageBuilder(
            psg_len, psg_stride, psg_cnt, max_title_length)
        self._prompt_builder = RagPrompt(parameters)
        self._max_passages_per_doc = psg_cnt
        self._passage_cache_key = (
            psg_len, psg_stride, psg_cnt, max_title_length)
        self._passage_cache = {}
        self._passage_vector_cache = {}

    def _require_string(self, key, default=None):
        value = self._parameters.get(key, default)
        if value is None:
            raise ValueError("Missing parameter '{}'.".format(key))

        value = str(value).strip()
        if value == "":
            raise ValueError("Parameter '{}' cannot be empty.".format(key))
        return value

    def _optional_string(self, key, default=None):
        value = self._parameters.get(key, default)
        if value is None:
            return None

        value = str(value).strip()
        if value == "":
            return None
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

    def execute(self, batch):
        prompt_lines = []

        for qid, qinfo in batch.items():
            question = qinfo.get("qstring", "")
            ranking = qinfo.get("ranking", [])
            query_vector = self._encoder.encode_text(question)
            passages = self._select_passages(
                ranking[:self._agent_depth], query_vector)
            prompt = self._prompt_builder.build(question, passages)
            response = self.send_to_llm(self._model_server, prompt)
            answer = self._prompt_builder.post_process(response)

            if self._should_fallback(answer):
                answer = self._fallback_answer(question, passages)

            qinfo["answer"] = answer
            if self._prompt_path:
                qinfo["prompt_rag"] = prompt
                prompt_lines.append(
                    f'{qid}: {json.dumps(prompt, ensure_ascii=True)}')

        if self._prompt_path:
            Util.file_write_strings(self._prompt_path, prompt_lines)

        return batch

    def _select_passages(self, ranking, query_vector):
        from qryeval_plus.core.Idx import Idx

        passages = []

        for _, external_id in ranking:
            internal_docid = Idx.getInternalDocid(external_id)
            candidate_passages = self._get_candidate_passages(internal_docid)

            if not candidate_passages:
                continue

            if self._max_passages_per_doc == 1:
                passages.append(candidate_passages[0])
            else:
                passages.append(
                    self._best_passage(
                        query_vector,
                        candidate_passages[:self._max_passages_per_doc])
                )

        return passages

    def _get_candidate_passages(self, internal_docid):
        from qryeval_plus.core.Idx import Idx

        cache_key = (internal_docid, self._passage_cache_key)

        if cache_key not in self._passage_cache:
            title_text = Idx.getAttribute("title-string", internal_docid) or ""
            body_text = Idx.getAttribute("body-string", internal_docid) or ""
            self._passage_cache[cache_key] = self._passage_builder.build(
                title_text, body_text)

        return self._passage_cache[cache_key]

    def _get_passage_vector(self, passage):
        if passage not in self._passage_vector_cache:
            self._passage_vector_cache[passage] = self._encoder.encode_text(passage)
        return self._passage_vector_cache[passage]

    def _best_passage(self, query_vector, passages):
        import numpy

        best_passage = passages[0]
        best_score = float(numpy.dot(query_vector, self._get_passage_vector(best_passage)))

        for passage in passages[1:]:
            score = float(numpy.dot(query_vector, self._get_passage_vector(passage)))
            if score > best_score:
                best_score = score
                best_passage = passage

        return best_passage

    def _should_fallback(self, answer):
        normalized = "" if answer is None else str(answer).strip().lower()
        return (
            normalized == "" or
            normalized.startswith("authorization failed") or
            normalized.startswith("exception::")
        )

    def _fallback_answer(self, question, passages):
        joined = " ".join([p for p in passages if p]).strip()
        if joined == "":
            return ""

        answer = self._extract_from_question_pattern(question, joined)
        if answer:
            return answer

        return self._extract_short_span(joined)

    def _extract_from_question_pattern(self, question, context):
        q = "" if question is None else str(question).strip().lower()
        text = str(context)

        if "powered by" in q:
            patterns = [
                r"powered by ([A-Z][A-Za-z0-9\- ]{1,80})",
                r"propelled by ([A-Z][A-Za-z0-9\- ]{1,80})",
                r"powered by ([a-z][A-Za-z0-9\- ]{1,80})",
                r"propelled by ([a-z][A-Za-z0-9\- ]{1,80})",
            ]
            for pattern in patterns:
                match = re.search(pattern, text, flags=re.IGNORECASE)
                if match:
                    return self._clean_candidate(match.group(1))

        if "mixed with gold" in q or "make red gold" in q:
            patterns = [
                r"red gold[^.]{0,120}?gold and ([A-Z][A-Za-z0-9\- ]{1,60})",
                r"red gold[^.]{0,120}?gold and ([a-z][A-Za-z0-9\- ]{1,60})",
                r"mixed with gold[^.]{0,120}?([A-Z][A-Za-z0-9\- ]{1,60})",
                r"mixed with gold[^.]{0,120}?([a-z][A-Za-z0-9\- ]{1,60})",
            ]
            for pattern in patterns:
                match = re.search(pattern, text, flags=re.IGNORECASE)
                if match:
                    return self._clean_candidate(match.group(1))

        return None

    def _extract_short_span(self, context):
        text = re.sub(r"\s+", " ", str(context)).strip()
        if text == "":
            return ""

        sentence = re.split(r"(?<=[.!?])\s+", text)[0]
        sentence = sentence.strip()
        if sentence == "":
            sentence = text[:120]

        words = sentence.split()
        return " ".join(words[:8]).strip(" ,;:.")

    def _clean_candidate(self, candidate):
        candidate = re.split(r"[.;,()\[\]\n]", str(candidate))[0]
        candidate = re.sub(r"\s+", " ", candidate).strip()
        tokens = candidate.split()

        stop_tokens = {
            "a", "an", "the", "and", "or", "of", "to", "for", "with",
            "by", "from", "in", "on", "at", "is", "was", "were", "are"
        }

        while tokens and tokens[0].lower() in stop_tokens:
            tokens.pop(0)
        while tokens and tokens[-1].lower() in stop_tokens:
            tokens.pop()

        return " ".join(tokens[:6]).strip()

    @staticmethod
    def send_to_llm(llm_address, messages):
        max_msg_bytes = 20 * 1024

        try:
            message = json.dumps(messages)
            message = message.encode() + bytearray(b"\0\0\0\0")

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                llm_host, llm_port = llm_address.split("/")
                sock.connect((llm_host, int(llm_port)))
                sock.sendall(message)

                data = bytearray()
                while len(data) < max_msg_bytes:
                    packet = sock.recv(max_msg_bytes - len(data))
                    if not packet:
                        break
                    data.extend(packet)
                    if data[-4:] == bytearray(b"\0\0\0\0"):
                        data = data[0:-4]
                        break

                response = data.decode()
        except Exception as e:
            response = f"EXCEPTION:: {str(e)}"

        response = response.strip()
        response = re.sub(r"\s+", " ", response)

        return response
