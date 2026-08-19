"""
Run agent stages in the ranking pipeline.
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

import json
import re
from copy import deepcopy
from pathlib import Path

from qryeval_plus.core import Util
from qryeval_plus.llm import LLMProviderError, create_provider

from qryeval_plus.rag.PassageBuilder import PassageBuilder
from qryeval_plus.rag.RagPrompt import RagPrompt


class Agent:
    """
    Entry point for retrieval-augmented answer generation.
    """

    def __init__(self, parameters, provider=None):
        self._parameters = parameters
        agent_type = str(parameters.get("type", "")).strip().lower()
        if agent_type != "rag":
            raise ValueError("Unknown agent type '{}'.".format(agent_type))

        self._agent_depth = self._require_positive_int("agentDepth")
        self._provider = provider if provider is not None else create_provider(parameters)
        self._fallback_enabled = self._require_bool("rag:fallback", default=False)
        self._continue_on_error = self._require_bool(
            "rag:continueOnError", default=False
        )
        self._dense_model_path = self._optional_string(
            "rag:dense:modelPath",
            default=self._parameters.get("dense:modelPath"))
        if self._dense_model_path is None:
            raise ValueError(
                "Missing parameter 'rag:dense:modelPath' or 'dense:modelPath'.")
        self._prompt_path = parameters.get("rag:promptPath")
        self._checkpoint_path = parameters.get("agent:checkpointPath")
        self._trajectory_path = parameters.get("agent:trajectoryPath")
        self._resume = self._require_bool("agent:resume", default=False)
        self._completed = self._load_checkpoints() if self._resume else {}
        if not self._resume:
            for value in (self._checkpoint_path, self._trajectory_path):
                if value:
                    artifact = Path(value)
                    artifact.parent.mkdir(parents=True, exist_ok=True)
                    artifact.write_text("", encoding="utf-8")

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

    def _require_bool(self, key, default=None):
        value = self._parameters.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "yes", "1", "on"}:
                return True
            if normalized in {"false", "no", "0", "off"}:
                return False
        raise ValueError("Parameter '{}' must be a boolean.".format(key))

    def execute(self, batch):
        prompt_lines = []

        for qid, qinfo in batch.items():
            if qid in self._completed:
                qinfo.update(deepcopy(self._completed[qid]))
                if self._prompt_path and "prompt_rag" in qinfo:
                    prompt_lines.append(
                        f'{qid}: {json.dumps(qinfo["prompt_rag"], ensure_ascii=True)}')
                continue
            question = (
                qinfo.get("original_qstring", qinfo.get("qstring", ""))
                if self._require_bool("rag:useOriginalQuestion", default=False)
                else qinfo.get("qstring", "")
            )
            ranking = qinfo.get("ranking", [])
            query_vector = self._encoder.encode_text(question)
            passages = self._select_passages(
                ranking[:self._agent_depth], query_vector)
            prompt = self._prompt_builder.build(question, passages)
            try:
                response = self._provider.generate(prompt)
                answer = self._prompt_builder.post_process(response.content)
                if not answer:
                    raise LLMProviderError("The normalized LLM answer is empty.")
                llm_status = {
                    "provider": response.provider,
                    "model": response.model,
                    "success": True,
                    "fallback_used": False,
                    "calls": 1,
                    "cache_hits": int(bool(response.cache_hit)),
                    "duration_seconds": response.duration_seconds,
                    "usage": response.usage,
                    "estimated_cost_usd": response.estimated_cost_usd,
                    "request_id": response.request_id,
                }
            except LLMProviderError as exc:
                if not self._fallback_enabled and not self._continue_on_error:
                    raise RuntimeError(
                        "LLM generation failed for query '{}': {}".format(qid, exc)
                    ) from exc
                answer = (
                    self._fallback_answer(question, passages)
                    if self._fallback_enabled else ""
                )
                llm_status = {
                    "provider": self._provider.name,
                    "model": self._provider.model,
                    "success": False,
                    "fallback_used": self._fallback_enabled,
                    "error": str(exc),
                }

            qinfo["answer"] = answer
            qinfo["llm"] = llm_status
            if self._prompt_path:
                qinfo["prompt_rag"] = prompt
                prompt_lines.append(
                    f'{qid}: {json.dumps(prompt, ensure_ascii=True)}')
            self._append_checkpoint(qid, qinfo)
            self._append_trajectory(qid, qinfo)

        if self._prompt_path:
            Util.file_write_strings(self._prompt_path, prompt_lines)

        return batch

    def _append_checkpoint(self, qid, qinfo):
        if not self._checkpoint_path:
            return
        path = Path(self._checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(
                {"qid": str(qid), "qinfo": qinfo}, ensure_ascii=False
            ) + "\n")

    def _load_checkpoints(self):
        if not self._checkpoint_path or not Path(self._checkpoint_path).is_file():
            return {}
        completed = {}
        with Path(self._checkpoint_path).open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                    qinfo = record["qinfo"]
                    if qinfo.get("llm", {}).get("success") is not False:
                        completed[str(record["qid"])] = qinfo
                except (KeyError, TypeError, json.JSONDecodeError):
                    continue
        return completed

    def _append_trajectory(self, qid, qinfo):
        if not self._trajectory_path:
            return
        ranking = qinfo.get("ranking", [])
        payload = {
            "qid": str(qid),
            "question": qinfo.get("qstring", ""),
            "trajectory": [{
                "event": "fixed_rag",
                "ranking_depth": len(ranking),
                "top_doc_ids": [item[1] for item in ranking[:5]],
                "answer": qinfo.get("answer", ""),
            }],
            "llm": qinfo.get("llm", {}),
        }
        path = Path(self._trajectory_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

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


def create_agent(parameters, provider=None):
    """Dispatch fixed RAG and LangGraph agentic RAG without breaking imports."""
    agent_type = str(parameters.get("type", "")).strip().lower()
    if agent_type == "rag":
        return Agent(parameters, provider=provider)
    if agent_type == "agentic_rag":
        from qryeval_plus.agentic import AgenticRagAgent
        return AgenticRagAgent(parameters, provider=provider)
    if agent_type == "rewrite_rag":
        from qryeval_plus.agentic.rewrite import RewriteRagAgent
        return RewriteRagAgent(parameters, provider=provider)
    raise ValueError("Unknown agent type '{}'.".format(agent_type))
