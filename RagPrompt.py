"""
Prompt construction and response post-processing for RAG.
"""

# Copyright (c) 2026, Carnegie Mellon University.  All Rights Reserved.

import os
import re


class RagPrompt:
    """
    Build prompts for the LLM server.
    """

    def __init__(self, parameters):
        self._parameters = parameters
        self._email = self._first_nonempty([
            parameters.get("rag:email"),
            parameters.get("email"),
            os.environ.get("RAG_EMAIL"),
            os.environ.get("AUTH_EMAIL"),
            os.environ.get("ANDREW_EMAIL"),
            "junchent@andrew.cmu.edu",
        ])
        self._code = self._first_nonempty([
            parameters.get("rag:code"),
            parameters.get("code"),
            os.environ.get("RAG_CODE"),
            os.environ.get("AUTH_CODE"),
            os.environ.get("ACCESS_CODE"),
            "pqI4",
        ])
        self._prompt_id = int(parameters.get("rag:prompt", 1))

    def _first_nonempty(self, values):
        for value in values:
            if value is None:
                continue
            value = str(value).strip()
            if value != "":
                return value
        return None

    def build(self, question, passages):
        context = " ".join(
            [str(passage).strip() for passage in passages if str(passage).strip()]
        ).strip()
        system_content, user_content = self._build_prompt_content(question, context)

        return [
            {
                "role": "authorize",
                "email": self._email if self._email is not None else "",
                "code": self._code if self._code is not None else "",
            },
            {
                "role": "system",
                "content": system_content,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

    def _build_prompt_content(self, question, context):
        question = "" if question is None else str(question).strip()
        context = "" if context is None else str(context).strip()

        prompts = {
            1: (
                "You are a helpful chatbot. Use the passage below to provide "
                "a short answer to the specified question. Don't explain your "
                "reasoning. Just generate a short answer.",
                (
                    f"Question: {question}\n"
                    f"Context:{context}\n"
                    f"Question: {question}\n"
                    "Answer: "
                ),
            ),
            2: (
                "Answer the question using only the provided context. "
                "Return a very short answer of at most five words. "
                "Do not add any explanation.",
                (
                    f"Question: {question}\n"
                    f"Context: {context}\n"
                    "Return only the answer phrase.\n"
                    "Answer: "
                ),
            ),
            3: (
                "Answer only from the provided context. "
                "If the context is insufficient, answer Not enough information. "
                "Keep the answer brief and do not speculate.",
                (
                    f"Question: {question}\n"
                    f"Context: {context}\n"
                    "Give a short answer grounded in the context.\n"
                    "Answer: "
                ),
            ),
            4: (
                "You are extracting a named answer span from the context. "
                "Respond with only the best entity, title, date, place, or number "
                "that answers the question. Do not write a sentence.",
                (
                    f"Question: {question}\n"
                    f"Context: {context}\n"
                    "Extract the shortest correct answer span.\n"
                    "Answer: "
                ),
            ),
            5: (
                "Use only the context to answer the question. "
                "Prefer canonical names over descriptions, and avoid lists unless "
                "the answer truly requires multiple items. Do not explain.",
                (
                    f"Question: {question}\n"
                    f"Context: {context}\n"
                    "Return a clean canonical answer with no extra words.\n"
                    "Answer: "
                ),
            ),
            6: (
                "Answer directly from the provided context. "
                "Give the answer first, in the shortest natural form possible. "
                "Avoid hedging, preambles, and explanations.",
                (
                    f"Question: {question}\n"
                    f"Context: {context}\n"
                    "Write only the direct answer.\n"
                    "Answer: "
                ),
            ),
        }

        if self._prompt_id not in prompts:
            raise ValueError(
                "Unsupported rag:prompt value '{}'. Expected 1-6.".format(
                    self._prompt_id)
            )

        return prompts[self._prompt_id]

    def post_process(self, response):
        response = "" if response is None else str(response)
        response = response.strip()
        response = re.sub(r"\s+", " ", response)
        response = re.sub(r"^(answer|short answer)\s*:\s*", "", response, flags=re.I)
        response = re.sub(r"\s*\([^)]*insufficient[^)]*\)\s*$", "", response, flags=re.I)
        response = re.sub(r"^(the answer is|it is)\s+", "", response, flags=re.I)
        response = response.strip(" \t\r\n\"'")
        return response
