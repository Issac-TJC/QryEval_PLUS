"""Retrieval tools exposed to the LangGraph planner."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import numpy

from qryeval_plus.agentic.state import (
    FinishArgs,
    FuseArgs,
    RerankArgs,
    SearchArgs,
)
from qryeval_plus.core import Util
from qryeval_plus.rag.PassageBuilder import PassageBuilder


TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "search_bm25": {
        "description": "Search the collection with BM25. Rewrite the query when evidence is weak.",
        "parameters": SearchArgs.model_json_schema(),
    },
    "search_dense": {
        "description": "Search the collection by dense semantic similarity.",
        "parameters": SearchArgs.model_json_schema(),
    },
    "rerank": {
        "description": "Rerank an existing result list with MiniLM-L6 for the supplied query.",
        "parameters": RerankArgs.model_json_schema(),
    },
    "fuse_rankings": {
        "description": "Fuse two to four existing result lists with reciprocal-rank fusion.",
        "parameters": FuseArgs.model_json_schema(),
    },
    "finish_research": {
        "description": "Stop retrieval and answer from the top five passages of this ranking.",
        "parameters": FinishArgs.model_json_schema(),
    },
}


def tool_definitions(allowed_tools: Iterable[str]) -> List[Dict[str, Any]]:
    definitions = []
    for name in allowed_tools:
        schema = TOOL_SCHEMAS[name]
        definitions.append({
            "type": "function",
            "function": {
                "name": name,
                "description": schema["description"],
                "parameters": schema["parameters"],
            },
        })
    return definitions


class AgentToolbox:
    """Execute tools while keeping heavyweight rankers cached per experiment."""

    def __init__(self, parameters: Dict[str, Any]):
        self.parameters = parameters
        self.depth = int(parameters.get("agent:retrievalDepth", 1000))
        self.context_docs = int(
            parameters.get("agent:maxContextDocs", parameters.get("agentDepth", 5))
        )
        self._baseline_path = parameters.get("agent:bm25InRankPath")
        self._baseline = None
        self._bm25 = None
        self._dense = None
        self._reranker = None
        self._encoder = None
        self._passage_builder = PassageBuilder(
            int(parameters.get("rag:psgLen", 150)),
            int(parameters.get("rag:psgStride", 140)),
            int(parameters.get("rag:psgCnt", 6)),
            int(parameters.get("rag:maxTitleLength", 15)),
        )
        self._passage_cache: Dict[int, List[str]] = {}
        self._vector_cache: Dict[str, Any] = {}

    def execute(
        self,
        *,
        name: str,
        arguments: Dict[str, Any],
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        if name == "search_bm25":
            args = SearchArgs.model_validate(arguments)
            ranking = self._search_bm25(state["qid"], state["question"], args.query)
            return self._store_ranking(state, "bm25", args.query, ranking)
        if name == "search_dense":
            args = SearchArgs.model_validate(arguments)
            ranking = self._search_dense(state["qid"], args.query)
            return self._store_ranking(state, "dense", args.query, ranking)
        if name == "rerank":
            args = RerankArgs.model_validate(arguments)
            record = self._record(state, args.ranking_id)
            ranking = self._rerank_results(state["qid"], args.query, record["ranking"])
            return self._store_ranking(state, "rerank", args.query, ranking)
        if name == "fuse_rankings":
            args = FuseArgs.model_validate(arguments)
            rankings = [self._record(state, item)["ranking"] for item in args.ranking_ids]
            ranking = self._rrf(rankings)
            return self._store_ranking(state, "rrf", state["question"], ranking)
        if name == "finish_research":
            args = FinishArgs.model_validate(arguments)
            self._record(state, args.ranking_id)
            return {"ranking_id": args.ranking_id, "finished": True}
        raise ValueError("Unknown tool: {}".format(name))

    def context_for(self, state: Dict[str, Any], ranking_id: str) -> Tuple[List[str], Any]:
        record = self._record(state, ranking_id)
        passages = self._passages(record["query"], record["ranking"])
        return passages, self._format_evidence(record, passages)

    def _search_bm25(self, qid: str, original: str, query: str):
        if self._baseline_path and _same_query(original, query):
            if self._baseline is None:
                self._baseline = Util.read_rankings(self._baseline_path)
            if qid in self._baseline:
                return list(self._baseline[qid])[:self.depth]

        if self._bm25 is None:
            from qryeval_plus.retrieval.Ranker import Ranker
            self._bm25 = Ranker({
                "type": "BM25",
                "outputLength": self.depth,
                "BM25:k_1": 1.2,
                "BM25:b": 0.75,
            })
        batch = {qid: {"qstring": query}}
        return self._bm25.execute(batch)[qid]["ranking"]

    def _search_dense(self, qid: str, query: str):
        if self._dense is None:
            from qryeval_plus.retrieval.Ranker import Ranker
            self._dense = Ranker({
                "type": "dense",
                "outputLength": self.depth,
                "dense:indexPath": self.parameters["dense:indexPath"],
                "dense:modelPath": self.parameters["dense:modelPath"],
            })
        batch = {qid: {"qstring": query}}
        return self._dense.execute(batch)[qid]["ranking"]

    def _rerank_results(self, qid: str, query: str, ranking):
        if self._reranker is None:
            from qryeval_plus.rerank.Reranker import Reranker
            self._reranker = Reranker({
                "type": "bertrr",
                "rerankDepth": 100,
                "bertrr:modelPath": self.parameters["bertrr:modelPath"],
                "bertrr:psgLen": 150,
                "bertrr:psgStride": 140,
                "bertrr:psgCnt": int(self.parameters.get("rag:psgCnt", 6)),
                "bertrr:scoreAggregation": "maxp",
                "bertrr:maxTitleLength": 15,
            })
        batch = {qid: {"qstring": query, "ranking": list(ranking)}}
        return self._reranker.execute(batch)[qid]["ranking"]

    def _store_ranking(self, state, source, query, ranking):
        ranking_id = "r{}".format(len(state["ranking_order"]) + 1)
        state["rankings"][ranking_id] = {
            "ranking_id": ranking_id,
            "source": source,
            "query": query,
            "ranking": [(float(score), str(docid)) for score, docid in ranking],
        }
        state["ranking_order"].append(ranking_id)
        passages = self._passages(query, ranking)
        return {
            "ranking_id": ranking_id,
            "source": source,
            "result_count": len(ranking),
            "evidence": self._format_evidence(state["rankings"][ranking_id], passages),
        }

    def _record(self, state, ranking_id):
        try:
            return state["rankings"][ranking_id]
        except KeyError as exc:
            raise ValueError("Unknown ranking_id: {}".format(ranking_id)) from exc

    def _passages(self, query: str, ranking):
        from qryeval_plus.core.Idx import Idx
        from qryeval_plus.retrieval.DenseEncoder import DenseEncoder

        if self._encoder is None:
            model_path = self.parameters.get(
                "rag:dense:modelPath", self.parameters.get("dense:modelPath")
            )
            self._encoder = DenseEncoder.get(model_path)
        query_vector = self._encoder.encode_text(query)
        result = []
        for _, external_id in list(ranking)[:self.context_docs]:
            internal_id = Idx.getInternalDocid(external_id)
            if internal_id not in self._passage_cache:
                title = Idx.getAttribute("title-string", internal_id) or ""
                body = Idx.getAttribute("body-string", internal_id) or ""
                self._passage_cache[internal_id] = self._passage_builder.build(title, body)
            candidates = self._passage_cache[internal_id]
            if not candidates:
                result.append("")
                continue
            best = max(
                candidates,
                key=lambda passage: float(numpy.dot(
                    query_vector, self._passage_vector(passage)
                )),
            )
            result.append(best)
        return result

    def _passage_vector(self, passage):
        if passage not in self._vector_cache:
            self._vector_cache[passage] = self._encoder.encode_text(passage)
        return self._vector_cache[passage]

    @staticmethod
    def _format_evidence(record, passages):
        rows = []
        for index, ((score, docid), passage) in enumerate(
            zip(record["ranking"], passages), start=1
        ):
            rows.append({
                "rank": index,
                "doc_id": docid,
                "score": float(score),
                "passage": passage,
            })
        return rows

    def _rrf(self, rankings):
        scores: Dict[str, float] = {}
        for ranking in rankings:
            for rank, (_, docid) in enumerate(ranking, start=1):
                scores[docid] = scores.get(docid, 0.0) + 1.0 / (60.0 + rank)
        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [(score, docid) for docid, score in ordered[:self.depth]]


def _same_query(left: str, right: str) -> bool:
    return " ".join(str(left).lower().split()) == " ".join(str(right).lower().split())
