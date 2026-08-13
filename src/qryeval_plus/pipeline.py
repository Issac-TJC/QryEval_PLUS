"""Configurable retrieval, reranking, and answer-generation pipeline."""

from __future__ import annotations

from typing import Any, Dict

from qryeval_plus.config import ConfigError, load_config, ordered_tasks, validate_config
from qryeval_plus.core import Util
from qryeval_plus.core.Timer import Timer


def run_pipeline(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a loaded configuration and return the resulting query batch."""
    errors = validate_config(parameters, check_assets=True)
    if errors:
        raise ConfigError("; ".join(errors))

    from qryeval_plus.core.Idx import Idx
    from qryeval_plus.io.Output import Output
    from qryeval_plus.rag.Agent import Agent
    from qryeval_plus.rerank.Reranker import Reranker
    from qryeval_plus.retrieval.Ranker import Ranker
    from qryeval_plus.rewrite.Rewriter import Rewriter

    timer = Timer()
    timer.start()
    opened = False
    try:
        opened = bool(Idx.open(parameters["indexPath"]))
        if not opened:
            raise RuntimeError(f"Unable to open index: {parameters['indexPath']}")

        queries = Util.read_queries(parameters["queryFilePath"])
        if queries is None:
            raise RuntimeError(f"Unable to read queries: {parameters['queryFilePath']}")
        batch = {qid: {"qstring": qstring} for qid, qstring in queries.items()}

        factories = {
            "agent": Agent,
            "output": Output,
            "ranker": Ranker,
            "reranker": Reranker,
            "rewriter": Rewriter,
        }
        for task_name, role, task_parameters in ordered_tasks(parameters):
            print(f"\n-- {task_name}: {task_parameters['type']} --\n")
            batch = factories[role](task_parameters).execute(batch)
        return batch
    finally:
        if Idx.indexReader is not None:
            Idx.close()
        timer.stop()
        print("Time:  " + str(timer))


def run_config(path: str) -> Dict[str, Any]:
    return run_pipeline(load_config(path))
