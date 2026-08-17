"""Typed state and tool arguments for the agentic RAG graph."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TypedDict

from pydantic import BaseModel, ConfigDict, Field


Ranking = List[Tuple[float, str]]


class RankingRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ranking_id: str
    source: str
    query: str
    ranking: Ranking


class SearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=500)


class RerankArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ranking_id: str = Field(min_length=1)
    query: str = Field(min_length=1, max_length=500)


class FuseArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ranking_ids: List[str] = Field(min_length=2, max_length=4)


class FinishArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ranking_id: str = Field(min_length=1)


class AgentStateSnapshot(BaseModel):
    """Validate a serializable graph state at node boundaries."""

    model_config = ConfigDict(extra="allow")
    qid: str
    question: str
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    rankings: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    ranking_order: List[str] = Field(default_factory=list)
    pending_tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    trajectory: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    selected_ranking_id: Optional[str] = None
    stop_reason: Optional[str] = None
    answer: str = ""
    final_prompt: List[Dict[str, Any]] = Field(default_factory=list)
    turns: int = 0
    retrieval_calls: int = 0
    rerank_calls: int = 0
    fusion_calls: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    budget_exhaustions: int = 0
    tool_usage: Dict[str, int] = Field(default_factory=dict)
    llm_calls: int = 0
    llm_duration_seconds: float = 0.0
    usage: Dict[str, float] = Field(default_factory=dict)
    response_models: List[str] = Field(default_factory=list)


class AgentState(TypedDict, total=False):
    qid: str
    question: str
    messages: List[Dict[str, Any]]
    rankings: Dict[str, Dict[str, Any]]
    ranking_order: List[str]
    pending_tool_calls: List[Dict[str, Any]]
    trajectory: List[Dict[str, Any]]
    errors: List[str]
    selected_ranking_id: Optional[str]
    stop_reason: Optional[str]
    answer: str
    final_prompt: List[Dict[str, Any]]
    turns: int
    retrieval_calls: int
    rerank_calls: int
    fusion_calls: int
    tool_calls: int
    tool_errors: int
    budget_exhaustions: int
    tool_usage: Dict[str, int]
    llm_calls: int
    llm_duration_seconds: float
    usage: Dict[str, float]
    response_models: List[str]


def validated(state: AgentState) -> AgentState:
    return AgentStateSnapshot.model_validate(state).model_dump()
