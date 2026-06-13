from __future__ import annotations

from typing import Protocol

from src.agents.query_planner import QueryPlan
from src.models import RetrievedChunk


class SearchableVectorStore(Protocol):
    def query(self, question: str, top_k: int) -> list[RetrievedChunk]:
        ...


class RetrievalAgent:
    def __init__(self, vector_store: SearchableVectorStore) -> None:
        self.vector_store = vector_store

    def retrieve(self, plan: QueryPlan) -> list[RetrievedChunk]:
        return self.vector_store.query(plan.question, plan.top_k)
