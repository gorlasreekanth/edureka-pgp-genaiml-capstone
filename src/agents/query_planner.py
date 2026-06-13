from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class QueryPlan:
    question: str
    top_k: int


class QueryPlannerAgent:
    def plan(self, question: str, default_top_k: int) -> QueryPlan:
        normalized_question = re.sub(r"\s+", " ", question).strip()
        if not normalized_question:
            raise ValueError("Enter a question before asking the documents.")
        if default_top_k <= 0:
            raise ValueError("default_top_k must be greater than zero.")
        return QueryPlan(question=normalized_question, top_k=default_top_k)
