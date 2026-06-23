from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from src.agents.query_planner import (
    DEFAULT_INTENT,
    MAX_PLAN_TOP_K,
    MIN_PLAN_TOP_K,
    QueryPlan,
    QueryPlannerAgent,
)
from src.llm.ollama import ChatMessage


@dataclass
class FakeChatClient:
    configured: bool
    response: str = ""
    raises: Exception | None = None
    seen_messages: list[list[ChatMessage]] = field(default_factory=list)

    @property
    def is_configured(self) -> bool:
        return self.configured

    @property
    def configuration_issue(self) -> str | None:
        return None if self.configured else "not configured"

    def chat(self, messages: list[ChatMessage]) -> str:
        self.seen_messages.append(messages)
        if self.raises:
            raise self.raises
        return self.response


def test_plan_normalizes_whitespace_without_llm() -> None:
    plan = QueryPlannerAgent().plan("   What   is the   risk?   ", default_top_k=4)
    assert plan.question == "What is the risk?"
    assert plan.top_k == 4
    assert plan.intent == DEFAULT_INTENT
    assert plan.resolved_search_query() == "What is the risk?"


def test_plan_rejects_empty_question() -> None:
    with pytest.raises(ValueError, match="Enter a question"):
        QueryPlannerAgent().plan("   ", default_top_k=4)


def test_plan_rejects_non_positive_top_k() -> None:
    with pytest.raises(ValueError, match="default_top_k"):
        QueryPlannerAgent().plan("question", default_top_k=0)


def test_plan_uses_deterministic_path_when_llm_not_configured() -> None:
    client = FakeChatClient(configured=False, response='{"intent": "summarize"}')
    plan = QueryPlannerAgent(llm_client=client).plan("Summarize this.", default_top_k=4)
    assert plan.intent == DEFAULT_INTENT
    assert plan.search_query == "Summarize this."
    assert client.seen_messages == []  # LLM was not called


def test_plan_uses_llm_when_configured() -> None:
    client = FakeChatClient(
        configured=True,
        response='{"intent": "summarize", "search_query": "support metrics summary trends", "top_k": 6}',
    )
    plan = QueryPlannerAgent(llm_client=client).plan(
        "Can you summarize the support metrics?", default_top_k=4
    )
    assert plan.intent == "summarize"
    assert plan.search_query == "support metrics summary trends"
    assert plan.top_k == 6
    assert plan.question == "Can you summarize the support metrics?"
    assert client.seen_messages, "LLM should have been called"


def test_plan_clamps_top_k_into_supported_range() -> None:
    client = FakeChatClient(
        configured=True,
        response='{"intent": "factual", "search_query": "risk", "top_k": 99}',
    )
    plan = QueryPlannerAgent(llm_client=client).plan("risk?", default_top_k=4)
    assert plan.top_k == MAX_PLAN_TOP_K

    client.response = '{"intent": "factual", "search_query": "risk", "top_k": 1}'
    plan = QueryPlannerAgent(llm_client=client).plan("risk?", default_top_k=4)
    assert plan.top_k == MIN_PLAN_TOP_K


def test_plan_falls_back_when_llm_returns_garbage() -> None:
    client = FakeChatClient(configured=True, response="I think you should ask differently.")
    plan = QueryPlannerAgent(llm_client=client).plan("What is the risk?", default_top_k=4)
    assert plan.intent == DEFAULT_INTENT
    assert plan.search_query == "What is the risk?"
    assert plan.top_k == 4


def test_plan_falls_back_when_llm_raises() -> None:
    client = FakeChatClient(configured=True, raises=RuntimeError("network blew up"))
    plan = QueryPlannerAgent(llm_client=client).plan("What is the risk?", default_top_k=4)
    assert plan.intent == DEFAULT_INTENT
    assert plan.search_query == "What is the risk?"
    assert plan.top_k == 4


def test_plan_ignores_unknown_intent_and_blank_search_query() -> None:
    client = FakeChatClient(
        configured=True,
        response='{"intent": "philosophical", "search_query": "   ", "top_k": 5}',
    )
    plan = QueryPlannerAgent(llm_client=client).plan("Why?", default_top_k=4)
    assert plan.intent == DEFAULT_INTENT
    assert plan.search_query == "Why?"
    assert plan.top_k == 5


def test_plan_extracts_json_embedded_in_prose() -> None:
    client = FakeChatClient(
        configured=True,
        response='Sure, here you go:\n{"intent": "compare", "search_query": "x vs y", "top_k": 5}\nThanks!',
    )
    plan = QueryPlannerAgent(llm_client=client).plan("Compare x and y", default_top_k=4)
    assert plan.intent == "compare"
    assert plan.search_query == "x vs y"
    assert plan.top_k == 5


def test_resolved_search_query_falls_back_to_question() -> None:
    plan = QueryPlan(question="What is the risk?", top_k=4, search_query="   ")
    assert plan.resolved_search_query() == "What is the risk?"
