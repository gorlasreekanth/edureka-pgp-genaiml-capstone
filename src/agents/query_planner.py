from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Protocol

from src.llm.ollama import ChatMessage


_LOG = logging.getLogger(__name__)


SUPPORTED_INTENTS = ("factual", "summarize", "compare", "list")
DEFAULT_INTENT = "factual"
MIN_PLAN_TOP_K = 3
MAX_PLAN_TOP_K = 8


@dataclass(frozen=True)
class QueryPlan:
    """A retrieval plan produced for one user question.

    ``question`` is the cleaned wording the user actually typed and is what the
    answer agent shows the LLM. ``search_query`` is the wording used against
    the vector store; it can differ when the planner LLM rewrites the question
    to improve retrieval. ``intent`` is a coarse label used only for logging
    and adaptive ``top_k`` selection.
    """

    question: str
    top_k: int
    intent: str = DEFAULT_INTENT
    search_query: str = ""

    def resolved_search_query(self) -> str:
        return self.search_query.strip() or self.question


class ChatClient(Protocol):
    @property
    def is_configured(self) -> bool: ...

    @property
    def configuration_issue(self) -> str | None: ...

    def chat(self, messages: list[ChatMessage]) -> str: ...


_PLANNER_SYSTEM_PROMPT = (
    "You help a document Q&A system plan retrieval for a single user question. "
    "Reply with ONE JSON object and nothing else (no prose, no code fences). "
    "Fields:\n"
    '  "intent": one of "factual", "summarize", "compare", "list"\n'
    '  "search_query": a short retrieval query (3-15 words). You may add likely synonyms '
    "that enterprise documents use (e.g. risk, blocker, issue). Do not invent facts.\n"
    '  "top_k": integer between 3 and 8. Use 3-4 for narrow factual or list questions, '
    "5-8 for summarize or compare questions."
)


class QueryPlannerAgent:
    """Plans the retrieval step for a user question.

    Without an LLM client, the agent is deterministic: it cleans whitespace and
    uses the configured default ``top_k``. With an LLM client, it asks the
    model to classify intent, rewrite the question for vector search, and
    choose a ``top_k`` that suits the question shape. All LLM failures fall
    back to the deterministic path so the demo never breaks on a planner glitch.
    """

    def __init__(self, llm_client: ChatClient | None = None) -> None:
        self.llm_client = llm_client

    def plan(self, question: str, default_top_k: int) -> QueryPlan:
        normalized_question = re.sub(r"\s+", " ", question).strip()
        if not normalized_question:
            raise ValueError("Enter a question before asking the documents.")
        if default_top_k <= 0:
            raise ValueError("default_top_k must be greater than zero.")

        if self.llm_client and self.llm_client.is_configured:
            llm_plan = self._plan_with_llm(normalized_question, default_top_k)
            if llm_plan is not None:
                return llm_plan

        return QueryPlan(
            question=normalized_question,
            top_k=default_top_k,
            intent=DEFAULT_INTENT,
            search_query=normalized_question,
        )

    def _plan_with_llm(self, question: str, default_top_k: int) -> QueryPlan | None:
        try:
            raw = self.llm_client.chat(
                [
                    ChatMessage(role="system", content=_PLANNER_SYSTEM_PROMPT),
                    ChatMessage(role="user", content=f"Question: {question}"),
                ]
            )
        except Exception as exc:  # noqa: BLE001 - planner must never break the pipeline
            _LOG.warning("Planner LLM call failed, falling back to deterministic plan: %s", exc)
            return None

        payload = _extract_json_object(raw)
        if not payload:
            _LOG.info("Planner LLM did not return JSON; falling back. Response: %r", raw[:200])
            return None

        intent = payload.get("intent", DEFAULT_INTENT)
        if not isinstance(intent, str) or intent.lower() not in SUPPORTED_INTENTS:
            intent = DEFAULT_INTENT
        else:
            intent = intent.lower()

        search_query = payload.get("search_query", question)
        if not isinstance(search_query, str) or not search_query.strip():
            search_query = question
        else:
            search_query = re.sub(r"\s+", " ", search_query).strip()

        top_k_raw = payload.get("top_k", default_top_k)
        top_k = _coerce_top_k(top_k_raw, default_top_k)

        return QueryPlan(
            question=question,
            top_k=top_k,
            intent=intent,
            search_query=search_query,
        )


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_object(raw: str) -> dict | None:
    if not raw:
        return None
    candidates: list[str] = []
    stripped = raw.strip()
    if stripped.startswith("{"):
        candidates.append(stripped)
    match = _JSON_OBJECT_RE.search(stripped)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None


def _coerce_top_k(value: object, default_top_k: int) -> int:
    try:
        as_int = int(value)
    except (TypeError, ValueError):
        return _clamp_top_k(default_top_k)
    return _clamp_top_k(as_int)


def _clamp_top_k(value: int) -> int:
    if value < MIN_PLAN_TOP_K:
        return MIN_PLAN_TOP_K
    if value > MAX_PLAN_TOP_K:
        return MAX_PLAN_TOP_K
    return value
