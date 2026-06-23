"""Input validation for uploaded files and user questions.

These checks run before any document parsing, embedding, or LLM call so that
malformed, oversized, or suspicious input is rejected with a clear message
instead of failing deeper in the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.ingestion.loaders import SUPPORTED_EXTENSIONS


DEFAULT_MAX_FILE_SIZE_MB = 10
DEFAULT_MIN_QUESTION_CHARS = 3
DEFAULT_MAX_QUESTION_CHARS = 500


PROMPT_INJECTION_PATTERNS: tuple[str, ...] = (
    "ignore previous instructions",
    "ignore all previous",
    "disregard your instructions",
    "disregard the above",
    "forget everything above",
    "you are now",
    "act as the system",
    "reveal the system prompt",
    "print the system prompt",
)


class InputValidationError(ValueError):
    """Raised when a user-provided file or question fails validation."""


@dataclass(frozen=True)
class ValidationLimits:
    max_file_size_mb: int = DEFAULT_MAX_FILE_SIZE_MB
    min_question_chars: int = DEFAULT_MIN_QUESTION_CHARS
    max_question_chars: int = DEFAULT_MAX_QUESTION_CHARS

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


def validate_question(question: str, limits: ValidationLimits | None = None) -> str:
    """Return a cleaned question or raise ``InputValidationError``."""
    limits = limits or ValidationLimits()

    if question is None:
        raise InputValidationError("Enter a question before asking the documents.")

    cleaned = " ".join(question.split())
    if not cleaned:
        raise InputValidationError("Enter a question before asking the documents.")

    if len(cleaned) < limits.min_question_chars:
        raise InputValidationError(
            f"Question is too short. Use at least {limits.min_question_chars} characters."
        )

    if len(cleaned) > limits.max_question_chars:
        raise InputValidationError(
            f"Question is too long. Keep it under {limits.max_question_chars} characters."
        )

    lowered = cleaned.lower()
    for pattern in PROMPT_INJECTION_PATTERNS:
        if pattern in lowered:
            raise InputValidationError(
                "That question looks like a prompt-injection attempt. Rephrase and try again."
            )

    return cleaned


def validate_uploaded_file(
    file_name: str,
    content: bytes,
    limits: ValidationLimits | None = None,
) -> None:
    """Raise ``InputValidationError`` if the upload should not be processed."""
    limits = limits or ValidationLimits()

    if not file_name or not file_name.strip():
        raise InputValidationError("Uploaded file is missing a name.")

    extension = Path(file_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise InputValidationError(
            f"{file_name} is not a supported file type. Use one of: {supported}."
        )

    if content is None or len(content) == 0:
        raise InputValidationError(f"{file_name} is empty.")

    if len(content) > limits.max_file_size_bytes:
        raise InputValidationError(
            f"{file_name} is larger than the {limits.max_file_size_mb} MB upload limit."
        )
