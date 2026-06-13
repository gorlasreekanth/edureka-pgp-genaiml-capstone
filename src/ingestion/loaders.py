from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError
from pypdf import PdfReader

from src.models import SourceDocument


SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".csv", ".xlsx", ".xls"}


class DocumentLoadError(ValueError):
    """Raised when an uploaded document cannot be parsed into useful text."""


def load_uploaded_file(file_name: str, content: bytes) -> list[SourceDocument]:
    extension = Path(file_name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise DocumentLoadError(f"{file_name} is not supported. Use one of: {supported}.")

    if not content:
        raise DocumentLoadError(f"{file_name} is empty.")

    if extension == ".txt":
        return [_load_txt(file_name, content)]
    if extension == ".pdf":
        return _load_pdf(file_name, content)
    if extension == ".csv":
        return [_load_csv(file_name, content)]
    return _load_excel(file_name, content)


def _load_txt(file_name: str, content: bytes) -> SourceDocument:
    text = _decode_text(file_name, content)
    return SourceDocument(
        text=_require_text(file_name, text),
        metadata={"source_name": file_name, "doc_type": "txt"},
    )


def _load_pdf(file_name: str, content: bytes) -> list[SourceDocument]:
    reader = PdfReader(BytesIO(content))
    documents: list[SourceDocument] = []

    for page_index, page in enumerate(reader.pages, start=1):
        text = _normalize_text(page.extract_text() or "")
        if text.strip():
            documents.append(
                SourceDocument(
                    text=text,
                    metadata={
                        "source_name": file_name,
                        "doc_type": "pdf",
                        "page": page_index,
                    },
                )
            )

    if not documents:
        raise DocumentLoadError(f"{file_name} has no extractable text.")

    return documents


def _load_csv(file_name: str, content: bytes) -> SourceDocument:
    try:
        data_frame = pd.read_csv(BytesIO(content))
    except EmptyDataError as exc:
        raise DocumentLoadError(f"{file_name} has no readable CSV rows.") from exc

    text = _dataframe_to_text(file_name, data_frame)
    return SourceDocument(
        text=text,
        metadata={
            "source_name": file_name,
            "doc_type": "csv",
            "row_range": _row_range(data_frame),
        },
    )


def _load_excel(file_name: str, content: bytes) -> list[SourceDocument]:
    sheets = pd.read_excel(BytesIO(content), sheet_name=None)
    documents: list[SourceDocument] = []

    for sheet_name, data_frame in sheets.items():
        if data_frame.empty:
            continue
        documents.append(
            SourceDocument(
                text=_dataframe_to_text(f"{file_name} / {sheet_name}", data_frame),
                metadata={
                    "source_name": file_name,
                    "doc_type": "excel",
                    "sheet": str(sheet_name),
                    "row_range": _row_range(data_frame),
                },
            )
        )

    if not documents:
        raise DocumentLoadError(f"{file_name} has no readable Excel rows.")

    return documents


def _decode_text(file_name: str, content: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return _normalize_text(content.decode(encoding))
        except UnicodeDecodeError:
            continue
    raise DocumentLoadError(f"{file_name} could not be decoded as text.")


def _dataframe_to_text(label: str, data_frame: pd.DataFrame) -> str:
    if data_frame.empty:
        raise DocumentLoadError(f"{label} has no readable rows.")

    clean_frame = data_frame.fillna("")
    lines = ["Columns: " + ", ".join(str(column) for column in clean_frame.columns)]

    for row_number, (_, row) in enumerate(clean_frame.iterrows(), start=1):
        values = [
            f"{column}: {value}"
            for column, value in row.items()
            if str(value).strip()
        ]
        if values:
            lines.append(f"Row {row_number}: " + "; ".join(values))

    return _require_text(label, "\n".join(lines))


def _row_range(data_frame: pd.DataFrame) -> str:
    if data_frame.empty:
        return "none"
    return f"1-{len(data_frame)}"


def _normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _require_text(file_name: str, text: str) -> str:
    normalized = _normalize_text(text)
    if not normalized:
        raise DocumentLoadError(f"{file_name} has no extractable text.")
    return normalized
