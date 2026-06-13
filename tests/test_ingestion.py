from io import BytesIO

import pandas as pd
import pytest

from src.ingestion.loaders import DocumentLoadError, load_uploaded_file


def test_load_txt_normalizes_text() -> None:
    documents = load_uploaded_file("memo.txt", b"Line one\r\nLine two")

    assert len(documents) == 1
    assert documents[0].text == "Line one\nLine two"
    assert documents[0].metadata == {"source_name": "memo.txt", "doc_type": "txt"}


def test_load_csv_turns_rows_into_readable_text() -> None:
    documents = load_uploaded_file("metrics.csv", b"name,value\nRevenue,120\nCost,80\n")

    assert len(documents) == 1
    assert "Columns: name, value" in documents[0].text
    assert "Row 1: name: Revenue; value: 120" in documents[0].text
    assert documents[0].metadata["row_range"] == "1-2"


def test_load_excel_preserves_sheet_metadata() -> None:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame({"item": ["risk"], "status": ["open"]}).to_excel(
            writer,
            sheet_name="Actions",
            index=False,
        )

    documents = load_uploaded_file("workbook.xlsx", buffer.getvalue())

    assert len(documents) == 1
    assert documents[0].metadata["sheet"] == "Actions"
    assert "Row 1: item: risk; status: open" in documents[0].text


def test_unsupported_file_type_is_rejected() -> None:
    with pytest.raises(DocumentLoadError, match="not supported"):
        load_uploaded_file("image.png", b"fake")
