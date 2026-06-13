from __future__ import annotations

import streamlit as st

from src.config import AppConfig


def main() -> None:
    config = AppConfig.from_env()

    st.set_page_config(page_title="Enterprise Document Q&A", page_icon="📄")
    st.title("Enterprise Document Q&A")
    st.write(
        "Upload enterprise documents, ask a natural-language question, and get a grounded answer with sources."
    )

    st.info(
        "Implementation is starting with the project foundation. Document ingestion, retrieval, and answer generation are wired in the next batch."
    )

    with st.sidebar:
        st.header("Runtime settings")
        st.caption("Values come from environment variables or `.env`.")
        st.write(f"Embedding model: `{config.embedding_model}`")
        st.write(f"Vector store: `{config.chroma_path}`")
        st.write(f"Ollama model: `{config.ollama_model}`")


if __name__ == "__main__":
    main()
