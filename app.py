from __future__ import annotations

import streamlit as st

from src.config import AppConfig
from src.models import IndexResult, QueryResult, RetrievedChunk
from src.rag import DocumentQAWorkflow


@st.cache_resource(show_spinner=False)
def _get_workflow(config: AppConfig) -> DocumentQAWorkflow:
    return DocumentQAWorkflow.from_config(config)


def main() -> None:
    config = AppConfig.from_env()
    workflow = _get_workflow(config)

    st.set_page_config(page_title="Enterprise Document Q&A")
    st.title("Enterprise Document Q&A")
    st.write(
        "Upload enterprise documents, ask a natural-language question, and get a grounded answer with sources."
    )

    top_k = _render_sidebar(config)
    _render_upload_and_index(workflow)
    _render_question_flow(workflow, top_k)


def _render_sidebar(config: AppConfig) -> int:
    with st.sidebar:
        st.header("Runtime settings")
        st.caption("Values come from environment variables or `.env`.")
        st.write(f"Embedding model: `{config.embedding_model}`")
        st.write(f"Vector store: `{config.chroma_path}`")
        st.write(f"Collection: `{config.chroma_collection}`")
        top_k = st.slider(
            "Top K source chunks",
            min_value=1,
            max_value=10,
            value=config.retrieval_top_k,
            help="How many matching chunks to retrieve for each question.",
        )
        if config.embedding_model == "local-hash":
            st.info("Using fast local embeddings. Switch `EMBEDDING_MODEL` for stronger semantic retrieval.")
        if config.has_configured_llm:
            st.success(f"Ollama model: `{config.ollama_model}`")
        else:
            st.warning("Ollama model is still a placeholder. Retrieval will work, but final answers need `.env`.")
        return top_k


def _render_upload_and_index(workflow: DocumentQAWorkflow) -> None:
    st.subheader("1. Upload and index documents")
    uploaded_files = st.file_uploader(
        "Choose PDF, TXT, CSV, or Excel files",
        type=["pdf", "txt", "csv", "xlsx", "xls"],
        accept_multiple_files=True,
    )

    col_index, col_clear = st.columns([1, 1])
    with col_index:
        index_clicked = st.button(
            "Index documents",
            type="primary",
            disabled=not uploaded_files,
        )
    with col_clear:
        if st.button("Clear indexed state"):
            st.session_state.pop("indexed", None)
            st.session_state.pop("index_result", None)
            workflow.vector_store.clear()
            st.success("Indexed document state was cleared.")

    if index_clicked:
        files = [(file.name, file.getvalue()) for file in uploaded_files]
        with st.status("Preparing documents for search...", expanded=True) as status:
            result = workflow.index_files(
                files,
                reset=True,
                progress_callback=status.write,
            )
            if result.indexed_chunk_count > 0:
                status.update(label="Document index is ready.", state="complete")
            else:
                status.update(label="No document text was indexed.", state="error")
        st.session_state["indexed"] = result.indexed_chunk_count > 0
        st.session_state["index_result"] = result

    result = st.session_state.get("index_result")
    if isinstance(result, IndexResult):
        _render_index_result(result)
    elif not uploaded_files:
        st.info("Upload one or more supported files to build the searchable document index.")


def _render_index_result(result: IndexResult) -> None:
    if result.indexed_chunk_count > 0:
        st.success(
            f"Indexed {result.indexed_chunk_count} chunks from {result.document_count} parsed document sections."
        )
    else:
        st.warning("No document text was indexed yet.")

    if result.errors:
        with st.expander("Files that need attention", expanded=True):
            for error in result.errors:
                st.write(f"- {error}")


def _render_question_flow(workflow: DocumentQAWorkflow, top_k: int) -> None:
    st.subheader("2. Ask a question")
    indexed = bool(st.session_state.get("indexed"))
    question = st.text_input(
        "Question",
        placeholder="Example: What risks or action items are mentioned in these documents?",
        disabled=not indexed,
    )
    ask_clicked = st.button("Ask documents", disabled=not indexed or not question.strip())

    if not indexed:
        st.info("Index documents first, then ask a question.")
        return

    if ask_clicked:
        with st.spinner("Retrieving sources and preparing an answer..."):
            result = workflow.ask(question, top_k=top_k)
        _render_query_result(result)


def _render_query_result(result: QueryResult) -> None:
    st.markdown("### Answer")
    st.write(result.answer)

    if result.warnings:
        for warning in result.warnings:
            st.warning(warning)

    st.markdown("### Sources")
    if not result.sources:
        st.write("No source chunks were retrieved.")
        return

    for index, source in enumerate(result.sources, start=1):
        with st.expander(
            f"Source {index}: {_source_label(source)} (score {source.relevance_score:.2f})",
            expanded=index == 1,
        ):
            st.write(source.text)


def _source_label(source: RetrievedChunk) -> str:
    metadata = source.metadata
    parts = [str(metadata.get("source_name", "unknown source"))]
    if metadata.get("page"):
        parts.append(f"page {metadata['page']}")
    if metadata.get("sheet"):
        parts.append(f"sheet {metadata['sheet']}")
    if metadata.get("row_range"):
        parts.append(f"rows {metadata['row_range']}")
    if metadata.get("chunk_index"):
        parts.append(f"chunk {metadata['chunk_index']}")
    return ", ".join(parts)


if __name__ == "__main__":
    main()
