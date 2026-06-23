from __future__ import annotations

import streamlit as st

from src.config import AppConfig
from src.llm import LLMClientError, LLMConfigurationError
from src.models import IndexResult, QueryResult, RetrievedChunk, normalize_source_indices
from src.rag import DocumentQAWorkflow
from src.validation import InputValidationError, validate_question, validate_uploaded_file


@st.cache_resource(show_spinner=False)
def _get_workflow(config: AppConfig) -> DocumentQAWorkflow:
    return DocumentQAWorkflow.from_config(config)


def main() -> None:
    config = AppConfig.from_env()
    workflow = _get_workflow(config)

    st.set_page_config(page_title="Enterprise Document Q&A")
    st.title("Enterprise Document Q&A")
    st.write(
        "Upload a few documents, ask a question, and get an answer that cites the passages it used."
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
        st.write(f"LLM host: `{config.ollama_runtime_label}`")
        st.caption(f"Ollama API base: `{config.ollama_api_base}`")
        top_k = st.slider(
            "Top K source chunks",
            min_value=1,
            max_value=10,
            value=config.retrieval_top_k,
            help="How many matching chunks to retrieve for each question.",
        )
        if config.embedding_model == "local-hash":
            st.info("Using fast local embeddings. Set `EMBEDDING_MODEL` to a sentence-transformer for stronger retrieval.")
        if config.has_configured_llm:
            st.success(f"{config.ollama_runtime_label} model: `{config.ollama_model}`")
        else:
            st.warning(
                f"{config.llm_configuration_issue} Retrieval still works, but you'll get a retrieval-only answer until `.env` is set."
            )
        if config.uses_ollama_cloud:
            key_status = "configured" if config.ollama_api_key else "missing"
            st.caption(f"Ollama Cloud API key: {key_status}")
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
            st.success("Cleared the indexed documents. Upload again to rebuild the index.")

    if index_clicked:
        limits = workflow.config.validation_limits
        files: list[tuple[str, bytes]] = []
        rejected: list[str] = []
        for file in uploaded_files:
            content = file.getvalue()
            try:
                validate_uploaded_file(file.name, content, limits)
            except InputValidationError as exc:
                rejected.append(str(exc))
                continue
            files.append((file.name, content))

        for message in rejected:
            st.error(message)

        if not files:
            st.warning("Nothing valid to index. Fix the errors above and try again.")
            return

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
        st.info("Upload a PDF, TXT, CSV, or Excel file to get started.")


def _render_index_result(result: IndexResult) -> None:
    if result.indexed_chunk_count > 0:
        st.success(
            f"Indexed {result.indexed_chunk_count} chunks from {result.document_count} document sections. "
            "Ready for questions."
        )
    else:
        st.warning("Nothing was indexed yet. Check the file errors below.")

    if result.errors:
        with st.expander("Files that need attention", expanded=True):
            for error in result.errors:
                st.write(f"- {error}")


def _render_question_flow(workflow: DocumentQAWorkflow, top_k: int) -> None:
    st.subheader("2. Ask a question")
    indexed = bool(st.session_state.get("indexed"))
    with st.form("question_form"):
        question = st.text_input(
            "Question",
            placeholder="Example: What risks or action items are mentioned in these documents?",
            disabled=not indexed,
        )
        ask_clicked = st.form_submit_button("Ask documents", disabled=not indexed)

    if not indexed:
        st.info("Index some documents first, then come back here to ask a question.")
        return

    if ask_clicked:
        try:
            cleaned_question = validate_question(
                question, workflow.config.validation_limits
            )
        except InputValidationError as exc:
            st.warning(str(exc))
            return
        with st.spinner("Retrieving sources and preparing an answer..."):
            try:
                result = workflow.ask(cleaned_question, top_k=top_k)
            except (LLMConfigurationError, LLMClientError) as exc:
                st.error(str(exc))
                return
        _render_query_result(result)


def _render_query_result(result: QueryResult) -> None:
    sources = result.sources
    used_source_indices = normalize_source_indices(
        getattr(result, "used_source_indices", []),
        len(sources),
    )

    st.markdown("### Answer")
    st.write(result.answer)

    if result.warnings:
        for warning in result.warnings:
            st.warning(warning)

    if used_source_indices:
        st.markdown("### Sources used in the answer")
        for source_number in used_source_indices:
            if not 1 <= source_number <= len(sources):
                continue
            source = sources[source_number - 1]
            with st.expander(
                f"Source {source_number}: {_source_label(source)} (score {source.relevance_score:.2f})",
                expanded=True,
            ):
                st.write(source.text)
    elif result.used_llm and sources:
        st.info("The model didn't flag any specific source. The retrieved candidates are listed below.")

    st.markdown("### Retrieved source candidates")
    if not sources:
        st.write("No source chunks were retrieved for this question.")
        return

    used_source_numbers = set(used_source_indices)
    for index, source in enumerate(sources, start=1):
        used_suffix = " - used" if index in used_source_numbers else ""
        with st.expander(
            f"Source {index}{used_suffix}: {_source_label(source)} (score {source.relevance_score:.2f})",
            expanded=index == 1 and not used_source_numbers,
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
