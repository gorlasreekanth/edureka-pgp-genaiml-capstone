# Project plan

This plan is the handoff for the implementation session. Keep it current as the project moves from design to working code.

## Goal

Build a Streamlit-based Generative AI document Q&A app that supports PDF, TXT, CSV, and Excel uploads. The app should retrieve relevant document chunks, call Ollama Cloud API for grounded answers, show sources, and demonstrate lightweight agent-based reasoning.

## Planned implementation phases

1. Project foundation
   - Add Python dependency files.
   - Add `.env.example` for Ollama Cloud and local app settings.
   - Create the initial module layout under `src/`.
   - Add `.gitignore` entries for local secrets, caches, vector DB files, and generated artifacts.

2. Document ingestion
   - Implement loaders for PDF, TXT, CSV, and Excel.
   - Normalize parsed content into a common document/chunk input shape.
   - Keep source metadata such as file name, page, sheet, row range, or document type when available.

3. Chunking and indexing
   - Add recursive chunking with overlap.
   - Generate embeddings using SentenceTransformers.
   - Store chunks, embeddings, and metadata in Chroma.
   - Make indexing repeatable from Streamlit uploads.

4. Retrieval and RAG
   - Search Chroma for relevant chunks.
   - Build a grounded prompt that tells the model to answer only from retrieved context.
   - Call Ollama Cloud API through a small LLM client module.
   - Return answer text, source chunks, and retrieval confidence signals.

5. Agent workflow
   - Add `QueryPlannerAgent`, `RetrievalAgent`, `AnswerAgent`, and `ValidationAgent`.
   - Keep planner and validator mostly rule-based for the MVP.
   - Use only one LLM call per user question at first, inside the answer agent.

6. Streamlit UI
   - Provide document upload controls.
   - Show indexing status in plain language.
   - Provide a question box and answer panel.
   - Show sources and any validation warnings beside the answer.

7. Reliability and safety
   - Reject unsupported file types.
   - Handle empty uploads, empty questions, and documents with no extractable text.
   - Warn when retrieval confidence is weak or no context is found.
   - Avoid success-shaped fallbacks that hide errors.

8. Documentation and packaging
   - Expand `README.md` with setup, run commands, architecture, workflow, agent roles, deployment steps, limitations, and challenges.
   - Keep `docs/decisions.md` updated with major choices and reasons.
   - Document what should be included in the final zip and what should be excluded.

## Implementation kickoff plan

Start with the smallest complete path through the system, then add file formats and polish around it. This keeps the demo honest: each phase should leave the app runnable or the core modules testable.

1. Foundation first
   - Add `requirements.txt`, `.env.example`, `.gitignore`, and the initial `src/` package layout.
   - Add `app.py` as the Streamlit entry point, even if the first version only wires upload and status messages.
   - Add config loading for Ollama Cloud settings and local vector store paths without committing secrets.

2. Build the vertical slice with TXT first
   - Parse uploaded TXT files into normalized document objects with source metadata.
   - Chunk the parsed text with overlap and stable chunk IDs.
   - Embed chunks with SentenceTransformers and persist them in Chroma.
   - Retrieve top matching chunks for a question and show the retrieved source text before adding the LLM call.

3. Add PDF support next
   - Reuse the same normalized document shape.
   - Preserve page numbers in metadata so source citations are useful in the UI.
   - Handle PDFs with no extractable text as a clear user-facing warning.

4. Add the answer path
   - Add a small Ollama Cloud client module.
   - Build a grounded prompt that asks the model to answer only from retrieved context.
   - Return answer text, source chunks, and warnings from one workflow function.

5. Wrap the workflow with lightweight agents
   - Add `QueryPlannerAgent`, `RetrievalAgent`, `AnswerAgent`, and `ValidationAgent` around the working retrieval and answer path.
   - Keep planner and validator rule-based for the MVP so the only required LLM call is answer generation.

6. Expand and harden
   - Add CSV and Excel ingestion.
   - Add tests for loaders, chunking, retrieval misses, and validation warnings.
   - Expand README documentation once the commands and app behavior are real.

## First coding batch

The first implementation batch should create the foundation and enough logic to parse, chunk, embed, store, and retrieve TXT content. It does not need to call the LLM yet. A good stopping point is:

```text
streamlit app starts
 -> TXT file uploads successfully
 -> chunks are indexed in Chroma
 -> a question retrieves relevant chunks
 -> UI shows source snippets and clear status messages
```

That gives the project a working base before adding Ollama Cloud, PDF, CSV, Excel, agent wrappers, and final documentation.

## Expected module shape

```text
app.py
src/
  agents/
  config.py
  ingestion/
  llm/
  rag/
  retrieval/
  safety/
tests/
docs/
  decisions.md
  project-plan.md
```

This shape may change during implementation, but keep the responsibilities separated so future sessions can work on one area without rewriting the whole app.

## First implementation target

The next session should aim for a working vertical slice:

```text
Upload TXT or PDF
 -> parse text
 -> chunk text
 -> embed and store in Chroma
 -> ask question
 -> retrieve chunks
 -> call Ollama Cloud API
 -> show answer and sources in Streamlit
```

After the vertical slice works, add CSV/Excel support, validation improvements, tests, and final documentation.

## Implementation defaults

- Use Streamlit as the only user interface for the first complete demo.
- Use `requirements.txt` and `.env.example` rather than Poetry, Conda, or committed local settings.
- Keep provider-specific logic behind an Ollama client so credentials and model names stay in environment variables.
- Prefer clear placeholders over blocking for missing runtime choices, especially for the Ollama model name and API base URL.
- Commit in small local batches: planning baseline, project foundation, core RAG workflow, and final docs/tests.

## Validation checkpoints

- After dependency files are added, install from `requirements.txt`.
- After core modules are added, run `pytest`.
- After the Streamlit entry point is wired, run a Python import/compile check and verify the app can be started locally.
- Before each commit, check `git status` so unrelated or generated files are not included.
