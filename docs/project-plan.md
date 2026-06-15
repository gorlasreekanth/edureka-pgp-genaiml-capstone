# Project plan

This plan is the handoff for the implementation session. Keep it current as the project moves from design to working code.

## Goal

Build a Streamlit-based Generative AI document Q&A app that supports PDF, TXT, CSV, and Excel uploads. The app should retrieve relevant document chunks, call Ollama Cloud API for grounded answers, show sources, and demonstrate lightweight agent-based reasoning.

## Planned implementation phases

1. Project foundation
   - Done: added Python dependency files.
   - Done: added `.env.example` for Ollama Cloud/local Ollama and local app settings.
   - Done: created the module layout under `src/`.
   - Done: added `.gitignore` entries for local secrets, caches, vector DB files, and generated artifacts.

2. Document ingestion
   - Done: implemented loaders for PDF, TXT, CSV, and Excel.
   - Done: normalized parsed content into a common document/chunk input shape.
   - Done: kept source metadata such as file name, page, sheet, row range, and document type when available.

3. Chunking and indexing
   - Done: added recursive chunking with overlap.
   - Done: generated embeddings using SentenceTransformers.
   - Done: stored chunks, embeddings, and metadata in Chroma.
   - Done: made indexing repeatable from Streamlit uploads.

4. Retrieval and RAG
   - Done: searched Chroma for relevant chunks.
   - Done: built a grounded prompt that tells the model to answer only from retrieved context.
   - Done: called an Ollama-compatible API through a small LLM client module.
   - Done: returned answer text, source chunks, and retrieval confidence signals.

5. Agent workflow
   - Done: added `QueryPlannerAgent`, `RetrievalAgent`, `AnswerAgent`, and `ValidationAgent`.
   - Done: kept planner and validator rule-based for the MVP.
   - Done: used only one LLM call per user question, inside the answer agent.

6. Streamlit UI
   - Done: provided document upload controls.
   - Done: showed indexing status in plain language.
   - Done: provided a question box and answer panel.
   - Done: showed sources and validation warnings beside the answer.

7. Reliability and safety
   - Done: rejected unsupported file types.
   - Done: handled empty uploads, empty questions, and documents with no extractable text.
   - Done: warned when retrieval confidence is weak or no context is found.
   - Done: avoided success-shaped fallbacks that hide missing model settings.

8. Documentation and packaging
   - Done: expanded `README.md` with setup, run commands, architecture, workflow, agent roles, deployment steps, limitations, and challenges.
   - Done: kept `docs/decisions.md` updated with major choices and reasons.
   - Done: documented what should be included in the final zip and what should be excluded.

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

Current status: the vertical slice is implemented for PDF, TXT, CSV, and Excel. The app can index uploaded files into Chroma, retrieve source chunks, call an Ollama-compatible model when configured, and return a retrieval-only provisional answer when model settings are placeholders. Local demos default to fast no-download hash embeddings, while SentenceTransformers remains available by setting `EMBEDDING_MODEL`.

## Implementation defaults

- Use Streamlit as the only user interface for the first complete demo.
- Use `requirements.txt` and `.env.example` rather than Poetry, Conda, or committed local settings.
- Keep provider-specific logic behind an Ollama client so credentials and model names stay in environment variables.
- Prefer clear placeholders over blocking for missing runtime choices, especially for the Ollama model name and API base URL.
- Commit in small local batches: planning baseline, project foundation, core RAG workflow, and final docs/tests.

## Validation checkpoints

- Done: after dependency files were added, installed from `requirements.txt`.
- Done: after core modules were added, ran `pytest`.
- Done: after the Streamlit entry point was wired, ran Python compile checks.
- Done: before each commit, checked `git status` so generated files were not included.

## Remaining nice-to-haves

- Add OCR for scanned PDFs if needed.
- Add Docker if the final submission needs containerized deployment.
- Add keyword or TF-IDF fallback retrieval if the embedding model is too heavy for the review machine.
- Add sample documents if the course submission allows non-sensitive demo data.
