# Project decisions

This file is the running decision log for the capstone project. Use it to capture assumptions, options considered, choices made, and the reason behind each choice.

## Source requirement

The project overview PDF defines the target as a Generative AI document Q&A application for enterprise documents. Users should be able to upload documents, ask natural-language questions, and receive grounded answers generated through RAG and agent-based reasoning.

Submission should include complete source code and documentation explaining setup, architecture, agent roles, deployment steps, limitations, and challenges.

The PDF was reviewed at implementation kickoff and calls out the same core tasks: project foundation, user interaction layer, document ingestion, chunking for semantic search, vector storage, retrieval, RAG, agent-based reasoning, reliability controls, and deployment documentation.

## Current assumptions

- This is a hack/capstone project, so the first goal is a working, explainable demo rather than a production platform.
- The app should support PDF, TXT, CSV, and Excel documents because those formats are explicitly listed in the overview.
- We will avoid a separate frontend/backend for the MVP unless the project later needs multiple clients, authentication, or production APIs.
- User-facing text, documentation, prompts, and generated answer templates should sound natural and human-written.
- The app should be honest when retrieved context is weak or missing instead of inventing confident answers.

## Recommended choices

| Area | Choice | Reason |
|---|---|---|
| Language | Python | Best fit for document parsing, embeddings, vector search, and quick GenAI app development. |
| UI | Streamlit | Fastest way to build upload, question, answer, and source display flows for a demo. |
| App shape | Single Streamlit app with modular `src/` code | Keeps deployment simple while still separating ingestion, retrieval, RAG, agents, and safety logic. |
| LLM | Ollama Cloud API | Project decision; isolate behind a small client module so the rest of the app does not depend on API details. |
| Embeddings | SentenceTransformers local model | Avoids a second cloud dependency for embeddings and keeps setup easier to explain. |
| Vector store | Chroma | Simple local persistent vector store with metadata support, good for a hack project. |
| Document parsing | Custom loaders using `pypdf`, `pandas`, and `openpyxl` | Transparent and easy to explain in the capstone documentation. |
| Chunking | Recursive chunking with overlap | Better than naive fixed splitting while staying simple. |
| RAG pipeline | Custom Python pipeline | Easier to understand and document than starting with LangChain or LlamaIndex. |
| Agents | Lightweight custom agents | Satisfies the agentic AI requirement without adding CrewAI/LangGraph complexity. |
| Tests | pytest | Lightweight Python testing standard. |
| Dependency management | `requirements.txt` | Simple for a zip-ready capstone submission and easier to explain than Poetry or Conda. |
| Runtime configuration | Environment variables with `.env.example` | Keeps provider keys out of source while documenting the required settings. |

## Kickoff decisions

| Area | Decision | Reason |
|---|---|---|
| First UI/runtime | Streamlit local app | Best balance of speed, demo clarity, and zip-friendly submission. |
| First implementation scope | Complete vertical slice before polish | Keeps the project runnable while adding formats, agents, and documentation. |
| Ollama model | Configure through `OLLAMA_MODEL`, defaulting to a placeholder in `.env.example` | The exact account/model can be changed without code edits or committed secrets. |
| Retrieval fallback | Keep the first implementation on SentenceTransformers + Chroma | This matches the planned architecture and avoids adding a second retrieval path before the main flow is proven. |
| Deployment path | Document local Streamlit deployment first | Docker can be added later if time allows, but local run steps are enough for the capstone handoff. |
| Folder structure | Use `app.py`, `src/`, `tests/`, and `docs/` | Keeps ingestion, retrieval, generation, agents, and safety code separated without making the project feel overbuilt. |

## Implementation decisions

| Area | Decision | Reason |
|---|---|---|
| LLM placeholder behavior | Return a retrieval-only answer when `OLLAMA_MODEL` is still a placeholder | Reviewers can validate upload, parsing, chunking, vector search, and source display before model credentials are available, while still getting a useful provisional answer. |
| Validation | Surface warnings beside the answer | This keeps weak context, retrieval misses, and missing model settings visible instead of hiding them behind success messages. |
| Tests | Use fakes for workflow tests | Tests validate orchestration without downloading embedding models or calling external APIs. |
| Spreadsheet ingestion | Convert rows into readable text with row metadata | Simple, transparent, and sufficient for the capstone document Q&A workflow. |
| Fast local embeddings | Default to `local-hash`, with SentenceTransformers still available through `EMBEDDING_MODEL` | Local demos should not look stuck while downloading a model; reviewers can opt into stronger semantic retrieval when the environment is ready. |
| Ollama Cloud auth | Treat `https://ollama.com` as the Cloud host and require `OLLAMA_API_KEY` before making LLM calls | The direct Cloud API uses bearer-token auth, while local Ollama should still work without a key. |
| Used-source display | Ask the answer model to end with a parseable `Used sources:` line and show those sources separately from all retrieved candidates | Reviewers can see which chunks were selected by retrieval and which chunks the generated answer claims to rely on. |

## RAG and agent plan

Initial flow:

```text
User uploads files
 -> document ingestion
 -> chunking
 -> embedding generation
 -> Chroma vector storage
 -> user asks a question
 -> QueryPlannerAgent
 -> RetrievalAgent
 -> AnswerAgent using Ollama Cloud API
 -> ValidationAgent
 -> Streamlit shows answer, used sources, retrieved source candidates, and any confidence warning
```

Agent roles should stay simple at first:

| Agent | First implementation |
|---|---|
| QueryPlannerAgent | Rule-based cleanup/normalization of the user question and retrieval settings. |
| RetrievalAgent | Searches Chroma for relevant chunks and returns source metadata. |
| AnswerAgent | Builds a grounded prompt, calls Ollama Cloud API, and parses the model's declared used-source numbers. |
| ValidationAgent | Checks for missing context, empty answers, weak retrieval, missing source-use declarations, and unsupported-answer warnings. |

Only the answer agent needs an LLM call in the MVP. Keeping the planner and validator mostly rule-based should make the demo faster, cheaper, and easier to debug.

## Options considered

| Option | Why not first |
|---|---|
| React + FastAPI | More production-like, but too much frontend/backend wiring for the MVP. |
| LangChain | Powerful and popular, but adds abstraction before the basic RAG flow is working. |
| LlamaIndex | Strong for document Q&A, but less transparent for a learner-focused capstone. |
| CrewAI/LangGraph/AutoGen | Strong agent story, but more moving parts and higher demo risk. |
| Pinecone/Qdrant/Weaviate/Azure AI Search | Useful for production, but Chroma is simpler for local demo and submission packaging. |

## Open decisions

- Whether to add Docker once the local Streamlit workflow is complete.
- Whether to add a TF-IDF or keyword fallback later if local embedding setup becomes too heavy for the demo environment.
