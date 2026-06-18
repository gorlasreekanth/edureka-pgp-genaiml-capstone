# edureka-pgp-genaiml-capstone

Generative AI and ML capstone project for an enterprise document Q&A application using RAG and lightweight agent-based reasoning.

The app lets a user upload PDF, TXT, CSV, and Excel files, builds a local vector index, retrieves relevant source chunks for a question, and uses an Ollama-compatible chat API to generate a grounded answer when model settings are configured. It also separates the source chunks retrieved for the LLM from the specific sources the LLM says it used. If the model is still a placeholder, the app returns a retrieval-only answer from the best matching chunks and explains what is missing.

## Setup

Use Python 3.12 or a compatible modern Python 3 version.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` before asking the LLM to generate final answers:

```text
OLLAMA_API_BASE=https://ollama.com
OLLAMA_API_KEY=your-ollama-cloud-api-key
OLLAMA_MODEL=gpt-oss:120b
OLLAMA_TIMEOUT_SECONDS=120
```

For Ollama Cloud, create an API key from your Ollama account settings and choose a Cloud model from the Ollama model library. Keep the host as `https://ollama.com`; the app calls `/api/chat` and sends the key as a bearer token. Do not commit `.env`.

For local Ollama, switch the same settings to values like:

```text
OLLAMA_API_BASE=http://localhost:11434
OLLAMA_API_KEY=
OLLAMA_MODEL=llama3.1
```

The default `EMBEDDING_MODEL=local-hash` is a fast no-download option for local demos. For stronger semantic retrieval, set `EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2`; the first upload may take longer while the model downloads and loads.

## Run the app

```powershell
.\.venv\Scripts\streamlit run app.py
```

Then open the local Streamlit URL, upload one or more supported documents, click **Index documents**, and ask a question. For a quick review, use the safe demo files in `samples\`.

Use the sidebar **Top K source chunks** slider to choose how many candidate chunks are sent to the LLM for each answer. `RETRIEVAL_TOP_K` in `.env` sets the default slider value. The answer panel marks which retrieved source numbers the LLM reported using.

## Demo checklist

1. Start the app with `.\.venv\Scripts\streamlit run app.py`.
2. Confirm the sidebar shows the expected Ollama runtime and model. For Ollama Cloud, the API key status should be `configured`.
3. Upload `samples\enterprise_rollout_memo.txt` and `samples\support_metrics.csv`.
4. Click **Index documents** and confirm chunks were indexed.
5. Ask: `What is the highest launch risk and who owns the next action?`
6. Confirm the answer uses the uploaded content, shows warnings only when appropriate, and separates **Sources used in the answer** from **Retrieved source candidates**.

## Run tests

```powershell
.\.venv\Scripts\python -m pytest
```

Run one test file or test function:

```powershell
.\.venv\Scripts\python -m pytest tests\test_ingestion.py
.\.venv\Scripts\python -m pytest tests\test_workflow.py::test_workflow_indexes_txt_and_returns_answer
```

## Architecture

```text
app.py
 -> src.ingestion loads PDF, TXT, CSV, and Excel content
 -> src.rag.chunking prepares overlapping chunks
 -> src.rag.embeddings creates local hash or SentenceTransformers vectors
 -> src.retrieval stores and searches chunks in Chroma
 -> src.agents plans, retrieves, answers, and validates
 -> src.llm calls Ollama Cloud or a local Ollama-compatible chat API
```

The code is intentionally small and modular so ingestion, retrieval, generation, and validation can be changed independently during the capstone.

## Workflow

1. Upload documents in Streamlit.
2. Parse each file into normalized text with source metadata such as file name, page, sheet, row range, and chunk number.
3. Split text into overlapping chunks.
4. Embed chunks with the configured embedding provider.
5. Store chunks and metadata in a local Chroma collection.
6. Normalize the question, retrieve top matching chunks, and pass that context to the answer agent.
7. Generate the answer with Ollama Cloud or local Ollama when configured, asking the model to cite source numbers and declare which retrieved chunks it used. If the LLM is not configured, show a retrieval-only answer from the best matching chunks.
8. Validate the result for missing context, weak retrieval, empty answers, and placeholder LLM settings.

## Agent roles

| Agent | Role |
|---|---|
| QueryPlannerAgent | Cleans the user question and chooses retrieval settings. |
| RetrievalAgent | Searches the vector store for relevant document chunks. |
| AnswerAgent | Builds a grounded prompt, asks the LLM to cite used source numbers, calls Ollama Cloud or local Ollama, or produces a retrieval-only answer when the LLM is not configured. |
| ValidationAgent | Adds warnings for missing context, weak retrieval, empty responses, or placeholder LLM settings. |

Only the answer agent makes an LLM call in the current implementation. The planner and validator are rule-based so the demo remains easy to explain and debug. When LLM settings are placeholders, the answer agent still returns the most relevant retrieved passages as a provisional answer.

## Deployment

The current deployment path is a local Streamlit run:

```powershell
.\.venv\Scripts\streamlit run app.py
```

For a simple review machine, install dependencies, copy `.env.example` to `.env`, configure the model settings, and run the command above. The local Chroma database is created under `chroma_db` by default and should not be included in the source zip unless a reviewer specifically asks for sample indexed data.

## Submission packaging

Include source files, docs, tests, `samples\`, `.env.example`, `requirements.txt`, and `project-overview-guidelines.pdf`.

Do not include `.env`, `.venv`, `chroma_db`, `data`, `uploads`, caches, or local editor files.

Before zipping, run:

```powershell
git check-ignore -v .env .venv chroma_db data uploads
git --no-pager status --short --ignored
.\.venv\Scripts\python -m pytest
```

The zip should contain the code and sample inputs, but not local secrets, generated vector databases, uploaded private documents, or virtual environment files.

## Limitations and challenges

- Scanned PDFs without extractable text are rejected with a clear warning; OCR is not included yet.
- The first retrieval path uses SentenceTransformers plus Chroma only; there is no keyword fallback yet.
- `local-hash` embeddings are fast and offline-friendly, but less semantic than SentenceTransformers.
- Ollama Cloud needs `OLLAMA_API_KEY`; local Ollama can run without a key. The model name and endpoint stay in environment settings because the exact runtime account can differ by reviewer.
- Used-source labels depend on the LLM following the requested `Used sources:` line. If it does not, the app warns and still shows all retrieved source candidates.
- CSV and Excel files are converted into row-oriented text, which is explainable but not optimized for very large spreadsheets.
- The app is designed for a capstone demo, not multi-user production hosting or authenticated enterprise deployment.

## Planning docs

- `docs/decisions.md` captures project assumptions, options considered, choices made, and the reasons behind them.
- `docs/project-plan.md` captures the planned implementation phases and next-session handoff.