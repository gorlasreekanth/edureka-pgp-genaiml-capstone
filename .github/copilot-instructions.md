# Copilot instructions

## Repository context

This repository is for an Edureka Generative AI and ML capstone. The authoritative project requirements are in `project-overview-guidelines.pdf`; `README.md` links to the planning docs that guide implementation.

The required solution is a Generative AI document Q&A application where users can upload enterprise documents and ask natural-language questions. The supported document types called out in the overview are PDF, TXT, CSV, and Excel.

Use `docs/decisions.md` as the running decision log for project assumptions, options considered, choices made, and the reasons behind them. Use `docs/project-plan.md` as the implementation handoff. Update these docs when major architecture, provider, tooling, or submission decisions change.

## Current project state

No application scaffold, dependency manifest, test suite, lint configuration, or build system exists yet. Do not assume npm, pytest, Streamlit, FastAPI, LangChain, LlamaIndex, or any other framework command until the corresponding files are added.

When project tooling is introduced, update this file with the exact commands for:

- installing dependencies
- running the app
- running the full test suite
- running a single test
- linting/formatting

## Target architecture

Future implementation should align with the workflow described in the overview PDF:

1. User interaction layer for document upload and natural-language questions.
2. Document ingestion for PDF, TXT, CSV, and Excel files.
3. Text preparation that chunks parsed document content for semantic search.
4. Embedding generation and vector-based knowledge storage.
5. Similarity retrieval for the user's query.
6. Retrieval-Augmented Generation that combines retrieved context with an LLM response.
7. Agent-based reasoning with clear roles for planning, retrieval, answer generation, and validation.
8. Reliability and safety controls for invalid inputs, retrieval misses, hallucination reduction, and unsafe outputs.

Keep these responsibilities separated in code so future sessions can modify ingestion, retrieval, generation, and agent behavior independently.

## Quality and human style

This is a hack/capstone project, so optimize for a working, explainable demo rather than a large production platform. User-facing UI text, generated answer templates, README content, architecture notes, and comments should sound natural and human-written: concise, clear, and specific to the document Q&A workflow.

Avoid generic AI-app boilerplate in documentation and prompts. When writing app copy or sample responses, prefer practical language that helps a reviewer understand what the system does, what evidence it used, and what limitations apply.

For generated answers, favor grounded, cautious responses over confident-sounding guesses. If retrieved context is missing or weak, the app should say so in plain language instead of inventing an answer.

## Submission requirements

The final deliverable must be zip-ready source code plus documentation. The documentation should cover setup, architecture, workflow, agent roles, deployment steps, limitations, and challenges faced during development.

Do not commit credentials, API keys, model secrets, or service tokens. If external LLM or embedding providers are added, use environment variables and provide a safe example configuration file instead of real secrets.
