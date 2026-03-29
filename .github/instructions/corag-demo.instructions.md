---
applyTo: "**"
---

When working in this repository, follow these implementation constraints unless the user explicitly overrides them.

## Goal

Build and maintain a real Streamlit comparison app between Traditional RAG and CoRAG.

## File layout to preserve

- app.py
- rag_engine.py
- corag_engine.py
- document_loader.py
- vectorstore.py
- requirements.txt
- docs/

## Functional requirements

1. document_loader.py

- Load PDF and TXT from docs folder and uploaded files.
- Use RecursiveCharacterTextSplitter with chunk_size=500 and chunk_overlap=100.
- Ensure metadata contains source and page when available.

2. vectorstore.py

- Use local Chroma persistence under ./chroma_db.
- Expose retriever factory that supports configurable k.
- Prefer HuggingFace embeddings for keyless demos, but allow OpenAI when configured.

3. rag_engine.py

- Retrieve once only.
- Generate one answer from retrieved chunks.
- Return structured dict with answer, retrieved_docs, steps, tokens_used.

4. corag_engine.py

- Implement iterative retrieval loop (no external corag package).
- At each step: evaluate sufficiency, create one sub-query when needed, retrieve, dedup, continue.
- Always enforce max_steps termination.
- Return structured dict with answer, chain, total_docs, steps, tokens_used.

5. app.py

- Two side-by-side columns for RAG and CoRAG.
- CoRAG column must display per-step status updates in real time.
- Inputs: question, max_steps slider, retrieval k slider, file uploader, run button.
- Keep outputs in session_state.
- Display visual comparison metrics.

## Reliability and UX

- Wrap all external calls (LLM, embeddings, retrieval) in try/except.
- Show actionable error messages via Streamlit, do not crash.
- Keep variable names and UI text clear for technical presentations.

## Scope guardrails

- Do not add authentication.
- Do not add external database for history.
- Do not add deployment workflow unless asked.
- Do not claim parity with full research paper implementation.
