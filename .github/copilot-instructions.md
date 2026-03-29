# Copilot Instructions for CoRAG Demo

This repository is dedicated to building a real Streamlit side-by-side demo of Traditional RAG vs CoRAG (Chain-of-Retrieval Augmented Generation).

## Non-negotiable rules

1. Do not use mock data for core demo behavior. Retrieval and generation must run for real.
2. CoRAG is not an installable package. Implement iterative retrieval logic manually.
3. Keep architecture simple and demo-friendly. Avoid over-engineering.
4. Never hardcode API keys. Load from .env or Streamlit secrets.
5. Wrap all model and retrieval calls with robust error handling.
6. Preserve live-demo reliability: clear user-visible errors, no silent failures.

## Required app behavior

- Build a Streamlit UI with two equal columns:
  - Left: Traditional RAG
  - Right: CoRAG with visible step-by-step reasoning status
- Support upload and indexing of PDF/TXT into local ChromaDB.
- Maintain session state so results do not disappear on rerun.
- Show comparison metrics: steps, docs, latency.

## Required implementation semantics

- Traditional RAG: single retrieval pass, single final generation.
- CoRAG: iterative loop of:
  - evaluate sufficiency of current context
  - generate one refined sub-query if context is insufficient
  - retrieve more chunks
  - merge with dedup
  - stop at sufficient=true or max_steps
- Final CoRAG answer must use the full accumulated context pool.

## Performance expectation

- CoRAG can be slower (often 3-5x) than traditional RAG due to multi-step LLM calls.
- Treat this as expected trade-off, not a bug.

## Preferred stack

- Python 3.10+
- streamlit, langchain, langchain-community, chromadb, pypdf, python-dotenv
- Model providers: OpenAI or Google Gemini
- Embeddings: OpenAIEmbeddings or HuggingFace all-MiniLM-L6-v2

## Deliverable checklist to satisfy before claiming done

- Upload PDF/TXT works and documents are indexed successfully.
- Traditional RAG returns answer and retrieved chunks.
- CoRAG displays retrieval chain and terminates safely.
- 2-hop or 3-hop questions trigger more than one retrieval step in CoRAG.
- No crash when running multiple consecutive demo questions.
- Include concise comments explaining CoRAG loop decisions.
