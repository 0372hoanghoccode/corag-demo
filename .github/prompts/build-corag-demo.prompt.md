---
agent: Describe what to build
description: Build a real Streamlit side-by-side demo for Traditional RAG vs CoRAG
---

Build a production-ready local demo app for technical presentation.

## Build target

Create these files with working code:

- app.py
- rag_engine.py
- corag_engine.py
- document_loader.py
- vectorstore.py
- requirements.txt
- docs/sample.pdf (or docs/sample.txt if PDF not available)

## Hard requirements

- CoRAG must be implemented manually as iterative retrieval (no corag package).
- Traditional RAG must perform single-pass retrieval and answer generation.
- Use Streamlit side-by-side layout to compare outputs.
- CoRAG UI must show step-by-step status updates while running.
- Document upload and indexing into local Chroma must work.
- Add robust try/except around all API-dependent calls.
- Use session_state to preserve results between reruns.
- Do not hardcode secrets.

## CoRAG control logic

At each step:

1. Evaluate if current context is sufficient for the original question.
2. If not sufficient, generate exactly one improved sub-query.
3. Retrieve new chunks with that sub-query.
4. Merge and deduplicate context.
5. Stop when sufficient or when reaching max_steps.

Generate final answer from accumulated context pool.

## Return contracts

Traditional RAG function returns:

- answer
- retrieved_docs
- steps
- tokens_used

CoRAG function returns:

- answer
- chain (step-by-step query and retrieved snippets)
- total_docs
- steps
- tokens_used

## Demo readiness checks

- App handles three consecutive questions without crashing.
- CoRAG shows additional retrieval steps for multi-hop questions.
- Metrics clearly compare time, steps, and retrieved docs.
- Code includes concise comments that help explain CoRAG loop in presentation.

## Sample questions for demo

- 1-hop: "VietAI Core đang chạy trên nền tảng nào và có bao nhiêu node Kubernetes?"
- 2-hop: "Model nào đang được nhóm Backend Engineering dùng nhiều nhất, và latency P95 của model đó là bao nhiêu?"
- 3-hop: "Người thiết kế kiến trúc VietAI Core là ai, họ đang phụ trách model nào lớn nhất trong hệ thống hiện tại, và người đó sẽ làm gì trong kế hoạch 2025?"

Use these questions with docs/sample.txt to ensure deterministic multi-hop behavior in live demos.
