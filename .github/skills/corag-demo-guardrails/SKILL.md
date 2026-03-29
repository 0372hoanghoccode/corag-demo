---
name: corag-demo-guardrails
description: Enforces implementation guardrails for the Streamlit Traditional RAG vs CoRAG demo.
---

# CoRAG Demo Guardrails

Use this skill whenever the task is about implementing, modifying, or reviewing this demo application.

## Always enforce

1. No mock retrieval or mock generation for core behavior.
2. CoRAG must be manual iterative retrieval logic.
3. Use robust try/except and user-visible errors.
4. Keep app architecture simple and presentation-friendly.
5. Do not hardcode API keys.

## Required behavior checklist

- Streamlit UI has two equal columns for Traditional RAG and CoRAG.
- CoRAG side shows retrieval chain progression in status updates.
- Document upload for PDF/TXT and indexing to local Chroma works.
- Session state preserves latest results across reruns.
- Comparison metrics for steps, documents, and latency are displayed.

## CoRAG loop reference

- Initial retrieval using original question.
- Evaluate sufficiency using current context.
- If insufficient, generate one refined sub-query.
- Retrieve additional chunks.
- Merge with deduplication.
- Stop at sufficient or max_steps.
- Final answer must use full accumulated context.

## Out-of-scope

- Authentication system
- External persistence database
- Cloud deployment pipeline
- Exact full-paper replication

## Done validation

- Multi-hop prompts trigger multiple CoRAG steps.
- Traditional RAG completes with one step.
- Three consecutive runs do not crash.
