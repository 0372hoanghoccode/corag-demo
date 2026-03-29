from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


def _extract_text_and_tokens(response: Any) -> Tuple[str, int]:
    if response is None:
        return "", 0

    if isinstance(response, str):
        return response, 0

    content = getattr(response, "content", "")
    if isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                chunks.append(str(item.get("text", "")))
            else:
                chunks.append(str(item))
        text = "\n".join(chunks)
    else:
        text = str(content)

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    tokens = 0
    metadata = getattr(response, "response_metadata", {}) or {}
    usage = metadata.get("token_usage") or metadata.get("usage") or {}
    if isinstance(usage, dict):
        tokens = int(usage.get("total_tokens", 0) or 0)

    return text.strip(), tokens


def run_rag(question: str, retriever, llm) -> Dict[str, Any]:
    """Traditional RAG: retrieve once, then generate one answer."""
    result: Dict[str, Any] = {
        "answer": "",
        "retrieved_docs": [],
        "steps": 1,
        "tokens_used": 0,
    }

    try:
        docs = retriever.invoke(question)
        doc_texts: List[str] = [doc.page_content for doc in docs]
        result["retrieved_docs"] = doc_texts

        context = "\n\n".join(doc_texts)
        prompt = (
            "Bạn là trợ lý kỹ thuật. Chỉ sử dụng context để trả lời câu hỏi. "
            "Nếu thiếu dữ liệu, hãy nói rõ phần còn thiếu.\n\n"
            f"Câu hỏi: {question}\n\n"
            f"Context:\n{context}\n\n"
            "Trả lời ngắn gọn, chính xác và có dẫn chiếu dữ kiện từ context."
        )

        response = llm.invoke(prompt)
        answer, tokens = _extract_text_and_tokens(response)

        result["answer"] = answer
        result["tokens_used"] = tokens
        return result
    except Exception as exc:
        result["answer"] = f"RAG execution error: {exc}"
        return result
