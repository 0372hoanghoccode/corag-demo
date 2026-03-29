from __future__ import annotations

import json
import math
import re
import unicodedata
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

MAX_REQUIRED_PARTS = 5


def _split_required_parts(question: str, max_parts: int = MAX_REQUIRED_PARTS) -> List[str]:
    parts = [piece.strip() for piece in re.split(r"\svà\s|,", question) if piece.strip()]
    return parts[:max_parts]


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", normalized.lower()).strip()


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

    metadata = getattr(response, "response_metadata", {}) or {}
    usage = metadata.get("token_usage") or metadata.get("usage") or {}
    tokens = int(usage.get("total_tokens", 0) or 0) if isinstance(usage, dict) else 0
    return text.strip(), tokens


def _parse_json_payload(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        return {}

    fence_match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", stripped, re.DOTALL | re.IGNORECASE)
    if fence_match:
        stripped = fence_match.group(1).strip()

    try:
        return json.loads(stripped)
    except Exception:
        pass

    match = re.search(r"\{.*?\}", stripped, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return {}

    return {}


def _derive_required_parts(question: str, llm=None, use_llm: bool = False) -> List[str]:
    if not use_llm:
        return _split_required_parts(question)

    prompt = f"""
Phân rã câu hỏi sau thành các vế thông tin cần trả lời độc lập.

Câu hỏi: {question}

Trả về JSON:
{{
    "parts": ["vế 1", "vế 2", "vế 3"]
}}
""".strip()

    response = llm.invoke(prompt)
    text, _ = _extract_text_and_tokens(response)
    parsed = _parse_json_payload(text)
    parts = parsed.get("parts", []) if isinstance(parsed, dict) else []

    if isinstance(parts, list):
        normalized = [str(item).strip() for item in parts if str(item).strip()]
        if normalized:
            return normalized[:MAX_REQUIRED_PARTS]

    # Fallback heuristic for Vietnamese multi-clause questions.
    return _split_required_parts(question)


def _evaluate_context(question: str, required_parts: List[str], context: str, llm) -> Tuple[Dict[str, Any], int]:
    # Hard mode: only skip the evaluator when every required part is covered with very high confidence.
    if context.strip():
        covered = _parts_covered_by_context(required_parts, context)
        if covered >= len(required_parts) and len(required_parts) <= 2:
            return (
                {
                    "sufficient": True,
                    "missing_parts": [],
                    "evidence_map": [
                        {
                            "part": part,
                            "covered": True,
                            "evidence": "Matched by heuristic coverage fast-path.",
                        }
                        for part in required_parts
                    ],
                    "reasoning": "heuristic-fast-path",
                    "sub_query": None,
                },
                0,
            )

    serialized_parts = "\n".join(f"- {part}" for part in required_parts)
    prompt = f"""
Bạn đang trả lời câu hỏi gốc: "{question}"

Các vế bắt buộc phải trả lời đầy đủ:
{serialized_parts}

Context đã thu thập:
{context}

Đánh giá NGHIÊM KHẮC:
1) Với TỪNG vế trong danh sách bắt buộc, cho biết đã có dữ kiện cụ thể trong context chưa.
2) Chỉ trả "sufficient": true khi TẤT CẢ các vế có evidence rõ ràng.
3) Nếu còn thiếu, liệt kê missing_parts đúng theo tên vế còn thiếu.
4) Nếu chưa đủ, tạo đúng MỘT sub-query tập trung vào phần thiếu quan trọng nhất.

Trả về JSON đúng schema:
{{
  "sufficient": bool,
  "missing_parts": ["..."],
    "evidence_map": [{{"part": "...", "covered": bool, "evidence": "..."}}],
  "reasoning": "...",
  "sub_query": "..." hoặc null
}}
""".strip()

    response = llm.invoke(prompt)
    text, tokens = _extract_text_and_tokens(response)
    parsed = _parse_json_payload(text)

    if not parsed:
        lowered = text.lower()
        fallback_sufficient = "true" in lowered and "sufficient" in lowered
        parsed = {
            "sufficient": fallback_sufficient,
            "missing_parts": [],
            "reasoning": text,
            "sub_query": None,
        }

    return parsed, tokens


def _parts_covered_by_context(required_parts: List[str], context_text: str) -> int:
    if not required_parts or not context_text.strip():
        return 0

    context_tokens = set(re.findall(r"\w+", _normalize_text(context_text)))
    covered = 0
    for part in required_parts:
        part_tokens = set(re.findall(r"\w+", _normalize_text(part)))
        if not part_tokens:
            continue
        overlap = len(part_tokens.intersection(context_tokens))
        overlap_ratio = overlap / max(1, len(part_tokens))
        has_numeric_token = any(token.isdigit() for token in part_tokens)
        # Numeric parts are stricter because partial matches often miss the actual value.
        if has_numeric_token:
            if overlap_ratio >= 1.0:
                covered += 1
            continue

        if overlap_ratio >= 0.9:
            covered += 1
    return covered


def _is_sufficient(
    eval_result: Dict[str, Any],
    required_parts: List[str],
    has_context: bool,
    context_text: str,
    embeddings: Optional[Any] = None,
) -> bool:
    if not has_context:
        return False

    llm_says_sufficient = bool(eval_result.get("sufficient", False))

    missing_parts = eval_result.get("missing_parts", [])
    has_missing = isinstance(missing_parts, list) and any(str(item).strip() for item in missing_parts)

    # Only trust a bare LLM sufficiency signal for short questions.
    # Multi-hop questions must still show concrete evidence before stopping.
    if llm_says_sufficient and not has_missing and len(required_parts) <= 2:
        return True

    if not llm_says_sufficient:
        return False

    evidence_map = eval_result.get("evidence_map", [])
    if isinstance(evidence_map, list) and evidence_map:
        covered = sum(1 for item in evidence_map if bool(item.get("covered", False)))
        if covered >= len(required_parts):
            return True

    if llm_says_sufficient and len(required_parts) > 2:
        return False

    # Last fallback: check lexical overlap if the LLM signaled sufficiency but evidence was missing.
    covered = _parts_covered_by_context(required_parts, context_text)
    return covered >= len(required_parts)


def _fallback_sub_query(question: str, missing_parts: List[str]) -> str:
    focus = "; ".join(missing_parts[:2]) if missing_parts else "phần thông tin còn thiếu"
    return f"Tìm dữ kiện cụ thể để trả lời: {focus}. Câu hỏi gốc: {question}"


def _generate_sub_query_candidates(
    question: str,
    missing_parts: List[str],
    context: str,
    llm,
    max_candidates: int = 3,
) -> List[str]:
    missing_serialized = "\n".join(f"- {part}" for part in missing_parts) if missing_parts else "- Chưa rõ"
    prompt = f"""
Bạn đang thực hiện iterative retrieval cho câu hỏi gốc:
{question}

Phần thông tin còn thiếu:
{missing_serialized}

Context hiện tại:
{context}

Hãy đề xuất tối đa {max_candidates} sub-query khác nhau để truy xuất phần còn thiếu.
Mỗi sub-query phải ngắn gọn và cụ thể.

Trả về JSON:
{{
  "candidates": ["query 1", "query 2", "query 3"]
}}
""".strip()

    response = llm.invoke(prompt)
    text, _ = _extract_text_and_tokens(response)
    parsed = _parse_json_payload(text)
    candidates = parsed.get("candidates", []) if isinstance(parsed, dict) else []

    if not isinstance(candidates, list):
        return []

    normalized: List[str] = []
    for item in candidates:
        candidate = str(item).strip()
        if candidate and candidate not in normalized:
            normalized.append(candidate)

    return normalized[:max_candidates]


def _extract_retriever_embeddings(retriever) -> Optional[Any]:
    vectorstore = getattr(retriever, "vectorstore", None)
    candidates = [
        getattr(vectorstore, "_embedding_function", None),
        getattr(vectorstore, "embedding_function", None),
        getattr(retriever, "embedding_function", None),
    ]
    for embedding in candidates:
        if embedding is not None and hasattr(embedding, "embed_query"):
            return embedding
    return None


def _candidate_score(
    query: str,
    gain_docs: List[str],
    missing_parts: List[str],
    embeddings: Optional[Any] = None,
) -> int:
    score = float(len(gain_docs) * 100)
    if not missing_parts:
        return int(score)

    q_tokens = set(re.findall(r"\w+", query.lower()))
    missing_tokens = set(re.findall(r"\w+", " ".join(missing_parts).lower()))
    score += len(q_tokens.intersection(missing_tokens))

    if embeddings and hasattr(embeddings, "embed_query"):
        try:
            q_emb = embeddings.embed_query(query)
            m_emb = embeddings.embed_query(" ".join(missing_parts))
            if q_emb and m_emb and len(q_emb) == len(m_emb):
                dot = sum(float(a) * float(b) for a, b in zip(q_emb, m_emb))
                q_norm = math.sqrt(sum(float(v) * float(v) for v in q_emb))
                m_norm = math.sqrt(sum(float(v) * float(v) for v in m_emb))
                if q_norm > 0 and m_norm > 0:
                    cos = dot / (q_norm * m_norm)
                    score += max(0.0, cos) * 50.0
        except Exception:
            pass

    return int(score)


def _rerank_context(question: str, context_pool: List[str], top_n: int = 6) -> List[str]:
    if not context_pool:
        return []

    q_tokens = set(re.findall(r"\w+", _normalize_text(question)))
    if not q_tokens:
        return context_pool[:top_n]

    scored: List[Tuple[float, int, str]] = []
    for idx, chunk in enumerate(context_pool):
        c_tokens = set(re.findall(r"\w+", _normalize_text(chunk)))
        lexical_overlap = len(q_tokens.intersection(c_tokens))
        recency_bonus = idx / max(1, len(context_pool))
        scored.append((lexical_overlap + recency_bonus, -idx, chunk))

    scored.sort(reverse=True)
    reranked = [item[2] for item in scored[:top_n]]
    return reranked


def _retrieve_docs(retriever, query: str, k_override: Optional[int] = None):
    if k_override and hasattr(retriever, "vectorstore") and hasattr(retriever.vectorstore, "similarity_search"):
        return retriever.vectorstore.similarity_search(query, k=int(k_override))
    return retriever.invoke(query)


def _final_answer(question: str, context: str, llm) -> Tuple[str, int]:
    prompt = f"""
Bạn là trợ lý kỹ thuật. Dùng toàn bộ context đã tích lũy để trả lời câu hỏi gốc.
Nếu còn thiếu dữ liệu, nêu rõ phần thiếu thay vì suy đoán.

Câu hỏi: {question}

Context tổng hợp:
{context}

Trả lời ngắn gọn, chính xác, có cấu trúc.
""".strip()

    response = llm.invoke(prompt)
    return _extract_text_and_tokens(response)


def run_corag(
    question: str,
    retriever,
    llm,
    max_steps: int = 4,
    first_step_k: Optional[int] = 1,
    step_k: Optional[int] = 2,
    use_llm_part_decomposition: bool = False,
    enable_candidate_generation: bool = False,
    step_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """CoRAG iterative retrieval loop with strict max_steps termination."""
    chain: List[Dict[str, Any]] = []
    context_pool: List[str] = []
    seen: Set[str] = set()
    seen_queries: Set[str] = set()
    current_query = question
    pending_query: Optional[str] = None
    pending_docs = None
    total_tokens = 0
    embeddings = _extract_retriever_embeddings(retriever)

    try:
        required_parts = _derive_required_parts(
            question,
            llm=llm,
            use_llm=use_llm_part_decomposition,
        )

        for step in range(1, max_steps + 1):
            k_for_step = first_step_k if step == 1 else step_k
            if step_callback is not None:
                step_callback(
                    {
                        "step": step,
                        "stage": "retrieve",
                        "query": current_query,
                        "reasoning": "Đang truy xuất tài liệu cho bước hiện tại...",
                        "retrieved": [],
                        "sufficient": False,
                        "missing_parts": [],
                        "required_parts": required_parts,
                        "selected_sub_query": None,
                        "rejected_queries": [],
                    }
                )

            if pending_query == current_query and pending_docs is not None:
                docs = pending_docs
                pending_query = None
                pending_docs = None
            else:
                docs = _retrieve_docs(retriever, current_query, k_override=k_for_step)
            retrieved: List[str] = []

            for doc in docs:
                content = doc.page_content.strip()
                if content and content not in seen:
                    seen.add(content)
                    context_pool.append(content)
                    retrieved.append(content)

            reranked_context = _rerank_context(question, context_pool)
            current_context = "\n\n".join(reranked_context)
            eval_result, eval_tokens = _evaluate_context(question, required_parts, current_context, llm)
            total_tokens += eval_tokens

            sufficient = _is_sufficient(
                eval_result,
                required_parts,
                has_context=bool(reranked_context),
                context_text=current_context,
            )
            reasoning = str(eval_result.get("reasoning", "Không có reasoning từ evaluator."))
            missing_parts = eval_result.get("missing_parts", [])
            if not isinstance(missing_parts, list):
                missing_parts = []

            sub_query = eval_result.get("sub_query")

            selected_sub_query: Optional[str] = None
            rejected_queries: List[Dict[str, str]] = []
            best_preview_docs = None

            if not sufficient:
                candidate_queries: List[str] = []
                if sub_query and str(sub_query).strip():
                    candidate_queries.append(str(sub_query).strip())

                if enable_candidate_generation:
                    generated = _generate_sub_query_candidates(
                        question=question,
                        missing_parts=missing_parts,
                        context=current_context,
                        llm=llm,
                        max_candidates=3,
                    )
                    for candidate in generated:
                        if candidate not in candidate_queries:
                            candidate_queries.append(candidate)

                scored_candidates: List[Tuple[int, str, Any]] = []
                for candidate in candidate_queries:
                    normalized = candidate.strip()
                    if not normalized:
                        continue
                    if normalized == current_query or normalized in seen_queries:
                        rejected_queries.append({"query": normalized, "reason": "duplicate"})
                        continue

                    preview_docs = _retrieve_docs(retriever, normalized, k_override=step_k)
                    gain_docs: List[str] = []
                    for doc in preview_docs:
                        content = doc.page_content.strip()
                        if content and content not in seen:
                            gain_docs.append(content)

                    score = _candidate_score(
                        normalized,
                        gain_docs,
                        missing_parts,
                        embeddings=embeddings,
                    )
                    scored_candidates.append((score, normalized, preview_docs))

                scored_candidates.sort(key=lambda item: item[0], reverse=True)

                if scored_candidates:
                    best_score, selected_sub_query, best_preview_docs = scored_candidates[0]
                    for _, rejected_query, _ in scored_candidates[1:]:
                        rejected_queries.append({"query": rejected_query, "reason": "lower_score"})
                    if best_score <= 0:
                        rejected_queries.append({"query": selected_sub_query, "reason": "no_context_gain"})
                        selected_sub_query = None
                        best_preview_docs = None

                if not selected_sub_query:
                    fallback_query = _fallback_sub_query(question, missing_parts).strip()
                    if fallback_query and fallback_query not in seen_queries and fallback_query != current_query:
                        selected_sub_query = fallback_query
                    else:
                        selected_sub_query = None

            step_entry = {
                "step": step,
                "query": current_query,
                "reasoning": reasoning,
                "retrieved": retrieved,
                "sufficient": sufficient,
                "missing_parts": missing_parts,
                "required_parts": required_parts,
                "selected_sub_query": selected_sub_query,
                "rejected_queries": rejected_queries,
            }
            chain.append(step_entry)

            if step_callback is not None:
                step_callback(step_entry)

            if sufficient:
                break

            if step_callback is not None:
                step_callback(
                    {
                        "step": step,
                        "stage": "evaluate",
                        "query": current_query,
                        "reasoning": "Đang đánh giá context và chọn sub-query nếu cần...",
                        "retrieved": retrieved,
                        "sufficient": sufficient,
                        "missing_parts": missing_parts,
                        "required_parts": required_parts,
                        "selected_sub_query": selected_sub_query,
                        "rejected_queries": rejected_queries,
                    }
                )

            normalized_query = str(selected_sub_query).strip() if selected_sub_query else ""
            if not normalized_query:
                break
            if normalized_query in seen_queries:
                break

            seen_queries.add(normalized_query)
            pending_query = normalized_query
            pending_docs = best_preview_docs
            current_query = normalized_query

        final_context = "\n\n".join(_rerank_context(question, context_pool, top_n=8))
        if step_callback is not None:
            step_callback(
                {
                    "step": len(chain) + 1,
                    "stage": "final",
                    "query": question,
                    "reasoning": "Đang tổng hợp câu trả lời cuối cùng từ toàn bộ context...",
                    "retrieved": [],
                    "sufficient": True,
                    "missing_parts": [],
                    "required_parts": required_parts,
                    "selected_sub_query": None,
                    "rejected_queries": [],
                }
            )
        answer, final_tokens = _final_answer(question, final_context, llm)
        total_tokens += final_tokens

        return {
            "answer": answer,
            "chain": chain,
            "total_docs": len(context_pool),
            "steps": len(chain),
            "tokens_used": total_tokens,
        }
    except Exception as exc:
        return {
            "answer": f"CoRAG execution error: {exc}",
            "chain": chain,
            "total_docs": len(context_pool),
            "steps": len(chain),
            "tokens_used": total_tokens,
        }
