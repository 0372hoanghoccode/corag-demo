from __future__ import annotations

import hashlib
import os
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv

from corag_engine import run_corag
from document_loader import load_documents_from_docs_folder, load_documents_from_uploads
from eval import simple_answer_score
from rag_engine import run_rag
from vectorstore import DEFAULT_PERSIST_DIR, get_retriever, index_documents
from llm_factory import create_llm, describe_llm

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent

SAMPLE_QUESTIONS = [
    "Cluster nào ghi nhận nhiều incident nhất trong Q2 2025, người phụ trách cluster đó đang triển khai dự án gì, và KPI của dự án đó là gì?",
    "Model nào có latency P95 cao nhất theo benchmark mới nhất, đội nào sử dụng model đó nhiều nhất, cluster nào đang chạy model đó, và ai là người phụ trách cluster đó?",
    "Đội nào chịu ảnh hưởng nhiều nhất từ incident Q2, model chính của đội đó là gì, latency P95 của model đó theo báo cáo MỚI NHẤT là bao nhiêu, ai phụ trách model đó, và deadline dự án của người đó là tháng mấy?",
]


@st.cache_resource
def _get_cached_llm(provider: str):
    return create_llm(provider)


def _resolve_embedding_provider(llm_provider: str) -> str:
    if llm_provider == "openai":
        return "openai"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "huggingface"


def _init_state() -> None:
    defaults = {
        "question": "",
        "rag_result": None,
        "corag_result": None,
        "rag_time": 0.0,
        "corag_time": 0.0,
        "indexed_chunks": 0,
        "retriever_version": 0,
        "run_mode": "CoRAG only",
        "corag_live_steps": [],
        "persist_directory": "",
        "embedding_provider": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _get_persist_directory(provider: str) -> str:
    persist_directory = st.session_state.get("persist_directory", "")
    selected_provider = provider or "huggingface"
    if st.session_state.get("embedding_provider", "") != selected_provider:
        st.session_state["embedding_provider"] = selected_provider
        st.session_state["persist_directory"] = ""
        st.session_state["indexed_chunks"] = 0
        st.session_state["retriever_version"] = st.session_state.get("retriever_version", 0) + 1
        persist_directory = ""

    if not persist_directory:
        persist_directory = str(Path(DEFAULT_PERSIST_DIR) / f"{selected_provider}-session-{uuid.uuid4().hex}")
        st.session_state["persist_directory"] = persist_directory
    Path(persist_directory).mkdir(parents=True, exist_ok=True)
    return persist_directory


def _docs_fingerprint(docs_dir: Path) -> str:
    hasher = hashlib.md5()
    if not docs_dir.exists():
        return "missing"

    for file_path in sorted(docs_dir.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in {".pdf", ".txt"}:
            continue
        stat = file_path.stat()
        hasher.update(str(file_path.relative_to(docs_dir)).encode("utf-8"))
        hasher.update(str(stat.st_size).encode("utf-8"))
        hasher.update(str(int(stat.st_mtime)).encode("utf-8"))
    return hasher.hexdigest()


def _sync_docs_state() -> None:
    current_fingerprint = _docs_fingerprint(PROJECT_ROOT / "docs")
    if st.session_state.get("docs_fingerprint") != current_fingerprint:
        st.session_state["docs_fingerprint"] = current_fingerprint
        st.session_state["indexed_chunks"] = 0
        st.session_state["retriever_version"] = st.session_state.get("retriever_version", 0) + 1
        st.session_state["persist_directory"] = ""
        st.session_state["rag_result"] = None
        st.session_state["corag_result"] = None


def _render_sample_question_buttons() -> None:
    st.markdown("### Câu hỏi mẫu")
    cols = st.columns(3)
    for idx, question in enumerate(SAMPLE_QUESTIONS):
        with cols[idx]:
            if st.button(f"Dùng câu {idx + 1}", use_container_width=True):
                st.session_state["question"] = question


def _render_metrics(
    rag_result: Dict[str, Any],
    corag_result: Dict[str, Any],
    rag_time: float,
    corag_time: float,
    question: str,
) -> None:
    st.markdown("### So sánh nhanh")
    metrics_cols = st.columns(3)

    rag_docs = len(rag_result.get("retrieved_docs", [])) if rag_result else 0
    corag_docs = int(corag_result.get("total_docs", 0)) if corag_result else 0

    with metrics_cols[0]:
        st.metric("Số bước (RAG / CoRAG)", f"{rag_result.get('steps', 0)} / {corag_result.get('steps', 0)}")
    with metrics_cols[1]:
        st.metric("Số docs (RAG / CoRAG)", f"{rag_docs} / {corag_docs}")
    with metrics_cols[2]:
        st.metric("Thời gian (RAG / CoRAG)", f"{rag_time:.2f}s / {corag_time:.2f}s")

    rag_quality = simple_answer_score(str(rag_result.get("answer", "")), question)
    corag_quality = simple_answer_score(str(corag_result.get("answer", "")), question)
    quality_cols = st.columns(4)
    with quality_cols[0]:
        st.metric(
            "Coverage token (RAG / CoRAG)",
            f"{rag_quality['covers_question_tokens']:.2f} / {corag_quality['covers_question_tokens']:.2f}",
        )
    with quality_cols[1]:
        st.metric("Độ dài từ (RAG / CoRAG)", f"{rag_quality['length']} / {corag_quality['length']}")
    with quality_cols[2]:
        st.metric(
            "Có số liệu (RAG / CoRAG)",
            f"{'Có' if rag_quality['has_numbers'] else 'Không'} / {'Có' if corag_quality['has_numbers'] else 'Không'}",
        )
    with quality_cols[3]:
        rag_keyword = float(rag_quality.get("keyword_recall", -1.0))
        corag_keyword = float(corag_quality.get("keyword_recall", -1.0))
        if rag_keyword < 0 or corag_keyword < 0:
            keyword_text = "N/A"
        else:
            keyword_text = f"{rag_keyword:.2f} / {corag_keyword:.2f}"
        st.metric("Keyword recall GT (RAG / CoRAG)", keyword_text)

    rag_answer = str(rag_result.get("answer", "")).strip()
    corag_answer = str(corag_result.get("answer", "")).strip()
    rag_note = ""
    corag_note = ""

    if rag_answer:
        if "không có dữ liệu" in rag_answer.lower() or "không xác định" in rag_answer.lower():
            rag_note = "RAG đã trả lời phần có trong context, nhưng vẫn thiếu một số mảnh dữ kiện để kết luận đầy đủ."
        else:
            rag_note = "RAG trả lời một lần từ context đã retrieve, phù hợp khi câu hỏi ngắn và có đủ dữ kiện trực tiếp."

    if corag_answer:
        if "không có dữ liệu" in corag_answer.lower() or "không xác định" in corag_answer.lower():
            corag_note = "CoRAG vẫn còn thiếu vài mảnh dữ kiện, nhưng đã đi thêm bước truy hồi so với RAG."
        else:
            corag_note = "CoRAG đã đi qua nhiều bước truy hồi để ghép đủ dữ kiện trước khi chốt câu trả lời."

    note_cols = st.columns(2)
    with note_cols[0]:
        st.caption(f"RAG nhanh: {rag_note or 'Chưa có kết quả.'}")
    with note_cols[1]:
        st.caption(f"CoRAG nhanh: {corag_note or 'Chưa có kết quả.'}")


@st.cache_resource
def _get_cached_retriever(k: int, provider: str, persist_directory: str, _version: int = 0):
    return get_retriever(k=k, provider=provider, persist_directory=persist_directory)


def _render_corag_step(step_data: Dict[str, Any]) -> None:
    st.session_state.setdefault("corag_live_steps", [])
    st.session_state["corag_live_steps"].append(step_data)


def _build_corag_step_callback(live_status, live_steps):
    def _live_step_callback(step_data: Dict[str, Any]) -> None:
        _render_corag_step(step_data)

        with live_status:
            stage = step_data.get("stage", "step")
            if stage == "retrieve":
                st.caption(f"Bước {step_data['step']}: đang lấy context...")
            elif stage == "evaluate":
                st.caption(f"Bước {step_data['step']}: đang đánh giá context...")
            elif stage == "final":
                st.caption("Đang tổng hợp câu trả lời cuối cùng...")
            else:
                st.caption(f"Đang xử lý bước {step_data['step']}...")

        with live_steps.container():
            for existing_step in st.session_state.get("corag_live_steps", []):
                with st.expander(f"Live step {existing_step['step']} - Query", expanded=True):
                    st.write(existing_step.get("query", ""))
                    st.write(existing_step.get("reasoning", ""))
                    selected_sub_query = existing_step.get("selected_sub_query")
                    if selected_sub_query:
                        st.write(f"Selected sub-query: {selected_sub_query}")
                    rejected_queries = existing_step.get("rejected_queries", [])
                    if rejected_queries:
                        st.write("Rejected queries:")
                        for rejected in rejected_queries:
                            st.write(f"- {rejected.get('query', '')} ({rejected.get('reason', 'unknown')})")
                    missing_parts = existing_step.get("missing_parts", [])
                    if missing_parts:
                        st.write(f"Missing parts: {', '.join(str(p) for p in missing_parts)}")
                    for doc in existing_step.get("retrieved", []):
                        st.write(f"- {doc}")

    return _live_step_callback


def _result_summary(result: Dict[str, Any] | None, is_corag: bool) -> str:
    if not result:
        return "Chưa có kết quả cho lần chạy này."

    answer = str(result.get("answer", "")).strip()
    steps = int(result.get("steps", 0) or 0)
    if is_corag:
        total_docs = int(result.get("total_docs", 0) or 0)
        if answer:
            return f"Đã đi {steps} bước, dùng {total_docs} docs để ghép câu trả lời cuối cùng."
        return f"Đã đi {steps} bước, nhưng chưa tạo được câu trả lời đầy đủ."

    retrieved_docs = len(result.get("retrieved_docs", []))
    if answer:
        return f"Chỉ retrieve 1 lần, dùng {retrieved_docs} docs để trả lời ngay."
    return f"Chỉ retrieve 1 lần, nhưng chưa sinh được câu trả lời hoàn chỉnh."


def _run_rag_once(question: str, retriever, llm) -> tuple[Dict[str, Any], float]:
    start = time.perf_counter()
    result = run_rag(question, retriever, llm)
    return result, time.perf_counter() - start


def _run_corag_once(
    question: str,
    retriever,
    llm,
    max_steps: int,
    retrieval_k: int,
    step_callback=None,
) -> tuple[Dict[str, Any], float]:
    start = time.perf_counter()
    result = run_corag(
        question=question,
        retriever=retriever,
        llm=llm,
        max_steps=max_steps,
        first_step_k=max(3, retrieval_k),
        step_k=max(3, retrieval_k),
        use_llm_part_decomposition=False,
        enable_candidate_generation=False,
        step_callback=step_callback,
    )
    return result, time.perf_counter() - start


def _index_available_docs(uploaded_files, llm_provider: str) -> int:
    """Index docs folder plus current uploads and return indexed chunk count."""
    embedding_provider = _resolve_embedding_provider(llm_provider)
    docs_from_folder = load_documents_from_docs_folder(str(PROJECT_ROOT / "docs"))
    docs_from_upload = load_documents_from_uploads(uploaded_files)
    total_docs = docs_from_folder + docs_from_upload
    if not total_docs:
        return 0

    persist_directory = _get_persist_directory(embedding_provider)
    total_indexed = index_documents(total_docs, provider=embedding_provider, persist_directory=persist_directory)
    st.session_state["indexed_chunks"] = total_indexed
    st.session_state["retriever_version"] = st.session_state.get("retriever_version", 0) + 1
    return total_indexed


def _rebuild_vectorstore(uploaded_files, llm_provider: str) -> int:
    embedding_provider = _resolve_embedding_provider(llm_provider)
    st.session_state["persist_directory"] = str(Path(DEFAULT_PERSIST_DIR) / f"{embedding_provider}-session-{uuid.uuid4().hex}")
    st.session_state["indexed_chunks"] = 0
    st.session_state["retriever_version"] = st.session_state.get("retriever_version", 0) + 1
    return _index_available_docs(uploaded_files, llm_provider)


def _needs_collection_rebuild(result: Dict[str, Any] | None) -> bool:
    if not result:
        return False
    answer = str(result.get("answer", ""))
    return "does not exist" in answer or "Error getting collection" in answer


def _run_with_collection_retry(
    run_fn,
    question: str,
    retriever,
    llm,
    uploaded_files,
    provider,
    *args,
    **kwargs,
):
    result = run_fn(question, retriever, llm, *args, **kwargs)
    if not _needs_collection_rebuild(result):
        return result, retriever, False

    rebuilt_chunks = _rebuild_vectorstore(uploaded_files, provider)
    st.info(f"Đã rebuild ChromaDB từ docs/ ({rebuilt_chunks} chunks) và chạy lại.")
    retriever = _get_cached_retriever(
        k=kwargs.get("step_k", 3) if run_fn is run_corag else kwargs.get("retrieval_k", 3),
        provider=_resolve_embedding_provider(provider),
        _version=st.session_state.get("retriever_version", 0),
    )
    result = run_fn(question, retriever, llm, *args, **kwargs)
    return result, retriever, True


def main() -> None:
    st.set_page_config(page_title="RAG vs CoRAG Demo", layout="wide")
    st.title("So sánh RAG truyền thống và CoRAG")
    st.caption("Demo kỹ thuật: Traditional RAG vs Chain-of-Retrieval Augmented Generation")

    _init_state()
    _sync_docs_state()
    _render_sample_question_buttons()

    if not os.getenv("OPENAI_API_KEY") and st.session_state.get("run_mode") == "Compare both":
        st.session_state["run_mode"] = "CoRAG only"

    with st.sidebar:
        st.header("Thiết lập demo")
        provider = st.selectbox("Model provider", ["auto", "groq", "gemini", "openai"], index=0)
        embedding_provider = _resolve_embedding_provider(provider)
        st.session_state["question"] = st.text_area("Nhập câu hỏi", value=st.session_state["question"], height=120)
        max_steps = st.slider("max_steps cho CoRAG", min_value=1, max_value=6, value=5)
        retrieval_k = st.slider("k documents mỗi lần retrieve", min_value=2, max_value=5, value=3)
        run_mode = st.selectbox(
            "Chế độ chạy",
            ["CoRAG only", "RAG only", "Compare both"],
            index=0,
            key="run_mode",
            help="Compare both sẽ tốn quota gấp đôi khi chưa có OpenAI key.",
        )
        uploaded_files = st.file_uploader("Upload tài liệu (PDF/TXT)", type=["pdf", "txt"], accept_multiple_files=True)

        if st.button("Upload tài liệu", use_container_width=True):
            try:
                total_indexed = _index_available_docs(uploaded_files, provider)
                st.success(f"Index thành công {total_indexed} chunks vào ChromaDB.")
            except Exception as exc:
                st.error(f"Không thể index tài liệu: {exc}")

        run_demo = st.button("Chạy Demo", use_container_width=True)

    col_rag, col_corag = st.columns(2)
    rag_panel = col_rag.container()
    corag_panel = col_corag.container()

    if run_demo:
        question = st.session_state["question"].strip()
        if not question:
            st.warning("Vui lòng nhập câu hỏi trước khi chạy demo.")
        else:
            st.session_state["corag_live_steps"] = []
            st.session_state["rag_result"] = None
            st.session_state["corag_result"] = None
            st.session_state["rag_time"] = 0.0
            st.session_state["corag_time"] = 0.0

            if st.session_state.get("indexed_chunks", 0) == 0:
                try:
                    total_indexed = _index_available_docs(uploaded_files, provider)
                    if total_indexed > 0:
                        st.info(f"Tự động index {total_indexed} chunks từ docs/ trước khi chạy demo.")
                except Exception as exc:
                    st.error(f"Không thể tự động index tài liệu: {exc}")
                    return

            if st.session_state.get("indexed_chunks", 0) == 0:
                st.warning("Chưa có tài liệu nào để index trong docs/ hoặc upload. Hãy thêm tài liệu rồi chạy lại.")
                return

            try:
                llm = _get_cached_llm(provider)
                st.caption(f"LLM đang dùng: {describe_llm(llm)}")
                retriever = _get_cached_retriever(
                    k=retrieval_k,
                    provider=embedding_provider,
                    persist_directory=_get_persist_directory(embedding_provider),
                    _version=st.session_state.get("retriever_version", 0),
                )
            except Exception as exc:
                message = str(exc)
                if "does not exist" in message or "Error getting collection" in message:
                    try:
                        rebuilt_chunks = _rebuild_vectorstore(uploaded_files, provider)
                        st.info(f"Đã rebuild ChromaDB từ docs/ ({rebuilt_chunks} chunks).")
                        retriever = _get_cached_retriever(
                            k=retrieval_k,
                            provider=embedding_provider,
                            persist_directory=_get_persist_directory(embedding_provider),
                            _version=st.session_state.get("retriever_version", 0),
                        )
                    except Exception as rebuild_exc:
                        st.error(f"Không thể rebuild ChromaDB: {rebuild_exc}")
                        return
                else:
                    st.error(f"Không thể khởi tạo LLM hoặc retriever: {exc}")
                    return

            run_rag_enabled = run_mode in {"Compare both", "RAG only"}
            run_corag_enabled = run_mode in {"Compare both", "CoRAG only"}
            live_status = corag_panel.empty()
            live_steps = corag_panel.empty()
            corag_step_callback = _build_corag_step_callback(live_status, live_steps)

            if run_mode == "Compare both":
                with st.status("RAG và CoRAG đang chạy song song...", expanded=True) as status:
                    compare_rag_llm = create_llm(provider)
                    compare_corag_llm = create_llm(provider)
                    st.caption(f"RAG LLM: {describe_llm(compare_rag_llm)}")
                    st.caption(f"CoRAG LLM: {describe_llm(compare_corag_llm)}")

                    with ThreadPoolExecutor(max_workers=2) as executor:
                        rag_future = executor.submit(_run_rag_once, question, retriever, compare_rag_llm)
                        corag_result, corag_time = _run_corag_once(
                            question,
                            retriever,
                            compare_corag_llm,
                            max_steps,
                            retrieval_k,
                            step_callback=corag_step_callback,
                        )
                        rag_result, rag_time = rag_future.result()

                    st.session_state["rag_result"] = rag_result
                    st.session_state["rag_time"] = rag_time
                    st.session_state["corag_result"] = corag_result
                    st.session_state["corag_time"] = corag_time
                    status.update(label="Hoàn thành song song!", state="complete")

                if _needs_collection_rebuild(rag_result) or _needs_collection_rebuild(corag_result):
                    st.warning("Kết quả compare song song chưa khớp collection hiện tại. Hãy chạy lại sau khi index xong.")

            if run_rag_enabled and run_mode != "Compare both":
                with rag_panel:
                    with st.spinner("RAG đang chạy..."):
                        rag_result, rag_time = _run_rag_once(question, retriever, llm)
                        if _needs_collection_rebuild(rag_result):
                            rebuilt_chunks = _rebuild_vectorstore(uploaded_files, provider)
                            st.info(f"Đã rebuild ChromaDB từ docs/ ({rebuilt_chunks} chunks) và chạy lại RAG.")
                            retriever = _get_cached_retriever(
                                k=retrieval_k,
                                provider=embedding_provider,
                                persist_directory=_get_persist_directory(embedding_provider),
                                _version=st.session_state.get("retriever_version", 0),
                            )
                            rag_result, rag_time = _run_rag_once(question, retriever, llm)
                        st.session_state["rag_result"] = rag_result
                        st.session_state["rag_time"] = rag_time

            if run_corag_enabled and run_mode != "Compare both":
                with corag_panel:
                    with st.status("CoRAG đang suy luận...", expanded=True) as status:
                        corag_result, corag_time = _run_corag_once(
                            question,
                            retriever,
                            llm,
                            max_steps,
                            retrieval_k,
                            step_callback=corag_step_callback,
                        )

                        if _needs_collection_rebuild(corag_result):
                            rebuilt_chunks = _rebuild_vectorstore(uploaded_files, provider)
                            st.info(f"Đã rebuild ChromaDB từ docs/ ({rebuilt_chunks} chunks) và chạy lại CoRAG.")
                            retriever = _get_cached_retriever(
                                k=retrieval_k,
                                provider=embedding_provider,
                                persist_directory=_get_persist_directory(embedding_provider),
                                _version=st.session_state.get("retriever_version", 0),
                            )
                            st.session_state["corag_live_steps"] = []
                            corag_result, corag_time = _run_corag_once(question, retriever, llm, max_steps, retrieval_k)

                        status.update(label="Hoàn thành!", state="complete")
                    st.session_state["corag_result"] = corag_result
                    st.session_state["corag_time"] = corag_time

            if not run_rag_enabled:
                st.session_state["rag_result"] = None
                st.session_state["rag_time"] = 0.0
            if not run_corag_enabled:
                st.session_state["corag_result"] = None
                st.session_state["corag_time"] = 0.0

    rag_result = st.session_state.get("rag_result")
    corag_result = st.session_state.get("corag_result")

    with col_rag:
        st.subheader("RAG truyền thống")
        st.caption(_result_summary(rag_result, is_corag=False))
        if rag_result:
            st.write(rag_result.get("answer", ""))
            for i, doc in enumerate(rag_result.get("retrieved_docs", []), start=1):
                with st.expander(f"Retrieved doc {i}"):
                    st.write(doc)
        elif run_mode == "Compare both":
            st.caption("RAG đang chạy hoặc chưa có kết quả cho lần so sánh này.")

    with col_corag:
        st.subheader("CoRAG")
        st.caption(_result_summary(corag_result, is_corag=True))
        if corag_result:
            st.write(corag_result.get("answer", ""))
            for step in corag_result.get("chain", []):
                with st.expander(f"Step {step['step']} - Query"):
                    st.write(step.get("query", ""))
                    st.write(step.get("reasoning", ""))
                    selected_sub_query = step.get("selected_sub_query")
                    if selected_sub_query:
                        st.write(f"Selected sub-query: {selected_sub_query}")
                    rejected_queries = step.get("rejected_queries", [])
                    if rejected_queries:
                        st.write("Rejected queries:")
                        for rejected in rejected_queries:
                            st.write(f"- {rejected.get('query', '')} ({rejected.get('reason', 'unknown')})")
                    missing_parts = step.get("missing_parts", [])
                    if missing_parts:
                        st.write(f"Missing parts: {', '.join(str(p) for p in missing_parts)}")
                    for doc in step.get("retrieved", []):
                        st.write(f"- {doc}")
        elif run_mode == "Compare both":
            st.caption("CoRAG đang chạy hoặc chưa có kết quả cho lần so sánh này.")

    if rag_result and corag_result:
        _render_metrics(
            rag_result=rag_result,
            corag_result=corag_result,
            rag_time=st.session_state.get("rag_time", 0.0),
            corag_time=st.session_state.get("corag_time", 0.0),
            question=st.session_state.get("question", ""),
        )


if __name__ == "__main__":
    main()
