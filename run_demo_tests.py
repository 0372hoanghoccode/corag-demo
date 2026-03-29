from __future__ import annotations

import hashlib
import os
import shutil
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from corag_engine import run_corag
from document_loader import load_documents_from_docs_folder
from eval import keyword_recall
from rag_engine import run_rag
from vectorstore import DEFAULT_PERSIST_DIR, get_retriever, index_documents
from llm_factory import create_llm, describe_llm


def _build_llm():
    return create_llm('auto')


def main() -> None:
    load_dotenv('.env')
    key = os.getenv('GOOGLE_API_KEY')
    print('GOOGLE_KEY_SET=', bool(key))
    if not key:
        raise RuntimeError('Missing GOOGLE_API_KEY in .env')

    persist_path = Path(DEFAULT_PERSIST_DIR)
    if persist_path.exists():
        shutil.rmtree(persist_path)

    docs = load_documents_from_docs_folder('docs')
    print('Chunks loaded:', len(docs))

    indexed = index_documents(docs, provider='auto', persist_directory=DEFAULT_PERSIST_DIR)
    print('Chunks indexed:', indexed)

    retriever = get_retriever(k=3, provider='auto', persist_directory=DEFAULT_PERSIST_DIR)
    llm = _build_llm()
    print('LLM in use:', describe_llm(llm))

    questions = [
        ('3-hop', 'Cluster nào ghi nhận nhiều incident nhất trong Q2 2025, người phụ trách cluster đó đang triển khai dự án gì, và KPI của dự án đó là gì?'),
        ('4-hop', 'Model nào có latency P95 cao nhất theo benchmark mới nhất, đội nào sử dụng model đó nhiều nhất, cluster nào đang chạy model đó, và ai là người phụ trách cluster đó?'),
        ('5-hop', 'Đội nào chịu ảnh hưởng nhiều nhất từ incident Q2, model chính của đội đó là gì, latency P95 của model đó theo báo cáo MỚI NHẤT là bao nhiêu, ai phụ trách model đó, và deadline dự án của người đó là tháng mấy?'),
    ]

    ground_truth = {
        '3-hop': 'Cluster Gamma Trần Văn Đức Gamma Upgrade 2025 giảm 50% incident tháng 9',
        '4-hop': 'LogParser-4B 890ms DevOps Cluster Gamma Trần Văn Đức',
        '5-hop': 'DevOps LogParser-4B 890ms Trần Văn Đức tháng 9 2025',
    }

    for name, question in questions:
        print(f"\n=== {name} ===")

        t0 = time.time()
        rag = run_rag(question, retriever, llm)
        t1 = time.time()

        corag = run_corag(
            question,
            retriever,
            llm,
            max_steps=5,
            first_step_k=1,
            step_k=2,
            use_llm_part_decomposition=False,
            enable_candidate_generation=False,
        )
        t2 = time.time()

        print('RAG steps:', rag.get('steps'), 'time:', round(t1 - t0, 2), 'docs:', len(rag.get('retrieved_docs', [])))
        print('RAG answer:', (rag.get('answer') or '').replace('\n', ' ')[:260])

        print('CoRAG steps:', corag.get('steps'), 'time:', round(t2 - t1, 2), 'total_docs:', corag.get('total_docs'))
        print('CoRAG answer:', (corag.get('answer') or '').replace('\n', ' ')[:260])

        rag_recall = keyword_recall(str(rag.get('answer', '')), ground_truth[name])
        corag_recall = keyword_recall(str(corag.get('answer', '')), ground_truth[name])
        print(f'Keyword Recall - RAG: {rag_recall:.2f} | CoRAG: {corag_recall:.2f}')

        for step in corag.get('chain', []):
            query = (step.get('query') or '').replace('\n', ' ')[:120]
            reasoning = (step.get('reasoning') or '').replace('\n', ' ')[:160]
            print(
                f"  - step {step.get('step')}: sufficient={step.get('sufficient')} "+
                f"retrieved={len(step.get('retrieved', []))}"
            )
            print('    query:', query)
            print('    reasoning:', reasoning)
            selected = step.get('selected_sub_query')
            if selected:
                print('    selected_sub_query:', str(selected)[:140])
            rejected = step.get('rejected_queries', [])
            if rejected:
                for item in rejected:
                    rq = str(item.get('query', ''))[:100]
                    rs = str(item.get('reason', 'unknown'))
                    print(f'    rejected_query: {rq} ({rs})')


if __name__ == '__main__':
    main()
