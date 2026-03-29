from __future__ import annotations

import re
from typing import Dict

GROUND_TRUTH = {
    "Cluster nào ghi nhận nhiều incident nhất trong Q2 2025, người phụ trách cluster đó đang triển khai dự án gì, và KPI của dự án đó là gì?": "Cluster Gamma Trần Văn Đức Gamma Upgrade 2025 giảm 50% incident tháng 9",
    "Model nào có latency P95 cao nhất theo benchmark mới nhất, đội nào sử dụng model đó nhiều nhất, cluster nào đang chạy model đó, và ai là người phụ trách cluster đó?": "LogParser-4B 890ms DevOps Cluster Gamma Trần Văn Đức",
    "Đội nào chịu ảnh hưởng nhiều nhất từ incident Q2, model chính của đội đó là gì, latency P95 của model đó theo báo cáo MỚI NHẤT là bao nhiêu, ai phụ trách model đó, và deadline dự án của người đó là tháng mấy?": "DevOps LogParser-4B 890ms Trần Văn Đức tháng 9 2025",
}


def keyword_recall(answer: str, reference: str) -> float:
    reference_text = (reference or "").strip()
    if not reference_text:
        return -1.0

    if reference_text in GROUND_TRUTH:
        reference_text = GROUND_TRUTH[reference_text]

    gt_tokens = set(re.findall(r"\w+", reference_text.lower()))
    ans_tokens = set(re.findall(r"\w+", (answer or "").lower()))
    return len(gt_tokens.intersection(ans_tokens)) / max(1, len(gt_tokens))


def simple_answer_score(answer: str, question: str) -> Dict[str, float | int | bool]:
    """Heuristic quality score without extra LLM calls."""
    answer_text = (answer or "").strip()
    question_text = (question or "").strip()

    question_tokens = set(re.findall(r"\w+", question_text.lower()))
    answer_tokens = set(re.findall(r"\w+", answer_text.lower()))
    overlap = len(question_tokens.intersection(answer_tokens))
    coverage = overlap / max(1, len(question_tokens))

    return {
        "length": len(answer_text.split()),
        "has_numbers": bool(re.search(r"\d+", answer_text)),
        "covers_question_tokens": coverage,
        "keyword_recall": keyword_recall(answer_text, question_text),
    }
