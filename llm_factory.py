from __future__ import annotations

import os
import time
import hashlib
import re
from typing import Any, Dict, List


def _clean_groq_text(text: str) -> str:
    cleaned = text.replace("tráách", "trách").replace("modell", "model")
    cleaned = re.sub(r"([A-Za-zÀ-ỹ])\1{2,}", r"\1", cleaned)
    return cleaned


class _GroqResponseWrapper:
    def __init__(self, inner_llm: Any) -> None:
        self._inner_llm = inner_llm

    def invoke(self, prompt: str):
        result = self._inner_llm.invoke(prompt)
        if isinstance(result, str):
            return _clean_groq_text(result)

        content = getattr(result, "content", None)
        if isinstance(content, str):
            cleaned = _clean_groq_text(content)
            if cleaned != content:
                try:
                    result.content = cleaned
                    return result
                except Exception:
                    return cleaned
        return result

    @property
    def model_name(self) -> str:
        return getattr(self._inner_llm, "model_name", None) or getattr(self._inner_llm, "model", "groq")


class GeminiFailoverLLM:
    def __init__(
        self,
        model_names: List[str],
        temperature: float = 0.0,
        timeout_seconds: int = 12,
        min_interval: float = 4.5,
        max_cache_entries: int = 256,
    ) -> None:
        from langchain_google_genai import ChatGoogleGenerativeAI

        self.model_names = model_names
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self._clients = {
            name: ChatGoogleGenerativeAI(
                model=name,
                temperature=self.temperature,
                timeout=self.timeout_seconds,
                max_retries=0,
            )
            for name in self.model_names
        }
        self._cooldown_until: Dict[str, float] = {}
        self._last_call_time = 0.0
        self.min_interval = max(0.0, float(min_interval))
        self.max_cache_entries = max(1, int(max_cache_entries))
        self._response_cache: Dict[str, Any] = {}

    def _is_in_cooldown(self, model_name: str) -> bool:
        return time.time() < self._cooldown_until.get(model_name, 0.0)

    def _mark_cooldown(self, model_name: str, seconds: int = 120) -> None:
        self._cooldown_until[model_name] = time.time() + seconds

    def _wait_if_needed(self) -> None:
        elapsed = time.time() - self._last_call_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

    def _wait_for_available_model(self) -> None:
        now = time.time()
        available_models = [
            model_name
            for model_name in self.model_names
            if now >= self._cooldown_until.get(model_name, 0.0)
        ]
        if available_models:
            return

        earliest_available = min(self._cooldown_until.get(model_name, 0.0) for model_name in self.model_names)
        wait_time = max(0.0, earliest_available - now)
        if wait_time > 0:
            print(f"[LLM] Tất cả model đang cooldown, chờ {wait_time:.1f}s...")
            time.sleep(wait_time + 1.0)

    def _invoke_with_failover(self, prompt: str):
        last_error = None
        for model_name in self.model_names:
            if self._is_in_cooldown(model_name):
                continue
            try:
                return self._clients[model_name].invoke(prompt)
            except Exception as exc:
                text = str(exc)
                if "RESOURCE_EXHAUSTED" in text or "NOT_FOUND" in text or "429" in text or "404" in text:
                    self._mark_cooldown(model_name)
                    last_error = exc
                    continue
                if "timed out" in text.lower() or "deadline" in text.lower() or "timeout" in text.lower():
                    self._mark_cooldown(model_name, seconds=30)
                    last_error = exc
                    continue
                raise

        raise RuntimeError(f"All Gemini models failed: {last_error}")

    def invoke(self, prompt: str):
        cache_key = hashlib.md5(prompt.encode("utf-8")).hexdigest()
        if cache_key in self._response_cache:
            return self._response_cache[cache_key]

        self._wait_if_needed()
        self._wait_for_available_model()
        result = self._invoke_with_failover(prompt)
        self._last_call_time = time.time()

        if len(self._response_cache) >= self.max_cache_entries:
            oldest_key = next(iter(self._response_cache))
            self._response_cache.pop(oldest_key, None)
        self._response_cache[cache_key] = result
        return result


def create_llm(provider: str = "auto") -> Any:
    selected = provider.lower()

    if selected == "groq" or (selected == "auto" and os.getenv("GROQ_API_KEY")):
        try:
            from langchain_groq import ChatGroq

            return _GroqResponseWrapper(ChatGroq(model="qwen/qwen3-32b", temperature=0))
        except Exception as exc:
            raise RuntimeError(f"Cannot initialize Groq model: {exc}") from exc

    if selected == "gemini" or (selected == "auto" and os.getenv("GOOGLE_API_KEY")):
        return GeminiFailoverLLM(
            model_names=["gemini-2.5-flash", "gemini-2.0-flash"],
            temperature=0,
            timeout_seconds=12,
            min_interval=7.5,
        )

    if selected == "openai" or (selected == "auto" and os.getenv("OPENAI_API_KEY")):
        try:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(model="gpt-4o-mini", temperature=0)
        except Exception as exc:
            raise RuntimeError(f"Cannot initialize OpenAI model: {exc}") from exc

    raise RuntimeError("Missing API key. Set OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in environment.")


def describe_llm(llm: Any) -> str:
    if isinstance(llm, GeminiFailoverLLM):
        return f"GeminiFailover({', '.join(llm.model_names)})"

    inner_llm = getattr(llm, "_inner_llm", None)
    if inner_llm is not None:
        model_name = getattr(inner_llm, "model_name", None) or getattr(inner_llm, "model", None)
        if model_name:
            return f"Groq({model_name})"
        return "Groq"

    model_name = getattr(llm, "model_name", None) or getattr(llm, "model", None)
    if model_name:
        return f"{llm.__class__.__name__}({model_name})"

    return llm.__class__.__name__
