from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional

from langchain_core.documents import Document

try:
    from langchain_chroma import Chroma
except Exception:
    from langchain_community.vectorstores import Chroma  # type: ignore

from langchain_huggingface import HuggingFaceEmbeddings

DEFAULT_COLLECTION = "corag_demo"
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PERSIST_DIR = str(Path.home() / ".cache" / "corag-demo" / "chroma_db")
DEFAULT_HF_EMBEDDING_MODEL = os.getenv(
    "HF_EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
)


@lru_cache(maxsize=4)
def get_embeddings(provider: str = "huggingface") -> Optional[object]:
    """Return embedding model with graceful fallback for constrained environments."""
    selected = (provider or "huggingface").lower()

    if selected == "openai" and os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(model="text-embedding-3-small")
        except Exception as exc:
            raise RuntimeError(f"OpenAI embeddings unavailable: {exc}") from exc

    if selected in {"huggingface", "auto"}:
        try:
            return HuggingFaceEmbeddings(model_name=DEFAULT_HF_EMBEDDING_MODEL)
        except Exception:
            # Fallback path: if local HF dependencies are missing, prefer OpenAI when configured.
            if os.getenv("OPENAI_API_KEY"):
                try:
                    from langchain_openai import OpenAIEmbeddings

                    return OpenAIEmbeddings(model="text-embedding-3-small")
                except Exception:
                    pass
            # Final fallback: let Chroma use its built-in default embedding function.
            return None

    # Non-recognized provider falls back to Chroma default embeddings.
    return None


def _doc_id(doc: Document) -> str:
    source = str(doc.metadata.get("source", "unknown"))
    page = str(doc.metadata.get("page", "na"))
    payload = f"{source}|{page}|{doc.page_content.strip()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def index_documents(
    documents: Iterable[Document],
    provider: str = "huggingface",
    persist_directory: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION,
) -> int:
    """Index documents in local Chroma and return number of chunks processed."""
    docs = list(documents or [])
    if not docs:
        return 0

    try:
        persist_path = Path(persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)

        embeddings = get_embeddings(provider)
        db = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(persist_path),
        )
        ids = [_doc_id(doc) for doc in docs]
        db.add_documents(docs, ids=ids)
        if hasattr(db, "persist"):
            db.persist()
        return len(docs)
    except Exception as exc:
        raise RuntimeError(f"Failed to index documents into ChromaDB: {exc}") from exc


def get_retriever(
    k: int = 3,
    provider: str = "huggingface",
    persist_directory: str = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION,
):
    """Create retriever from persisted Chroma collection with configurable k."""
    try:
        embeddings = get_embeddings(provider)
        db = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(Path(persist_directory)),
        )
        return db.as_retriever(search_kwargs={"k": max(1, int(k))})
    except Exception as exc:
        raise RuntimeError(f"Failed to create retriever: {exc}") from exc
