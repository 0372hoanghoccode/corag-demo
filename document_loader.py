from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Iterable, List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def _split_documents(documents: List[Document]) -> List[Document]:
    """Split documents with a fixed chunk size for stable demo behavior."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page")
        # Keep source mandatory for UI traceability; page is optional when available.
        chunk.metadata["source"] = source
        if page is not None:
            chunk.metadata["page"] = page
        chunk.metadata["chunk_id"] = f"{source}::{page if page is not None else 'na'}::{index}"

    return chunks


def _load_single_file(file_path: Path, source_name: str | None = None) -> List[Document]:
    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        return []

    if suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))
        docs = loader.load()
        total_chars = sum(len(doc.page_content.strip()) for doc in docs)
        if docs and total_chars < 50 * len(docs):
            display_name = source_name or file_path.name
            raise RuntimeError(
                f"'{display_name}' trả về quá ít text ({total_chars} ký tự / "
                f"{len(docs)} trang). Có thể là PDF scan - cần OCR."
            )
        return docs

    loader = TextLoader(str(file_path), encoding="utf-8")
    return loader.load()


def load_documents_from_docs_folder(docs_dir: str = "docs") -> List[Document]:
    """Load and chunk PDF/TXT files from docs directory."""
    try:
        docs_path = Path(docs_dir)
        if not docs_path.exists() or not docs_path.is_dir():
            return []

        raw_documents: List[Document] = []
        for file_path in sorted(docs_path.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                raw_documents.extend(_load_single_file(file_path, source_name=file_path.name))

        return _split_documents(raw_documents)
    except Exception as exc:
        raise RuntimeError(f"Failed to load documents from '{docs_dir}': {exc}") from exc


def load_documents_from_uploads(uploaded_files: Iterable) -> List[Document]:
    """Load and chunk uploaded PDF/TXT files from Streamlit uploader."""
    raw_documents: List[Document] = []

    try:
        for uploaded_file in uploaded_files or []:
            suffix = Path(uploaded_file.name).suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                continue

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_path = Path(tmp_file.name)
                tmp_file.write(uploaded_file.getvalue())

            try:
                docs = _load_single_file(tmp_path, source_name=uploaded_file.name)
                for doc in docs:
                    doc.metadata["source"] = uploaded_file.name
                raw_documents.extend(docs)
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()

        return _split_documents(raw_documents)
    except Exception as exc:
        raise RuntimeError(f"Failed to load uploaded files: {exc}") from exc
