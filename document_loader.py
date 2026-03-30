from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Iterable, List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
PDF_CHUNK_SIZE = 800
PDF_CHUNK_OVERLAP = 150
SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def _clean_pdf_text(text: str) -> str:
    """Clean common PDF extraction artifacts before chunking."""
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(
        r"(?<![.!?:,])\n(?=[a-záàảãạăắặằẳẵâấậầẩẫđéèẻẽẹêếệềểễíìỉĩịóòỏõọôốộồổỗơớợờởỡúùủũụưứựừửữýỳỷỹỵA-Z])",
        " ",
        text,
    )
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_documents(documents: List[Document], is_pdf: bool = False) -> List[Document]:
    """Split documents with a fixed chunk size for stable demo behavior."""
    chunk_size = PDF_CHUNK_SIZE if is_pdf else CHUNK_SIZE
    chunk_overlap = PDF_CHUNK_OVERLAP if is_pdf else CHUNK_OVERLAP
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
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


def _load_text_from_txt(file_path: Path) -> List[Document]:
    for encoding in ("utf-8", "utf-8-sig", "cp1258", "latin-1"):
        try:
            loader = TextLoader(str(file_path), encoding=encoding)
            return loader.load()
        except Exception:
            continue
    raise RuntimeError(f"Không thể đọc file TXT: {file_path.name}")


def _load_single_file(file_path: Path, source_name: str | None = None) -> List[Document]:
    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        return []

    if suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))
        docs = loader.load()
        page_lengths = [len(doc.page_content.strip()) for doc in docs]
        total_chars = sum(page_lengths)
        low_text_pages = sum(1 for length in page_lengths if length < 50)
        if docs and total_chars < 50 * len(docs):
            display_name = source_name or file_path.name
            raise RuntimeError(
                f"'{display_name}' trả về quá ít text ({total_chars} ký tự / "
                f"{len(docs)} trang, {low_text_pages} trang dưới 50 ký tự). "
                "Đây có thể là PDF scan; cần OCR trước khi index."
            )
        for doc in docs:
            doc.page_content = _clean_pdf_text(doc.page_content)
        return docs

    return _load_text_from_txt(file_path)


def load_documents_from_docs_folder(docs_dir: str = "docs") -> List[Document]:
    """Load and chunk PDF/TXT files from docs directory."""
    try:
        docs_path = Path(docs_dir)
        if not docs_path.exists() or not docs_path.is_dir():
            return []

        txt_documents: List[Document] = []
        pdf_documents: List[Document] = []
        for file_path in sorted(docs_path.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                try:
                    source_name = file_path.relative_to(docs_path).as_posix()
                except Exception:
                    source_name = file_path.name
                docs = _load_single_file(file_path, source_name=source_name)
                if file_path.suffix.lower() == ".pdf":
                    pdf_documents.extend(docs)
                else:
                    txt_documents.extend(docs)

        chunks: List[Document] = []
        if txt_documents:
            chunks.extend(_split_documents(txt_documents, is_pdf=False))
        if pdf_documents:
            chunks.extend(_split_documents(pdf_documents, is_pdf=True))
        return chunks
    except Exception as exc:
        raise RuntimeError(f"Failed to load documents from '{docs_dir}': {exc}") from exc


def load_documents_from_uploads(uploaded_files: Iterable) -> List[Document]:
    """Load and chunk uploaded PDF/TXT files from Streamlit uploader."""
    txt_documents: List[Document] = []
    pdf_documents: List[Document] = []

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
                if suffix == ".pdf":
                    pdf_documents.extend(docs)
                else:
                    txt_documents.extend(docs)
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()

        chunks: List[Document] = []
        if txt_documents:
            chunks.extend(_split_documents(txt_documents, is_pdf=False))
        if pdf_documents:
            chunks.extend(_split_documents(pdf_documents, is_pdf=True))
        return chunks
    except Exception as exc:
        raise RuntimeError(f"Failed to load uploaded files: {exc}") from exc
