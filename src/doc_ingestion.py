from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import pdfplumber
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

HEADING_PATTERN = re.compile(r"^(\d+(?:\.\d+)*)\s+(.+)$")


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.2))


def _contains_table_like_content(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    dense_separators = sum(1 for line in lines if "|" in line or re.search(r"\s{3,}", line))
    return dense_separators >= max(2, len(lines) // 4)


def _extract_pages_with_pdfplumber(pdf_path: Path) -> List[Tuple[int, str]]:
    pages: List[Tuple[int, str]] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            pages.append((idx, page.extract_text() or ""))
    return pages


def _extract_pages_with_pypdf(pdf_path: Path) -> List[Tuple[int, str]]:
    pages: List[Tuple[int, str]] = []
    reader = PdfReader(str(pdf_path))
    for idx, page in enumerate(reader.pages, start=1):
        pages.append((idx, page.extract_text() or ""))
    return pages


def extract_pdf_pages(pdf_path: Path) -> List[Tuple[int, str]]:
    """Extract page text with pdfplumber and fallback to pypdf."""
    try:
        pages = _extract_pages_with_pdfplumber(pdf_path)
        if any(text.strip() for _, text in pages):
            return pages
    except Exception:
        pass
    return _extract_pages_with_pypdf(pdf_path)


def _find_heading(line: str) -> Tuple[Optional[str], Optional[str]]:
    cleaned = line.strip()
    match = HEADING_PATTERN.match(cleaned)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def chunk_pages(
    pages: List[Tuple[int, str]],
    chunk_size_chars: int = 1300,
    overlap_chars: int = 180,
) -> List[Dict]:
    """Create structure-aware chunks with fallback to paragraph chunking."""
    chunks: List[Dict] = []
    chunk_index = 0
    current_section: Optional[str] = None
    current_section_title: Optional[str] = None

    for page_num, raw_text in pages:
        text = (raw_text or "").replace("\x00", " ").strip()
        if not text:
            continue

        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if not paragraphs:
            paragraphs = [line.strip() for line in text.splitlines() if line.strip()]

        for para in paragraphs:
            lines = [line.strip() for line in para.splitlines() if line.strip()]
            for line in lines[:3]:
                section, section_title = _find_heading(line)
                if section:
                    current_section = section
                    current_section_title = section_title
                    break

            start = 0
            while start < len(para):
                end = min(len(para), start + chunk_size_chars)
                chunk_text = para[start:end].strip()
                if not chunk_text:
                    break

                metadata = {
                    "page_start": page_num,
                    "section": current_section,
                    "section_title": current_section_title,
                    "chunk_index": chunk_index,
                    "token_count": _estimate_tokens(chunk_text),
                    "chunk_type": "paragraph",
                    "contains_table": _contains_table_like_content(chunk_text),
                }
                chunks.append({"text": chunk_text, "metadata": metadata})
                chunk_index += 1

                if end == len(para):
                    break
                next_start = max(0, end - overlap_chars)
                start = next_start if next_start > start else end

    return chunks


def _compute_source_id(pdf_bytes: bytes) -> str:
    return hashlib.sha256(pdf_bytes).hexdigest()[:16]


def _cache_files(cache_dir: Path, source_id: str) -> Dict[str, Path]:
    return {
        "chunks": cache_dir / f"chunks_{source_id}.json",
        "metadata": cache_dir / f"metadata_{source_id}.json",
        "faiss": cache_dir / f"faiss_{source_id}.index",
        "embeddings": cache_dir / f"embeddings_{source_id}.npy",
        "uploaded": cache_dir / "uploads" / f"{source_id}.pdf",
    }


def _persist_uploaded_pdf(uploaded_bytes: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(uploaded_bytes)


def _load_cached_chunks(cache_paths: Dict[str, Path]) -> Optional[List[Dict]]:
    chunks_path = cache_paths["chunks"]
    metadata_path = cache_paths["metadata"]
    if not chunks_path.exists() or not metadata_path.exists():
        return None

    with chunks_path.open("r", encoding="utf-8") as f:
        texts = json.load(f)
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata_list = json.load(f)

    return [{"text": t, "metadata": m} for t, m in zip(texts, metadata_list)]


def _save_chunks_to_cache(chunks: List[Dict], cache_paths: Dict[str, Path]) -> None:
    texts = [c["text"] for c in chunks]
    metadata = [c["metadata"] for c in chunks]

    with cache_paths["chunks"].open("w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False, indent=2)
    with cache_paths["metadata"].open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def ingest_pdf(
    pdf_path: Optional[str],
    uploaded_pdf_bytes: Optional[bytes] = None,
    cache_dir: str = ".cache",
) -> Tuple[List[Dict], str, Path]:
    """
    Ingest a PDF into text chunks and metadata with caching.

    Returns:
        chunks, source_id, resolved_pdf_path
    """
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    resolved_path: Path
    if uploaded_pdf_bytes is not None:
        source_id = _compute_source_id(uploaded_pdf_bytes)
        cache_paths = _cache_files(cache_root, source_id)
        _persist_uploaded_pdf(uploaded_pdf_bytes, cache_paths["uploaded"])
        resolved_path = cache_paths["uploaded"]
    else:
        if not pdf_path:
            raise ValueError("A PDF path is required when no uploaded bytes are provided.")
        resolved_path = Path(pdf_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"PDF not found at {resolved_path}")
        source_id = _compute_source_id(resolved_path.read_bytes())
        cache_paths = _cache_files(cache_root, source_id)

    cached = _load_cached_chunks(cache_paths)
    if cached is not None:
        return cached, source_id, resolved_path

    pages = extract_pdf_pages(resolved_path)
    chunks = chunk_pages(pages)
    if not chunks:
        raise ValueError("No readable text could be extracted from this PDF.")

    _save_chunks_to_cache(chunks, cache_paths)
    return chunks, source_id, resolved_path


def build_or_load_faiss(
    chunks: List[Dict],
    source_id: str,
    embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
    cache_dir: str = ".cache",
) -> Tuple[faiss.IndexFlatIP, np.ndarray, SentenceTransformer]:
    """Build or load FAISS index and embedding matrix for semantic retrieval."""
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_paths = _cache_files(cache_root, source_id)

    model = SentenceTransformer(embedding_model_name)

    if cache_paths["faiss"].exists() and cache_paths["embeddings"].exists():
        index = faiss.read_index(str(cache_paths["faiss"]))
        embeddings = np.load(str(cache_paths["embeddings"]))
        return index, embeddings, model

    texts = [item["text"] for item in chunks]
    embeddings = model.encode(
        texts,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(cache_paths["faiss"]))
    np.save(str(cache_paths["embeddings"]), embeddings)
    return index, embeddings, model
