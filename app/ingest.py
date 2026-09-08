# app/ingest.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from clients import embed_texts, get_supabase

# Supabase rejects very large request bodies; 384-dim vectors plus chunk text
# put the practical ceiling well above this, so 50 is a safe batch size.
INSERT_BATCH_SIZE = 50


@dataclass
class Chunk:
    idx: int
    text: str
    pages: List[int]


def extract_text_from_pdf(pdf_path: str) -> List[Tuple[int, str]]:
    """Return a list of (page_number_1_indexed, page_text) for non-empty pages."""
    import pdfplumber

    pages: List[Tuple[int, str]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append((i, text))
    return pages


def chunk_pages(
    pages: List[Tuple[int, str]],
    chunk_size: int = 900,
    overlap: int = 120,
) -> List[Chunk]:
    """
    Chunk each page independently so every chunk carries correct page metadata.

    `overlap` must be smaller than `chunk_size`, otherwise the sliding window
    would never advance.
    """
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})."
        )

    chunks: List[Chunk] = []
    idx = 0

    for page_num, page_text in pages:
        text = (page_text or "").strip()
        if not text:
            continue

        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(Chunk(idx=idx, text=chunk_text, pages=[page_num]))
                idx += 1

            if end >= len(text):
                break
            start = end - overlap

    return chunks


def ingest_pdf_to_supabase(pdf_path: str, filename: Optional[str] = None) -> str:
    """
    Extract, chunk, embed and store a PDF. Returns the new document's id.

    Raises ValueError for unusable input (scanned PDFs) and RuntimeError when
    Supabase rejects a write, so the caller can show the real cause.
    """
    sb = get_supabase()

    pages = extract_text_from_pdf(pdf_path)
    if not pages:
        raise ValueError(
            "No extractable text found in this PDF. Scanned or image-only PDFs "
            "need OCR before they can be indexed."
        )

    chunks = chunk_pages(pages)
    if not chunks:
        raise ValueError("PDF text was extracted, but chunking produced no chunks.")

    embeddings = embed_texts([c.text for c in chunks])

    doc_name = filename or os.path.basename(pdf_path)
    doc_insert = sb.table("documents").insert({"filename": doc_name}).execute()
    if not doc_insert.data:
        raise RuntimeError(f"Failed to insert document row into Supabase: {doc_insert}")

    document_id = doc_insert.data[0]["id"]

    # Column names here must match sql/001_tables.sql exactly: the chunks table
    # has `doc_id` and `pages`, not `document_id` and a `metadata` blob.
    rows: List[Dict[str, Any]] = [
        {
            "doc_id": document_id,
            "chunk_index": c.idx,
            "content": c.text,
            "pages": c.pages,
            "embedding": emb,
        }
        for c, emb in zip(chunks, embeddings)
    ]

    for i in range(0, len(rows), INSERT_BATCH_SIZE):
        batch = rows[i : i + INSERT_BATCH_SIZE]
        res = sb.table("chunks").insert(batch).execute()
        if not res.data:
            # Roll back the orphaned document row so a failed ingest does not
            # leave a document with no retrievable chunks behind.
            try:
                sb.table("documents").delete().eq("id", document_id).execute()
            except Exception:  # noqa: BLE001 - cleanup must not mask the real error
                pass
            raise RuntimeError(
                f"Chunk insert failed at batch {i // INSERT_BATCH_SIZE}: {res}"
            )

    return document_id
