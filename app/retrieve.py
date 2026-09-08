# app/retrieve.py
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from clients import embed_text, get_supabase

logger = logging.getLogger(__name__)


class RetrievalError(RuntimeError):
    """Raised when the similarity search cannot be completed."""


def retrieve_context(
    query: str,
    top_k: int = 6,
    document_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve the top-k most similar chunks for a query via the match_chunks RPC.

    Argument names must match the function signature in sql/002_match_chunks.sql,
    since PostgREST resolves RPCs by named argument. Pass document_id=None to
    search across every indexed document.

    Raises RetrievalError on failure rather than returning an empty list, so a
    misconfigured database cannot masquerade as "no results found".
    """
    if not query or not query.strip():
        return []

    sb = get_supabase()

    try:
        query_embedding = embed_text(query)
        response = sb.rpc(
            "match_chunks",
            {
                "query_embedding": query_embedding,
                "match_count": int(top_k),
                "filter_doc_id": document_id,
            },
        ).execute()
    except Exception as exc:  # noqa: BLE001 - re-raised as a typed error below
        logger.exception("match_chunks RPC failed")
        raise RetrievalError(
            f"Similarity search failed: {exc}\n\n"
            "Confirm sql/001_tables.sql and sql/002_match_chunks.sql have both "
            "been applied to this Supabase project."
        ) from exc

    data = response.data or []
    if not isinstance(data, list):
        raise RetrievalError(
            f"match_chunks returned {type(data).__name__}, expected a list of rows."
        )

    return data
