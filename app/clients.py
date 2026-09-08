# app/clients.py
"""
Shared, cached handles to external services.

Both ingestion and retrieval need a Supabase client and the embedding model.
Defining them once here means the sentence-transformers model is held in memory
a single time rather than once per importing module.

Heavy third-party imports are deferred into the functions that need them, and
the cache decorator falls back to `functools.lru_cache` when Streamlit is not
running. Together these let the pipeline be imported from a plain script or a
test run without pulling in Streamlit or torch.
"""
from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable, List, TypeVar

from settings import (
    EMBED_DIM,
    EMBED_MODEL,
    SUPABASE_SERVICE_KEY,
    SUPABASE_URL,
    require_env,
)

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from sentence_transformers import SentenceTransformer
    from supabase import Client

F = TypeVar("F", bound=Callable[..., Any])


def _cache_resource(func: F) -> F:
    """Cache a singleton with Streamlit when available, else with lru_cache."""
    try:
        import streamlit as st

        return st.cache_resource(show_spinner=False)(func)
    except ImportError:
        return lru_cache(maxsize=1)(func)


@_cache_resource
def get_supabase() -> "Client":
    """Create and cache a Supabase client (the service key stays server-side)."""
    from supabase import create_client

    require_env("SUPABASE_URL", SUPABASE_URL)
    require_env("SUPABASE_SERVICE_KEY", SUPABASE_SERVICE_KEY)
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


@_cache_resource
def get_embedder() -> "SentenceTransformer":
    """Load and cache the embedding model, verifying its output dimension."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL)
    actual = model.get_sentence_embedding_dimension()
    if actual != EMBED_DIM:
        raise RuntimeError(
            f"Embedding dimension mismatch: {EMBED_MODEL} produces {actual}-dim "
            f"vectors but EMBED_DIM is {EMBED_DIM}. Update EMBED_DIM and the "
            f"vector({EMBED_DIM}) columns in sql/001_tables.sql to match, then "
            f"re-ingest your documents."
        )
    return model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a batch of strings into normalized vectors."""
    if not texts:
        return []
    vectors = get_embedder().encode(texts, normalize_embeddings=True)
    return vectors.tolist()


def embed_text(text: str) -> List[float]:
    """Embed a single string into a normalized vector."""
    return embed_texts([text])[0]
