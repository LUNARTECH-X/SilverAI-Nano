# app/settings.py
from __future__ import annotations

import os

from dotenv import load_dotenv

# Load .env automatically (safe if the file is not present).
load_dotenv()


def _get_env(name: str, default: str | None = None) -> str:
    """Fetch an environment variable, stripped. Returns default when missing."""
    value = os.getenv(name)
    if value is None:
        return default if default is not None else ""
    return value.strip()


def _get_int_env(name: str, default: int) -> int:
    """Fetch an int environment variable, falling back on unparseable values."""
    raw = _get_env(name, str(default))
    try:
        return int(raw)
    except ValueError:
        return default


def require_env(name: str, value: str) -> None:
    """Raise a clear, actionable error if a required variable is missing."""
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}\n\n"
            f"Copy .env.example to .env and fill it in, or export {name} "
            f"before starting the app."
        )


# ---- Supabase ----
SUPABASE_URL: str = _get_env("SUPABASE_URL")
SUPABASE_SERVICE_KEY: str = _get_env("SUPABASE_SERVICE_KEY")

# ---- Grok / xAI ----
# Either name is accepted; XAI_API_KEY wins when both are set.
XAI_API_KEY: str = _get_env("XAI_API_KEY")
GROK_API_KEY: str = _get_env("GROK_API_KEY")
GROK_MODEL: str = _get_env("GROK_MODEL", "grok-4.20-0309-non-reasoning")
GROK_MAX_TOKENS: int = _get_int_env("GROK_MAX_TOKENS", 4000)
GROK_MAX_RETRIES: int = _get_int_env("GROK_MAX_RETRIES", 4)

# ---- Embeddings ----
# Must stay in sync with the vector(N) dimension in sql/001_tables.sql.
EMBED_MODEL: str = _get_env("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIM: int = _get_int_env("EMBED_DIM", 384)

# ---- Generation ----
TARGET_HANDBOOK_WORDS: int = _get_int_env("TARGET_HANDBOOK_WORDS", 20000)


def api_key() -> str:
    """The active xAI credential, or an empty string when unconfigured."""
    return XAI_API_KEY or GROK_API_KEY
