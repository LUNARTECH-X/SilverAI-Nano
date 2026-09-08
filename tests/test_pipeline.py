"""
Offline tests for the pure parts of the pipeline.

These run without Supabase, without an API key and without downloading the
embedding model, so they are safe in CI and as a pre-flight check.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from handbook import (  # noqa: E402
    DEFAULT_OUTLINE,
    clean_heading,
    generate_outline,
    get_pages_from_hit,
    slugify,
    word_count,
)
from ingest import chunk_pages  # noqa: E402
from llm_mock import MockLLM  # noqa: E402


class ScriptedLLM:
    """Returns a fixed response, so outline parsing can be tested in isolation."""

    def __init__(self, response: str) -> None:
        self.response = response

    def generate(self, prompt: str) -> str:
        return self.response


# ------------------------------------------------------------------- chunking
def test_chunk_pages_preserves_page_numbers():
    pages = [(1, "alpha " * 400), (2, "beta " * 400)]
    chunks = chunk_pages(pages, chunk_size=500, overlap=50)

    assert chunks, "expected at least one chunk"
    assert all(len(c.pages) == 1 for c in chunks)
    assert {p for c in chunks for p in c.pages} == {1, 2}
    assert [c.idx for c in chunks] == list(range(len(chunks)))


def test_chunk_pages_skips_empty_pages():
    assert chunk_pages([(1, "   "), (2, "")]) == []


def test_chunk_pages_rejects_overlap_larger_than_chunk():
    # Without this guard the sliding window would never advance and the loop
    # would spin forever on any page longer than chunk_size.
    with pytest.raises(ValueError):
        chunk_pages([(1, "text " * 200)], chunk_size=100, overlap=100)


def test_chunk_pages_terminates_on_long_page():
    chunks = chunk_pages([(1, "x" * 10_000)], chunk_size=900, overlap=120)
    assert len(chunks) > 1


# -------------------------------------------------------------------- helpers
def test_word_count_ignores_markdown_punctuation():
    assert word_count("## Heading\n\n- one, two; three!") == 4


def test_slugify_matches_anchor_style():
    assert slugify("Data Ingestion and Parsing") == "data-ingestion-and-parsing"
    assert slugify("Security, Privacy, and Compliance") == "security-privacy-and-compliance"


def test_clean_heading_truncates_and_defaults():
    assert clean_heading("!!!") == "Section"
    assert len(clean_heading("x" * 300)) == 120


@pytest.mark.parametrize(
    "hit,expected",
    [
        ({"pages": [1, 2]}, [1, 2]),
        ({"metadata": {"pages": [3]}}, [3]),
        ({"pages": ["4", "bad"]}, [4]),
        ({}, []),
        ({"pages": None, "metadata": None}, []),
    ],
)
def test_get_pages_from_hit(hit, expected):
    assert get_pages_from_hit(hit) == expected


# -------------------------------------------------------------------- outline
@pytest.mark.parametrize("sep", [".", ")", "-", ":"])
def test_generate_outline_accepts_numbering_styles(sep):
    body = "\n".join(f"{i}{sep} Section {i}" for i in range(1, 13))
    outline = generate_outline(ScriptedLLM(body), "Topic")
    assert len(outline) == 12


def test_generate_outline_falls_back_when_unparseable():
    outline = generate_outline(ScriptedLLM("no numbered list here"), "Topic")
    assert outline == DEFAULT_OUTLINE


def test_generate_outline_falls_back_on_too_few_sections():
    outline = generate_outline(ScriptedLLM("1. One\n2. Two\n3. Three"), "Topic")
    assert outline == DEFAULT_OUTLINE


# ----------------------------------------------------------------------- mock
def test_mock_llm_outline_mode_is_parseable():
    mock = MockLLM()
    reply = mock.generate("Return ONLY a numbered outline with 12-18 sections")
    outline = generate_outline(ScriptedLLM(reply), "Topic")
    assert len(outline) >= 8


def test_mock_llm_section_reaches_length_floor():
    text = MockLLM().generate('Now write this section:\n- Start with "## Alpha"')
    assert word_count(text) >= 1600
    assert text.startswith("## Alpha")
