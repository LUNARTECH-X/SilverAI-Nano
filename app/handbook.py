# app/handbook.py
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from llm_base import LLMClient
from retrieve import retrieve_context
from settings import TARGET_HANDBOOK_WORDS

WORD_RE = re.compile(r"\b\w+\b")

DEFAULT_OUTLINE = [
    "Introduction and Goals",
    "Core Concepts and Terminology",
    "System Architecture Overview",
    "Data Ingestion and Parsing",
    "Chunking Strategy",
    "Embeddings and Vector Storage",
    "Retrieval Strategies",
    "Knowledge Graph Augmentation",
    "Prompting and Context Packing",
    "Evaluation and Metrics",
    "Deployment and Operations",
    "Security, Privacy, and Compliance",
    "Troubleshooting and Failure Modes",
    "Case Studies and Examples",
    "Future Directions",
    "Appendix",
]


def as_text(resp) -> str:
    """Accept either a plain string or an object exposing `.text`."""
    value = getattr(resp, "text", resp)
    return value if isinstance(value, str) else str(value or "")


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def clean_heading(s: str) -> str:
    s = re.sub(r"[^\w\s\-:]", "", s).strip()
    return s[:120] if s else "Section"


def slugify(heading: str) -> str:
    """GitHub-style anchor slug for the table of contents."""
    slug = re.sub(r"[^a-z0-9\s-]", "", heading.lower())
    return re.sub(r"\s+", "-", slug.strip())


def get_pages_from_hit(hit: dict) -> List[int]:
    """Read page numbers from a retrieval hit, tolerating either row shape."""
    pages = hit.get("pages")
    if pages is None:
        pages = (hit.get("metadata") or {}).get("pages")
    if not pages:
        return []
    out: List[int] = []
    for p in pages:
        try:
            out.append(int(p))
        except (TypeError, ValueError):
            continue
    return out


@dataclass
class HandbookResult:
    title: str
    outline: List[str]
    markdown: str
    words: int
    sections_written: int = 0
    grounded: bool = True
    warnings: List[str] = field(default_factory=list)


def generate_outline(llm: LLMClient, topic: str) -> List[str]:
    prompt = f"""
You are creating a detailed handbook outline.

Topic: {topic}

Return ONLY a numbered outline with 12-18 sections, each as a short heading.
Example:
1. Introduction
2. Key Concepts
3. ...
"""
    text = as_text(llm.generate(prompt))

    outline: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Accept "1. X", "1) X", "1 - X" and "1: X".
        m = re.match(r"^\s*\d+\s*[).\-:]\s*(.+)$", line)
        if m:
            outline.append(clean_heading(m.group(1)))

    # A too-short outline means the model ignored the format; fall back rather
    # than generating a handbook with three sections.
    if len(outline) < 8:
        return list(DEFAULT_OUTLINE)
    return outline


def _build_context(hits: List[Dict]) -> tuple[str, List[int]]:
    allowed_pages = sorted({p for h in hits for p in get_pages_from_hit(h)})
    blocks: List[str] = []
    for h in hits:
        pages = get_pages_from_hit(h)
        sim = float(h.get("similarity", 0.0) or 0.0)
        content = h.get("content", "") or ""
        blocks.append(f"[pages={pages} sim={sim:.3f}]\n{content}".strip())
    return "\n\n---\n\n".join(blocks), allowed_pages


def generate_handbook_markdown(
    llm: LLMClient,
    topic: str,
    document_id: Optional[str],
    target_words: int = TARGET_HANDBOOK_WORDS,
    top_k_context: int = 8,
    progress_cb: Optional[Callable[[str, float], None]] = None,
) -> HandbookResult:
    """
    Generate a long-form handbook section by section, grounded in retrieved PDF
    context and carrying a rolling summary forward for continuity.

    Retrieval failures propagate, so a misconfigured database fails immediately
    rather than after burning tokens on an ungrounded document.
    """
    outline = generate_outline(llm, topic)

    title = f"{topic} - Handbook"
    md_parts: List[str] = [f"# {title}\n", "## Table of Contents\n"]
    for i, h in enumerate(outline, start=1):
        md_parts.append(f"{i}. [{h}](#{slugify(h)})")
    md_parts.append("\n---\n")

    memory = ""
    warnings: List[str] = []
    # Track the running total incrementally; re-joining and re-scanning the whole
    # document on every section is quadratic in the length of the handbook.
    total_words = sum(word_count(p) for p in md_parts)
    sections_written = 0
    n = max(len(outline), 1)

    for idx, heading in enumerate(outline, start=1):
        if progress_cb:
            progress_cb(f"Retrieving context for: {heading}", (idx - 1) / n)

        hits = retrieve_context(
            f"{topic} - {heading}",
            top_k=top_k_context,
            document_id=document_id,
        )
        context_block, allowed_pages = _build_context(hits)
        if not hits:
            warnings.append(f"No source context retrieved for section: {heading}")

        prompt = f"""
You are writing a structured handbook.

Topic: {topic}
Current section: {heading}

You MUST:
- Write in Markdown
- Include subsections (###) and bullet lists where useful
- Be detailed, practical, and explanatory
- Ground content in the provided sources when available

Citation rules:
- Cite ONLY from these pages: {allowed_pages}
- Use the format (PDF p. X)
- If sources do not support a claim, say so clearly.

Continuity notes from previous sections:
{memory[:2000]}

Source excerpts:
{context_block[:12000]}

Now write this section:
- Start with "## {heading}"
- Aim for 1200-1800 words (if possible)
"""

        if progress_cb:
            progress_cb(f"Generating section: {heading}", (idx - 0.5) / n)

        section_md = as_text(llm.generate(prompt)).strip()
        if not section_md.startswith("## "):
            section_md = f"## {heading}\n\n{section_md}"

        md_parts.append(section_md)
        total_words += word_count(section_md)
        sections_written += 1

        memory = as_text(
            llm.generate(
                "Summarize the key points from the latest section in 6-10 "
                f"concise bullet points.\n\nSection text:\n{section_md[:12000]}"
            )
        ).strip()

        if progress_cb:
            progress_cb(f"Progress: {total_words} words", idx / n)

        if total_words >= target_words:
            break

    if progress_cb:
        progress_cb("Generating final conclusion...", 1.0)

    conclusion_md = as_text(
        llm.generate(
            f"""
Write a final conclusion section in Markdown.

Topic: {topic}

Requirements:
- Start with "## Conclusion"
- Summarize the key takeaways
- Provide a short checklist of next steps
- Add a brief glossary of 8-12 terms
- End with a complete final sentence (no cutoff)
"""
        )
    ).strip()
    if not conclusion_md.startswith("## "):
        conclusion_md = f"## Conclusion\n\n{conclusion_md}"
    md_parts.append(conclusion_md)

    final_md = "\n\n".join(md_parts)
    return HandbookResult(
        title=title,
        outline=outline,
        markdown=final_md,
        words=word_count(final_md),
        sections_written=sections_written,
        grounded=len(warnings) < sections_written,
        warnings=warnings,
    )
