# app/llm_mock.py
from __future__ import annotations

import random
import re
import textwrap

from llm_base import LLMClient


class MockLLM(LLMClient):
    """
    Generates handbook-like filler long enough to test orchestration.
    NOT for final quality—just to reach 20k+ words reliably.
    """
    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)

    def _words(self, s: str) -> int:
        return len(re.findall(r"\b\w+\b", s))

    def generate(self, prompt: str) -> str:
        teaser = prompt.strip().replace("\n", " ")[:260]

        # Outline mode
        if "Return ONLY a numbered outline" in prompt:
            headings = [
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
            return "\n".join([f"{i+1}. {h}" for i, h in enumerate(headings)])

        # Memory summary mode
        if prompt.strip().lower().startswith("summarize"):
            bullets = [f"- Key point {i+1}: {teaser} (mock summary)." for i in range(8)]
            return "\n".join(bullets)

        # Try to respect requested heading if present in prompt
        heading_match = re.search(r'Now write this section:\s*-\s*Start with "##\s*(.+?)"', prompt, re.IGNORECASE)
        section_heading = heading_match.group(1).strip() if heading_match else None

        paras = []
        if section_heading:
            paras.append(f"## {section_heading}\n")

        while True:
            cite_page = self._rng.randint(1, 5)
            p = (
                f"Based on: {teaser}. This paragraph expands definitions, gives practical examples, "
                f"explains tradeoffs, and includes implementation considerations. "
                f"It is intentionally verbose to test long-form orchestration. (PDF p. {cite_page})"
            )
            paras.append(textwrap.fill(p, width=100))

            if len(paras) % 5 == 0:
                paras.append("### Practical checklist (mock)\n- Step A\n- Step B\n- Step C\n")
            if len(paras) % 7 == 0:
                paras.append("### Common pitfalls (mock)\n- Pitfall 1\n- Pitfall 2\n- Pitfall 3\n")

            text = "\n\n".join(paras).strip()
            if self._words(text) >= 1600:
                return text
