# app/main.py
from __future__ import annotations

import os
import tempfile
from typing import List, Optional, Tuple

import streamlit as st

from handbook import generate_handbook_markdown, get_pages_from_hit
from ingest import ingest_pdf_to_supabase
from llm_base import LLMClient
from llm_mock import MockLLM
from retrieve import RetrievalError, retrieve_context
from settings import TARGET_HANDBOOK_WORDS, api_key

st.set_page_config(page_title="SilverAI Nano", layout="wide")


def as_text(resp) -> str:
    value = getattr(resp, "text", resp)
    return value if isinstance(value, str) else str(value or "")


@st.cache_resource(show_spinner=False)
def _build_grok() -> LLMClient:
    from llm_grok import GrokLLM

    return GrokLLM()


def get_llm() -> Tuple[LLMClient, Optional[str]]:
    """
    Return the active model and, when running on the mock, the reason why.

    The mock is deliberately not cached: if a credential is added later, the
    next rerun picks up the real client instead of staying degraded for the
    lifetime of the process.
    """
    if not api_key():
        return MockLLM(), "No XAI_API_KEY (or GROK_API_KEY) is configured."
    try:
        return _build_grok(), None
    except Exception as exc:  # noqa: BLE001 - surfaced to the user below
        return MockLLM(), str(exc)


def format_context(hits: List[dict]) -> str:
    blocks: List[str] = []
    for h in hits:
        pages = get_pages_from_hit(h)
        sim = float(h.get("similarity", 0.0) or 0.0)
        content = h.get("content", "") or ""
        blocks.append(f"[pages={pages} sim={sim:.3f}]\n{content}".strip())
    return "\n\n---\n\n".join(blocks).strip()


# ---------------------------------------------------------------- session state
defaults = {
    "doc_id": None,
    "doc_name": None,
    "messages": [],
    "latest_handbook_md": "",
    "latest_handbook_topic": "",
    "latest_handbook_words": 0,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

llm, mock_reason = get_llm()

st.title("SilverAI Nano")
st.caption("Turn a PDF into a searchable knowledge base and a long-form handbook.")

# A mock run produces filler that reads like a finished document. Say so loudly
# and persistently rather than only printing to the server console.
if mock_reason:
    st.warning(
        "**Running on the mock model - output is placeholder text, not real "
        f"analysis.** {mock_reason} Set `XAI_API_KEY` in your `.env` and reload "
        "to generate genuine content."
    )

# ---------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Upload PDF")
    uploaded = st.file_uploader(
        "Upload a text-based PDF (not scanned images)",
        type=["pdf"],
        accept_multiple_files=False,
    )
    if uploaded is not None:
        st.caption(f"Selected: **{uploaded.name}**")

    if st.button("Index PDF", disabled=(uploaded is None), use_container_width=True):
        tmp_path = None
        with st.spinner("Extracting, chunking, embedding, uploading..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded.getbuffer())
                    tmp_path = tmp.name

                doc_id = ingest_pdf_to_supabase(tmp_path, filename=uploaded.name)
                st.session_state.doc_id = doc_id
                st.session_state.doc_name = uploaded.name
                st.success(f"Indexed. document_id={doc_id}")
            except Exception as exc:  # noqa: BLE001 - shown to the user
                st.error(f"Indexing failed: {exc}")
            finally:
                if tmp_path:
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

    st.divider()
    st.subheader("Status")
    st.write("**Model:**", "Mock (placeholder output)" if mock_reason else "Grok")
    if st.session_state.doc_id:
        st.write("**Active document:**", st.session_state.doc_name)
        st.write("**document_id:**", st.session_state.doc_id)
    else:
        st.write("No PDF indexed yet.")

    st.divider()
    st.subheader("Downloads")
    if st.session_state.latest_handbook_md:
        st.write(f"Latest handbook: **{st.session_state.latest_handbook_words} words**")
        topic = st.session_state.latest_handbook_topic or "handbook"
        st.download_button(
            "Download handbook (Markdown)",
            data=st.session_state.latest_handbook_md.encode("utf-8"),
            file_name=f"{topic.lower().replace(' ', '_')}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    else:
        st.caption("Generate a handbook with `/handbook <topic>` to enable downloads.")

    st.divider()
    if st.button("Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ------------------------------------------------------------------------- chat
st.subheader("Chat")

if not st.session_state.messages:
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": (
                "Upload a PDF in the sidebar and click **Index PDF**.\n\n"
                "Then ask questions and I will answer from that document.\n\n"
                f"Use `/handbook <topic>` to generate a "
                f"**{TARGET_HANDBOOK_WORDS:,}+ word** handbook grounded in it."
            ),
        }
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# The preview lives outside the chat loop so it survives the rerun that follows
# generation, instead of being drawn and immediately discarded.
if st.session_state.latest_handbook_md:
    with st.expander("Handbook preview (first 6000 characters)"):
        st.markdown(st.session_state.latest_handbook_md[:6000])

prompt = st.chat_input("Ask a question, or use /handbook <topic>")


def say(content: str) -> None:
    st.session_state.messages.append({"role": "assistant", "content": content})
    with st.chat_message("assistant"):
        st.markdown(content)


if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if not st.session_state.doc_id:
        say(
            "This chat answers from **uploaded PDFs**. Please upload and index a "
            "PDF in the sidebar first."
        )
        st.stop()

    # ------------------------------------------------------------- /handbook
    if prompt.strip().lower().startswith("/handbook"):
        topic = prompt.strip()[len("/handbook"):].strip() or "Handbook Topic"

        with st.chat_message("assistant"):
            progress = st.progress(0)
            status = st.empty()

            def progress_cb(msg: str, frac: float) -> None:
                status.write(msg)
                progress.progress(min(max(frac, 0.0), 1.0))

            try:
                result = generate_handbook_markdown(
                    llm=llm,
                    topic=topic,
                    document_id=st.session_state.doc_id,
                    target_words=TARGET_HANDBOOK_WORDS,
                    progress_cb=progress_cb,
                )
            except RetrievalError as exc:
                st.error(str(exc))
                st.stop()
            except Exception as exc:  # noqa: BLE001 - shown to the user
                st.error(f"Handbook generation failed: {exc}")
                st.stop()

        st.session_state.latest_handbook_md = result.markdown
        st.session_state.latest_handbook_topic = topic
        st.session_state.latest_handbook_words = result.words

        summary = (
            f"Generated **{result.words:,} words** across "
            f"{result.sections_written} sections for **{topic}**.\n\n"
            "Download the full handbook from the sidebar."
        )
        if mock_reason:
            summary += (
                "\n\n**Generated with the mock model - the text is placeholder "
                "filler, not real analysis.**"
            )
        if result.warnings:
            summary += (
                f"\n\n{len(result.warnings)} section(s) had no matching source "
                "context and may be weakly grounded."
            )
        st.session_state.messages.append({"role": "assistant", "content": summary})
        st.rerun()

    # ------------------------------------------------------------ normal Q&A
    try:
        hits = retrieve_context(prompt, top_k=6, document_id=st.session_state.doc_id)
    except RetrievalError as exc:
        say(f"Retrieval failed: {exc}")
        st.stop()

    if not hits:
        say("The uploaded PDFs do not mention this.")
        st.stop()

    allowed_pages = sorted({p for h in hits for p in get_pages_from_hit(h)})
    rag_prompt = f"""
Answer the question using ONLY the source excerpts.

Question:
{prompt}

Source excerpts (with page tags):
{format_context(hits)}

Rules:
- If the excerpts do not contain the answer, say: "The uploaded PDFs do not mention this."
- Be clear and concise.
- Cite ONLY from these pages: {allowed_pages}. Use (PDF p. X).
- If you cannot cite from allowed pages, say: "The uploaded PDFs do not mention this."
"""

    try:
        answer = as_text(llm.generate(rag_prompt))
    except Exception as exc:  # noqa: BLE001 - degraded answer beats a traceback
        answer = (
            f"**The model call failed, so this answer is mock filler.** ({exc})\n\n"
            + MockLLM().generate(rag_prompt)
        )

    say(answer)
