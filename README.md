<p align="center">
  <img src="assets/lunartech-banner.png" alt="LUNARTECH" width="100%">
</p>

<h1 align="center">L U N A R T E C H</h1>

<p align="center">
  <a href="https://www.linkedin.com/company/lunartechai/">LUNARTECH on LinkedIn</a>
  &nbsp;·&nbsp;
  <a href="https://www.linkedin.com/in/vahe-aslanyan/">Vahe Aslanyan on LinkedIn</a>
</p>

**LUNARTECH** builds applied AI systems and the training that goes with them.

| | |
|---|---|
| **Organization** | LUNARTECH |
| **Project** | SilverAI Nano |
| **LUNARTECH on LinkedIn** | https://www.linkedin.com/company/lunartechai/ |
| **Vahe Aslanyan on LinkedIn** | https://www.linkedin.com/in/vahe-aslanyan/ |

---

# SilverAI Nano

**A LUNARTECH project.**

SilverAI Nano turns a PDF into a searchable knowledge base and, from that same
source material, generates a structured long-form handbook of 20,000 words or
more. It is a compact reference implementation of retrieval-augmented long-form
generation: small enough to read end to end in an afternoon, complete enough to
run against a real database and a real model.

The problem it solves is that language models cap their output well below the
length of a genuine handbook. Asking for 20,000 words in a single call returns a
few thousand at best, and quality degrades as the model loses track of what it
has already written. SilverAI Nano works the way a human author does instead. It
plans an outline first, writes one section at a time against retrieved source
passages, and carries a rolling summary of everything written so far into the
next section so the document stays coherent from beginning to end.

---

## How it works

```
PDF upload
    |
    v
pdfplumber           extract text, one page at a time
    |
    v
page-aware chunking  900 chars, 120 overlap, page numbers preserved per chunk
    |
    v
MiniLM embeddings    384-dim, normalized
    |
    v
Supabase + pgvector  documents and chunks tables, cosine similarity index
    |
    v
match_chunks RPC     top-k retrieval, optionally scoped to one document
    |
    v
Grok                 section-by-section generation with rolling memory
    |
    v
Markdown handbook    table of contents, cited sections, conclusion, glossary
```

Chunking happens within page boundaries rather than across them, so every chunk
carries the exact page it came from. That page number travels through retrieval
into the generation prompt, which lets the model cite `(PDF p. 7)` accurately and
lets you verify any claim against the original document. The generator is told
explicitly which pages it is allowed to cite, so it cannot invent a reference to
a page that was never retrieved.

---

## Quickstart

**Requirements:** Python 3.10 or newer, a Supabase project, and an xAI API key.
The application runs without the API key, but see "Mock mode" below.

```bash
git clone https://github.com/LUNARTECH-X/SilverAI-Nano.git
cd SilverAI-Nano

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Apply the database schema. In the Supabase SQL editor, run the two migrations in
order:

```
sql/001_tables.sql        tables, pgvector extension, indexes
sql/002_match_chunks.sql  similarity search function
```

Configure your credentials, then start the app:

```bash
cp .env.example .env      # fill in SUPABASE_URL, SUPABASE_SERVICE_KEY, XAI_API_KEY
streamlit run app/main.py
```

Upload a text-based PDF in the sidebar, click **Index PDF**, then ask questions
in the chat. To produce a full handbook, type:

```
/handbook Retrieval-Augmented Generation
```

Generation takes several minutes and reports progress per section. The finished
Markdown is downloadable from the sidebar.

---

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `SUPABASE_URL` | required | Supabase project URL |
| `SUPABASE_SERVICE_KEY` | required | Service role key; server-side only |
| `XAI_API_KEY` | optional | xAI credential; `GROK_API_KEY` also accepted |
| `GROK_MODEL` | `grok-4.20-0309-non-reasoning` | Generation model |
| `GROK_MAX_TOKENS` | `4000` | Ceiling per section |
| `GROK_MAX_RETRIES` | `4` | Retry budget for transient API failures |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Embedding model |
| `EMBED_DIM` | `384` | Must match `vector(N)` in the schema |
| `TARGET_HANDBOOK_WORDS` | `20000` | Word target; generation stops once reached |

`EMBED_DIM` and the `vector(384)` columns in `sql/001_tables.sql` must agree. The
application checks this at startup and fails with an explicit message rather than
writing vectors the database will silently reject.

---

## Mock mode

When no API key is configured, the app falls back to a deterministic mock model
so the full pipeline stays exercisable without spending credits. Mock output is
structurally valid filler, not analysis, and the interface says so continuously:
a banner at the top of the page, a status line in the sidebar, and a note
attached to every handbook generated this way. This matters because a mock
handbook is 20,000 words long and looks finished at a glance. Adding a key and
reloading switches to the real model without clearing any cache.

---

## Layout

```
app/
  main.py        Streamlit interface, chat, handbook command
  settings.py    environment loading and validation
  clients.py     cached Supabase client and embedding model
  ingest.py      PDF extraction, chunking, embedding, storage
  retrieve.py    similarity search via the match_chunks RPC
  handbook.py    outline planning and section-by-section generation
  llm_base.py    model interface
  llm_grok.py    Grok client with exponential backoff
  llm_mock.py    offline stand-in
sql/             database migrations
tests/           offline tests for the deterministic pipeline
```

## Tests

```bash
pytest tests/
```

The suite covers chunking, page-metadata propagation, outline parsing across
numbering styles, fallback behavior, and the mock model's output contract. It
requires neither a database, an API key, nor the embedding model, so it runs in
CI in under a second.

---

## Operational notes

The `chunks_embedding_idx` ivfflat index builds its clusters from the rows
present when it is created, so an index created on an empty table gives poor
recall. Create it after your first ingest and rebuild it after any large one,
setting `lists` to roughly the row count divided by 1000.

Scanned or image-only PDFs contain no extractable text layer and are rejected at
upload with a clear message; run them through OCR first. The service role key
bypasses row-level security and must never be exposed to a browser client.

A failed ingest removes its own orphaned document row, so a partial upload does
not leave a document behind with no retrievable chunks.

---

## About LUNARTECH

LUNARTECH builds applied AI systems and the training that goes with them. Follow
the work on LinkedIn:

- LUNARTECH: https://www.linkedin.com/company/lunartechai/
- Vahe Aslanyan: https://www.linkedin.com/in/vahe-aslanyan/

Built and maintained by LUNARTECH.
