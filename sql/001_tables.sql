-- SilverAI Nano - core schema
-- Run once against your Supabase project (SQL Editor or psql).

create extension if not exists vector;

create table if not exists documents (
  id uuid primary key default gen_random_uuid(),
  filename text not null,
  created_at timestamptz default now()
);

create table if not exists chunks (
  id uuid primary key default gen_random_uuid(),
  doc_id uuid references documents(id) on delete cascade,
  chunk_index int not null,
  content text not null,
  pages int[] not null default '{}',
  embedding vector(384) not null,
  created_at timestamptz default now()
);

create index if not exists chunks_doc_id_idx on chunks(doc_id);

-- Note on the vector index: ivfflat builds its clusters from the rows present
-- at creation time, so an index created on an empty table gives poor recall.
-- Create it after your first ingest, and set `lists` to roughly rows/1000
-- (minimum 1). Re-run this statement after a large ingest to rebuild.
create index if not exists chunks_embedding_idx
  on chunks using ivfflat (embedding vector_cosine_ops) with (lists = 100);
