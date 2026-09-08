-- SilverAI Nano - similarity search RPC
-- `filter_doc_id` is optional: pass null to search across every document.

create or replace function match_chunks(
  query_embedding vector(384),
  match_count int default 6,
  filter_doc_id uuid default null
)
returns table (
  id uuid,
  doc_id uuid,
  chunk_index int,
  content text,
  pages int[],
  similarity float
)
language sql stable
as $$
  select
    c.id,
    c.doc_id,
    c.chunk_index,
    c.content,
    c.pages,
    1 - (c.embedding <=> query_embedding) as similarity
  from chunks c
  where filter_doc_id is null or c.doc_id = filter_doc_id
  order by c.embedding <=> query_embedding
  limit match_count;
$$;
