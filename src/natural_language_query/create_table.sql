-- Create the table to store GDELT event data and their vector embeddings
CREATE TABLE IF NOT EXISTS gdelt_embeddings (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding VECTOR(1536)
);

-- Create an HNSW index for high-dimensional vector search.
-- HNSW is generally preferred for its performance and accuracy, and it does not have the 2000-dimension limit.
CREATE INDEX IF NOT EXISTS idx_hnsw_embedding ON gdelt_embeddings USING hnsw (embedding vector_cosine_ops);
