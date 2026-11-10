-- Поиск по фрагментам: векторы (смысл) и полнотекстовый индекс (слова)

-- Размерность 768 — multilingual-e5-base. Смена модели с другой размерностью —
-- новая миграция; метрика (косинус) выбрана один раз, индекс построен под неё.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding vector(768);

-- Поиск по словам: генерируемая колонка, обновляется сама вместе с text
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('russian', text)) STORED;

-- Приближённый индекс по векторам (HNSW, косинусное расстояние)
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Инвертированный индекс по словам
CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);
