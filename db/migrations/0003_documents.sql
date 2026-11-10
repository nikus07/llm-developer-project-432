-- Документы поставщиков и их фрагменты
CREATE TABLE IF NOT EXISTS documents (
    id           TEXT PRIMARY KEY,
    filename     TEXT NOT NULL,
    -- хэш содержимого: повторная загрузка того же файла не создаёт дубль
    content_hash TEXT NOT NULL UNIQUE,
    size_bytes   INTEGER NOT NULL,
    status       TEXT NOT NULL DEFAULT 'new',  -- new | parsing | indexed | failed
    reason       TEXT,                         -- причина отказа
    pages        INTEGER,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    document    TEXT NOT NULL,      -- имя файла: для ссылки на источник
    idx         INTEGER NOT NULL,   -- порядковый номер фрагмента в документе
    kind        TEXT NOT NULL,      -- text | row
    text        TEXT NOT NULL,
    page        INTEGER NOT NULL,
    section     TEXT NOT NULL,
    article     TEXT,
    brand       TEXT,
    UNIQUE (document_id, idx)
);

CREATE INDEX IF NOT EXISTS chunks_article_idx ON chunks (article);
