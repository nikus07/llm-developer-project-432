-- Документы поставщиков и их фрагменты
CREATE TABLE IF NOT EXISTS documents (
    id           TEXT PRIMARY KEY,
    filename     TEXT NOT NULL,
    -- хэш содержимого: повторная загрузка того же файла не создаёт дубль
    content_hash TEXT NOT NULL UNIQUE,
    kind         TEXT NOT NULL,                -- pdf | docx | xlsx
    size_bytes   INTEGER,
    status       TEXT NOT NULL DEFAULT 'new',  -- new | parsing | parsed | indexed | failed
    reason       TEXT,                         -- причина отказа
    pages        INTEGER,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id       TEXT PRIMARY KEY,                 -- на него ссылается карточка: «<doc_id>-<ordinal>»
    doc_id   TEXT NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    ordinal  INTEGER NOT NULL,                 -- порядковый номер фрагмента в документе
    text     TEXT NOT NULL,
    -- doc_id, page, section, kind (text | row), articul, brand — для фильтров и ссылок
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (doc_id, ordinal)
);

CREATE INDEX IF NOT EXISTS chunks_doc_idx ON chunks (doc_id);
CREATE INDEX IF NOT EXISTS chunks_articul_idx ON chunks ((metadata ->> 'articul'));
