-- Учёт вызовов модели: пишет клиент модели после каждого вызова
CREATE TABLE IF NOT EXISTS llm_calls (
    id            BIGSERIAL PRIMARY KEY,
    job_id        TEXT,               -- задача (или eval-прогон); NULL у вызовов вне задачи
    request_id    TEXT,               -- HTTP-запрос, с которого всё началось
    agent         TEXT NOT NULL,
    model         TEXT NOT NULL,
    tier          TEXT NOT NULL,      -- main | light
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cost          NUMERIC(14, 6) NOT NULL DEFAULT 0,  -- ₽; деньги не во float
    duration_ms   INTEGER NOT NULL,
    status        TEXT NOT NULL DEFAULT 'ok',         -- ok | error
    error         TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS llm_calls_job_idx ON llm_calls (job_id);
CREATE INDEX IF NOT EXISTS llm_calls_model_idx ON llm_calls (model, tier);
