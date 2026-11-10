-- Расширение pgvector (образ с собранным pgvector — в docker-compose.yml)
CREATE EXTENSION IF NOT EXISTS vector;

-- Версии, которые уже применил раннер (app/core/migrate.py)
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
