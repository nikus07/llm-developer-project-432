-- Отчёт защиты: что замаскировано и что исключено из контекста (без исходных значений)
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS report JSONB;
