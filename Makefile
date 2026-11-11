.PHONY: help setup lint lint-fix test infra migrate api temporal-worker reindex search costs eval

help:
	@echo "make setup           - зависимости из uv.lock и миграции (база уже должна быть поднята)"
	@echo "make infra           - поднять postgres + temporal (docker compose)"
	@echo "make migrate         - применить миграции"
	@echo "make api             - FastAPI на :8000"
	@echo "make temporal-worker - воркер процессов (workflow + activities)"
	@echo "make reindex         - досчитать векторы фрагментам без вектора"
	@echo "make search Q=...    - поиск по фрагментам (гибрид)"
	@echo "make costs [JOB=id]  - стоимость вызовов модели"
	@echo "make eval [ALL=1]    - метрики качества на эталоне"
	@echo "make test / lint     - тесты и линтер"

setup:
	uv sync --frozen
	uv run --frozen python -m app.core.migrate

lint:
	uv run ruff check .

lint-fix:
	uv run ruff check . --fix

test:
	uv run pytest

infra:
	docker compose up -d

migrate:
	uv run --frozen python -m app.core.migrate

api:
	uv run uvicorn app.main:app --reload --port 8000

temporal-worker:
	uv run python -m app.temporal_worker

reindex:
	uv run python -m app.rag.reindex

search:
	uv run python -m app.rag.retrieval "$(Q)"

costs:
	uv run python -m app.llm.report $(if $(JOB),--job $(JOB),)

eval:
	uv run python -m evals.run_metrics $(if $(ALL),--all,)
