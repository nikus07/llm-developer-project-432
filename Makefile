.PHONY: help infra migrate reindex search lint test api temporal-worker

help:
	@echo "make infra           - поднять postgres + temporal + rabbitmq (docker compose)"
	@echo "make reindex         - досчитать векторы фрагментам без вектора"
	@echo "make search Q=...   - поиск по фрагментам (гибрид)"
	@echo "make api             - FastAPI на :8000"
	@echo "make temporal-worker - воркер процессов (workflow + activities)"

infra:
	docker compose up -d

migrate:
	python -m app.core.migrate

reindex:
	python -m app.rag.reindex

search:
	python -m app.rag.retrieval "$(Q)"

lint:
	ruff check .

test:
	pytest

api:
	uvicorn app.main:app --reload --port 8000

temporal-worker:
	python -m app.temporal_worker
