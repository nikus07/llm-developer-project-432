.PHONY: help infra migrate lint test api temporal-worker

help:
	@echo "make infra           - поднять postgres + temporal + rabbitmq (docker compose)"
	@echo "make api             - FastAPI на :8000"
	@echo "make temporal-worker - воркер процессов (workflow + activities)"

infra:
	docker compose up -d

migrate:
	python -m app.core.migrate

lint:
	ruff check .

test:
	pytest

api:
	uvicorn app.main:app --reload --port 8000

temporal-worker:
	python -m app.temporal_worker
