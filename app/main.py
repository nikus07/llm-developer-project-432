from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.db import close_pool, init_pool
from app.core.logging import setup_logging
from app.routers import cards, documents, health, jobs, workflows


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    init_pool()  # ресурс уровня приложения: живёт, пока живёт процесс
    yield
    close_pool()


app = FastAPI(title="card-pipeline", lifespan=lifespan)
app.include_router(health.router)
app.include_router(cards.router)
app.include_router(documents.router)
app.include_router(jobs.router)
app.include_router(workflows.router)
