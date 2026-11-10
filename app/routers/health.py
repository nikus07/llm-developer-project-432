from fastapi import APIRouter, HTTPException

from app.core.db import connection
from app.core.logging import get_logger

router = APIRouter(prefix="/health", tags=["health"])
log = get_logger(__name__)


@router.get("/live")
def live() -> dict:
    """Процесс жив. База не проверяется: живость не должна падать вместе с ней."""
    return {"status": "ok"}


@router.get("/ready")
def ready() -> dict:
    """Готов принимать работу: база отвечает и расширение vector активно."""
    try:
        with connection(timeout=2) as conn:
            row = conn.execute(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            ).fetchone()
    except Exception as exc:
        log.warning("readiness_failed", error=repr(exc))
        detail = {"status": "error", "database": "down"}
        raise HTTPException(status_code=503, detail=detail) from exc
    if row is None:
        raise HTTPException(
            status_code=503,
            detail={"status": "error", "database": "up", "vector": None},
        )
    return {"status": "ok", "database": "up", "vector": row[0]}
