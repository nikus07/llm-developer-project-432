"""Сквозные идентификаторы: запрос рождается на входе в приложение, задача — при создании.

Оба привязываются к контексту логов (structlog.contextvars): каждая запись лога несёт их
автоматически, а клиент модели берёт их оттуда для таблицы llm_calls. Один поиск по job_id
показывает весь путь задачи: обработчик, шаги процесса, вызовы модели с длительностью.
"""
import time
import uuid

import structlog
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.logging import get_logger

log = get_logger(__name__)

REQUEST_ID_HEADER = "x-request-id"


def current_request_id() -> str | None:
    return structlog.contextvars.get_contextvars().get("request_id")


def bind_job(job_id: str) -> None:
    """Задача создана: с этого момента все записи лога этого контекста несут её id."""
    structlog.contextvars.bind_contextvars(job_id=job_id)


class RequestContextMiddleware:
    """Идентификатор запроса из заголовка X-Request-ID или новый; запись в лог на каждый запрос.

    Чистый ASGI, а не BaseHTTPMiddleware: обработчик исполняется в той же задаче, поэтому
    привязанный им job_id виден итоговой записи лога об этом запросе.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope["headers"])
        request_id = headers.get(REQUEST_ID_HEADER.encode(), b"").decode() or uuid.uuid4().hex[:12]
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        started = time.perf_counter()
        status = 500

        async def send_with_id(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message.setdefault("headers", []).append(
                    (REQUEST_ID_HEADER.encode(), request_id.encode())
                )
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            log.info(
                "http_request",
                method=scope["method"],
                path=scope["path"],
                status=status,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )
            structlog.contextvars.clear_contextvars()
