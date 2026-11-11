"""Единственный файл, который знает про провайдера модели.

Здесь же — учёт: после каждого вызова клиент пишет строку в llm_calls (задача из контекста
логов, расход токенов из ответа, длительность). Вызывающий код про учёт не знает, поэтому
забыть его в новом месте нельзя: все вызовы идут через эту границу.
"""
import asyncio
import random
import time

import structlog
from agents import (
    Agent,
    Runner,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

from app.core.config import settings
from app.core.logging import get_logger
from app.llm.cost import call_cost
from app.llm.policy import tier_for
from app.repositories import llm_calls as llm_calls_repo

log = get_logger(__name__)

_client_configured = False


def setup_llm() -> None:
    """Настраивает SDK один раз на процесс (веб и temporal-воркеры)."""
    global _client_configured
    if _client_configured:
        return
    client = AsyncOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        timeout=settings.llm_timeout_s,
    )
    set_default_openai_client(client)
    # Локальные серверы реализуют chat completions, а не Responses API.
    set_default_openai_api("chat_completions")
    # Трассировки уходят на сервер OpenAI с чужим ключом — отключаем.
    set_tracing_disabled(True)
    _client_configured = True


def _retryable(exc: Exception) -> bool:
    """Повторяем то, что может пройти само: сеть, таймаут, 429, 5xx."""
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code == 429 or exc.status_code >= 500
    return False


def _usage(result) -> tuple[int, int]:
    """Расход токенов, который модель вернула вместе с ответом."""
    usage = getattr(getattr(result, "context_wrapper", None), "usage", None)
    if usage is None:
        return 0, 0
    return int(usage.input_tokens or 0), int(usage.output_tokens or 0)


async def _record(
    agent: Agent,
    started: float,
    *,
    result=None,
    error: Exception | None = None,
) -> None:
    """Лог и строка в llm_calls. Сбой учёта не должен ронять вызов модели."""
    duration_ms = int((time.perf_counter() - started) * 1000)
    tier = tier_for(agent.name)
    model = str(getattr(agent, "model", None) or settings.llm_model)
    input_tokens, output_tokens = _usage(result) if error is None else (0, 0)
    cost = call_cost(tier, input_tokens, output_tokens)
    ids = structlog.contextvars.get_contextvars()
    status = "ok" if error is None else "error"

    log.info(
        "llm_call_done" if error is None else "llm_call_error",
        agent=agent.name,
        model=model,
        tier=tier,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost=str(cost),
        duration_ms=duration_ms,
    )
    try:
        await asyncio.to_thread(
            llm_calls_repo.insert,
            job_id=ids.get("job_id"),
            request_id=ids.get("request_id"),
            agent=agent.name,
            model=model,
            tier=tier,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            duration_ms=duration_ms,
            status=status,
            error=repr(error)[:500] if error else None,
        )
    except Exception as exc:  # пул не создан, база недоступна — вызов важнее учёта
        log.warning("llm_call_record_failed", error=repr(exc))


async def run_agent(agent: Agent, prompt: str):
    setup_llm()
    max_retries = settings.llm_max_retries
    for attempt in range(1, max_retries + 1):
        started = time.perf_counter()
        try:
            log.info("llm_call", agent=agent.name, attempt=attempt)
            result = await asyncio.wait_for(
                Runner.run(agent, prompt), timeout=settings.llm_timeout_s
            )
        except Exception as exc:
            await _record(agent, started, error=exc)
            if attempt == max_retries or not _retryable(exc):
                log.error(
                    "llm_call_failed",
                    agent=agent.name,
                    attempt=attempt,
                    error=repr(exc),
                )
                raise
            delay = min(2**attempt * 0.5, 8.0) * (1 + random.random() * 0.25)
            log.warning(
                "llm_call_retry",
                agent=agent.name,
                attempt=attempt,
                delay_s=round(delay, 2),
                error=repr(exc),
            )
            await asyncio.sleep(delay)
        else:
            await _record(agent, started, result=result)
            return result.final_output
