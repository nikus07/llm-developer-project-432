import asyncio
from decimal import Decimal

import pytest
import structlog

from app.core.config import settings
from app.llm import client as llm_client
from app.llm.cost import call_cost
from app.llm.policy import model_for, tier_for


def test_cost_on_given_usage():
    # 1000 токенов на входе по 15 ₽/млн и 500 на выходе по 60 ₽/млн
    cost = call_cost("main", 1000, 500, prices=(Decimal("15"), Decimal("60")))
    assert cost == Decimal("0.045000")
    assert isinstance(cost, Decimal)


def test_cost_is_exact_decimal_not_float():
    # 0.1 + 0.2 != 0.3 для float; для денег накопленная ошибка недопустима
    one = call_cost("main", 1, 0, prices=(Decimal("0.1"), Decimal("0")))
    assert one == Decimal("0.000000")  # 1 токен по 0.1 ₽/млн — меньше миллионной доли
    big = call_cost("main", 3_000_000, 0, prices=(Decimal("0.1"), Decimal("0")))
    assert big == Decimal("0.300000")


def test_cost_rounds_half_up_to_micro_rubles():
    assert call_cost("main", 5, 0, prices=(Decimal("0.1"), Decimal("0"))) == Decimal("0.000001")
    assert call_cost("main", 4, 0, prices=(Decimal("0.1"), Decimal("0"))) == Decimal("0.000000")


def test_zero_usage_costs_nothing():
    assert call_cost("light", 0, 0) == Decimal("0.000000")


def test_light_tier_is_cheaper_by_settings():
    main = call_cost("main", 10_000, 2_000)
    light = call_cost("light", 10_000, 2_000)
    assert light < main
    assert main == Decimal("0.270000")  # 10000*15 + 2000*60 = 270000 / 1e6
    assert light == Decimal("0.054000")  # 10000*3 + 2000*12 = 54000 / 1e6


def test_service_roles_use_light_tier_and_generation_uses_main():
    for name in ("Critic", "RagCritic", "FieldFixer", "Judge"):
        assert tier_for(name) == "light"
    for name in ("Extractor", "Generator", "RagGenerator"):
        assert tier_for(name) == "main"


def test_light_model_falls_back_to_main_when_not_set(monkeypatch):
    monkeypatch.setattr(settings, "llm_model", "big")
    monkeypatch.setattr(settings, "llm_model_light", None)
    assert model_for("light") == "big"
    monkeypatch.setattr(settings, "llm_model_light", "small")
    assert model_for("light") == "small"
    assert model_for("main") == "big"


# --- клиент пишет учёт сам ---


class _Usage:
    input_tokens = 1000
    output_tokens = 500


class _Result:
    final_output = "ok"

    class context_wrapper:  # имитация RunResult SDK
        usage = _Usage()


def test_client_records_every_call_with_usage_job_and_duration(monkeypatch):
    recorded: list[dict] = []

    class FakeRunner:
        @staticmethod
        async def run(agent, prompt):
            return _Result()

    class FakeAgent:
        name = "RagCritic"
        model = "qwen3-14b"

    monkeypatch.setattr(llm_client, "setup_llm", lambda: None)
    monkeypatch.setattr(llm_client, "Runner", FakeRunner)
    monkeypatch.setattr(llm_client.llm_calls_repo, "insert", lambda **kw: recorded.append(kw))

    async def scenario():
        structlog.contextvars.bind_contextvars(job_id="job-1", request_id="req-1")
        try:
            return await llm_client.run_agent(FakeAgent(), "x")
        finally:
            structlog.contextvars.clear_contextvars()

    assert asyncio.run(scenario()) == "ok"

    (call,) = recorded
    assert call["job_id"] == "job-1" and call["request_id"] == "req-1"
    assert call["agent"] == "RagCritic" and call["model"] == "qwen3-14b"
    assert call["tier"] == "light"  # служебная роль — дешёвый тариф
    assert (call["input_tokens"], call["output_tokens"]) == (1000, 500)
    assert call["cost"] == call_cost("light", 1000, 500)
    assert call["duration_ms"] >= 0 and call["status"] == "ok"


def test_failed_attempt_is_recorded_as_error_without_cost(monkeypatch):
    recorded: list[dict] = []

    class FakeRunner:
        @staticmethod
        async def run(agent, prompt):
            raise ValueError("плохой запрос")

    class FakeAgent:
        name = "Extractor"

    monkeypatch.setattr(llm_client, "setup_llm", lambda: None)
    monkeypatch.setattr(llm_client, "Runner", FakeRunner)
    monkeypatch.setattr(llm_client.llm_calls_repo, "insert", lambda **kw: recorded.append(kw))

    with pytest.raises(ValueError):
        asyncio.run(llm_client.run_agent(FakeAgent(), "x"))

    (call,) = recorded
    assert call["status"] == "error" and call["cost"] == Decimal("0.000000")
    assert call["tier"] == "main"


def test_accounting_failure_does_not_break_the_call(monkeypatch):
    class FakeRunner:
        @staticmethod
        async def run(agent, prompt):
            return _Result()

    class FakeAgent:
        name = "Extractor"

    def broken(**kwargs):
        raise RuntimeError("база недоступна")

    monkeypatch.setattr(llm_client, "setup_llm", lambda: None)
    monkeypatch.setattr(llm_client, "Runner", FakeRunner)
    monkeypatch.setattr(llm_client.llm_calls_repo, "insert", broken)

    assert asyncio.run(llm_client.run_agent(FakeAgent(), "x")) == "ok"
