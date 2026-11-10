"""CardWorkflow: извлечение → (генерация → проверка) × до 3 кругов → ожидание подтверждения."""
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

# код процесса исполняется в песочнице: обычные импорты ограничены
with workflow.unsafe.imports_passed_through():
    from app.temporal.activities import (
        critique_activity,
        extract_activity,
        generate_activity,
        set_status_activity,
    )

MAX_ATTEMPTS = 3

# повтор ШАГА: если воркер умер или шаг упал; повтор отдельного вызова модели — в llm/client.py
_STEP_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=2),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(seconds=30),
    maximum_attempts=3,
)
_LLM_STEP_TIMEOUT = timedelta(minutes=10)  # reasoning-модель отвечает минутами
_LLM_HEARTBEAT_TIMEOUT = timedelta(seconds=30)  # воркер умер — шаг уйдёт на другой за ~30 с
_DB_STEP_TIMEOUT = timedelta(seconds=30)


@workflow.defn
class CardWorkflow:
    def __init__(self) -> None:
        self._status = "pending"
        self._attempts = 0
        self._decision: str | None = None
        self._reason: str | None = None

    @workflow.signal
    def approve(self) -> None:
        self._decision = "approve"

    @workflow.signal
    def reject(self, reason: str = "") -> None:
        self._decision = "reject"
        self._reason = reason or None

    @workflow.query
    def state(self) -> dict:
        return {"status": self._status, "attempts": self._attempts}

    async def _set_status(
        self, status: str, *, result: dict | None = None, error: str | None = None
    ) -> None:
        self._status = status
        await workflow.execute_activity(
            set_status_activity,
            {
                "job_id": self._job_id,
                "status": status,
                "attempts": self._attempts,
                "result": result,
                "error": error,
            },
            start_to_close_timeout=_DB_STEP_TIMEOUT,
            retry_policy=_STEP_RETRY,
        )

    async def _llm_step(self, activity_fn, arg):
        return await workflow.execute_activity(
            activity_fn,
            arg,
            start_to_close_timeout=_LLM_STEP_TIMEOUT,
            heartbeat_timeout=_LLM_HEARTBEAT_TIMEOUT,
            retry_policy=_STEP_RETRY,
        )

    @workflow.run
    async def run(self, job_id: str, supplier_text: str) -> dict:
        self._job_id = job_id
        try:
            await self._set_status("extracting")
            facts = await self._llm_step(extract_activity, supplier_text)

            feedback: list[str] = []
            draft: dict = {}
            critic_approved = False
            while self._attempts < MAX_ATTEMPTS:
                self._attempts += 1
                await self._set_status("generating")
                draft = await self._llm_step(
                    generate_activity, {"facts": facts, "feedback": feedback}
                )
                await self._set_status("reviewing")
                report = await self._llm_step(
                    critique_activity, {"facts": facts, "draft": draft}
                )
                if report["verdict"] == "approve":
                    critic_approved = True
                    break
                feedback = report["issues"]

            note = None
            if not critic_approved:
                note = "Критик не одобрил черновик за %d круга: %s" % (
                    self._attempts,
                    "; ".join(feedback),
                )
            # долгая пауза: ожидание сигнала не занимает ни потока, ни соединения
            await self._set_status("awaiting_confirmation", result=draft, error=note)
            await workflow.wait_condition(lambda: self._decision is not None)

            if self._decision == "approve":
                await self._set_status("approved")
            else:
                await self._set_status("rejected", error=self._reason)
            return {"status": self._status, "attempts": self._attempts}
        except Exception as exc:
            await self._set_status("failed", error=str(exc)[:500])
            raise
