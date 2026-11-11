"""CardWorkflow: извлечение → (генерация → проверка) × до 3 кругов → ожидание подтверждения."""
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError, ApplicationError

# код процесса исполняется в песочнице: обычные импорты ограничены
with workflow.unsafe.imports_passed_through():
    from app.temporal.activities import (
        critique_activity,
        documents_state_activity,
        extract_activity,
        generate_activity,
        index_document_activity,
        mark_document_failed_activity,
        process_document_activity,
        rag_card_activity,
        retrieve_activity,
        set_status_activity,
    )

MAX_ATTEMPTS = 3
_DOCUMENTS_WAIT = timedelta(minutes=15)
_DOCUMENTS_POLL = timedelta(seconds=2)
_RETRIEVE_TIMEOUT = timedelta(minutes=10)  # первая загрузка модели эмбеддингов

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
    async def run(
        self, job_id: str, supplier_text: str, confidence_threshold: float = 0.6
    ) -> dict:
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

            notes: list[str] = []
            if not critic_approved:
                notes.append(
                    "Критик не одобрил черновик за %d круга: %s"
                    % (self._attempts, "; ".join(feedback))
                )
            confidence = float(draft.get("confidence", 0.0))
            if confidence < confidence_threshold:
                notes.append(
                    f"Уверенность {confidence:.2f} ниже порога {confidence_threshold:.2f}"
                )
            # низкая уверенность — не «готово»: карточку ждёт человек
            needs_review = bool(notes)
            note = "; ".join(notes) or None
            # долгая пауза: ожидание сигнала не занимает ни потока, ни соединения
            await self._set_status(
                "needs_review" if needs_review else "awaiting_confirmation",
                result=draft,
                error=note,
            )
            await workflow.wait_condition(lambda: self._decision is not None)

            if self._decision == "approve":
                await self._set_status("approved")
            else:
                await self._set_status("rejected", error=self._reason)
            return {"status": self._status, "attempts": self._attempts}
        except Exception as exc:
            await self._set_status("failed", error=str(exc)[:500])
            raise


_DOCUMENT_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=2),
    backoff_coefficient=2.0,
    maximum_attempts=3,
    non_retryable_error_types=["LookupError"],
)


@workflow.defn
class DocumentWorkflow:
    """Разбор документа: файл → фрагменты в базе. Состояние пишет сам шаг разбора."""

    @workflow.run
    async def run(self, document_id: str) -> dict:
        try:
            parsed = await workflow.execute_activity(
                process_document_activity,
                document_id,
                start_to_close_timeout=timedelta(minutes=5),
                retry_policy=_DOCUMENT_RETRY,
            )
            if parsed["status"] != "parsed":
                return parsed  # отказ с причиной: индексировать нечего
            indexed = await workflow.execute_activity(
                index_document_activity,
                document_id,
                start_to_close_timeout=timedelta(minutes=15),  # первая загрузка модели
                retry_policy=_DOCUMENT_RETRY,
            )
            return {**parsed, **indexed}
        except ActivityError as exc:
            reason = str(exc.cause or exc)[:300]
            await workflow.execute_activity(
                mark_document_failed_activity,
                {"document_id": document_id, "reason": f"ошибка разбора: {reason}"},
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=_STEP_RETRY,
            )
            raise


@workflow.defn
class GenerateWorkflow:
    """Документы → поиск → генерация со ссылками → проверка → ожидание человека.

    Состояния задачи: queued → parsing → indexing → retrieving → generating → verifying →
    awaiting_confirmation | needs_review → approved | rejected | failed.
    """

    def __init__(self) -> None:
        self._status = "queued"
        self._attempts = 0
        self._decision: str | None = None
        self._reason: str | None = None
        self._job_id = ""

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
        self,
        status: str,
        *,
        result: dict | None = None,
        error: str | None = None,
        report: dict | None = None,
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
                "report": report,
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

    async def _db_step(self, activity_fn, arg, timeout: timedelta = _DB_STEP_TIMEOUT):
        return await workflow.execute_activity(
            activity_fn, arg, start_to_close_timeout=timeout, retry_policy=_STEP_RETRY
        )

    async def _wait_for_documents(self, document_ids: list[str]) -> dict:
        """Документы разбирают их собственные процессы; здесь только ждём результата."""
        deadline = workflow.now() + _DOCUMENTS_WAIT
        stage = None
        while True:
            state = await self._db_step(documents_state_activity, document_ids)
            if not state["waiting"]:
                return state
            if state["stage"] != stage:
                stage = state["stage"]
                await self._set_status(stage)
            if workflow.now() > deadline:
                raise ApplicationError("документы не успели разобраться за отведённое время")
            await workflow.sleep(_DOCUMENTS_POLL)

    @workflow.run
    async def run(
        self,
        job_id: str,
        document_ids: list[str],
        hint: str | None = None,
        confidence_threshold: float = 0.6,
    ) -> dict:
        self._job_id = job_id
        try:
            await self._set_status("queued")
            docs = await self._wait_for_documents(document_ids)
            if not docs["ready"]:
                reasons = "; ".join(str(s["reason"]) for s in docs["skipped"])
                raise ApplicationError(f"нет ни одного готового документа: {reasons}")
            info = [f"пропущен документ {s['id']}: {s['reason']}" for s in docs["skipped"]]

            await self._set_status("retrieving")
            found = await self._llm_step(
                retrieve_activity, {"document_ids": docs["ready"], "hint": hint}
            )
            guard = found["guard"]
            if found["needs_review"] or not found["context"].strip():
                # много подозрительных фрагментов или нечего показать модели: к человеку
                reason = (
                    found["escalation_reason"] or "в документах не осталось безопасных фрагментов"
                )
                await self._set_status("needs_review", error=reason, report=guard)
                return {"status": "needs_review", "attempts": 0}

            await self._set_status("generating", report=guard)
            try:
                done = await self._llm_step(
                    rag_card_activity,
                    {
                        "query": found["query"],
                        "context": found["context"],
                        "guard": guard,
                        "excluded_texts": found["excluded_texts"],
                    },
                )
            except ActivityError as exc:
                if not isinstance(exc.cause, ApplicationError) or not exc.cause.non_retryable:
                    raise
                # выдуманные ссылки при всех попытках: карточку получает человек
                await self._set_status(
                    "needs_review", error="Ссылки не прошли проверку: " + str(exc.cause)[:400],
                    report=guard,
                )
                return {"status": "needs_review", "attempts": 0}
            self._attempts = done["attempts"]
            card, guard = done["card"], done["guard"]
            review: list[str] = []
            if done["verdict"] != "approved":
                review.append("Проверка не одобрила карточку")
            if guard["output_issues"]:
                review.append("Фильтр выхода: " + "; ".join(sorted(set(guard["output_issues"]))))
            if done["status"] == "needs_review" and not review:
                review.append(
                    f"Уверенность {float(card.get('confidence', 0.0)):.2f} "
                    f"ниже порога {confidence_threshold:.2f} или нужна проверка человеком"
                )
            notes = "; ".join([*review, *info]) or None
            await self._set_status(
                "needs_review" if review or done["status"] != "done" else "awaiting_confirmation",
                result=card,
                error=notes,
                report=guard,
            )
            await workflow.wait_condition(lambda: self._decision is not None)

            if self._decision == "approve":
                await self._set_status("approved")
            else:
                await self._set_status("rejected", error=self._reason)
            return {"status": self._status, "attempts": self._attempts}
        except Exception as exc:
            await self._set_status("failed", error=str(exc)[:500])
            raise
