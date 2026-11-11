"""Структурированный ответ модели: разбор, валидация, лечение, точечная переделка."""
import json
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.agents.prompts import field_fixer_agent, generator_agent
from app.core.config import settings
from app.core.logging import get_logger
from app.llm.client import run_agent
from app.schemas import CardDraft
from app.services.json_utils import LLMJsonError, parse_llm_json

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

# если сломалось больше полей, точечно не чиним: модель переписывает ответ целиком
_MAX_POINT_FIELDS = 2


class StructuredOutputError(Exception):
    """Ответ модели не удалось привести к контракту за отведённое число попыток."""


def _as_text(out) -> str:
    if isinstance(out, str):
        return out
    if isinstance(out, BaseModel):
        return out.model_dump_json()
    return str(out)


def _describe(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(map(str, err['loc'])) or 'карточка'}: {err['msg']}" for err in exc.errors()
    )


def _failed_fields(exc: ValidationError) -> list[str] | None:
    """Верхнеуровневые поля с ошибкой. None — ошибка не про одно поле (общее правило)."""
    fields: list[str] = []
    for err in exc.errors():
        if not err["loc"]:
            return None
        name = str(err["loc"][0])
        if name not in fields:
            fields.append(name)
    return fields


async def fix_field(data: dict, field: str, problem: str) -> object:
    """Точечная переделка: модель получает текущий черновик и правит только одно поле."""
    prompt = (
        "Текущий черновик:\n"
        + json.dumps(data, ensure_ascii=False, indent=2)
        + f"\n\nПоле «{field}» сейчас: {json.dumps(data.get(field), ensure_ascii=False)}\n"
        f"Оно не прошло проверку: {problem}\n"
        f"Исправь ТОЛЬКО это поле, остальное не трогай. "
        f"Новое значение должно заметно отличаться от текущего и удовлетворять проверке. "
        f'Верни JSON вида {{"{field}": <новое значение>}} без пояснений.'
    )
    log.info("field_repair", field=field, problem=problem)
    out = await run_agent(field_fixer_agent(), prompt)
    fixed = parse_llm_json(_as_text(out))
    if field not in fixed:
        raise LLMJsonError(f"в ответе нет поля {field!r}")
    return fixed[field]


async def call_structured(
    agent,
    prompt: str,
    model_cls: type[T],
    *,
    max_repairs: int | None = None,
    field_repair: bool = False,
) -> T:
    """Вызывает агента и приводит ответ к model_cls.

    - не разобралось / не прошло валидацию → ответ уходит обратно в модель с текстом
      ошибки; попыток не больше max_repairs;
    - при field_repair и поломке одного-двух полей переделываются только они.
    """
    max_repairs = settings.structured_max_repairs if max_repairs is None else max_repairs
    out = await run_agent(agent, prompt)
    unchanged: set[str] = set()  # поля, которые правщик вернул без изменений

    for repair in range(max_repairs + 1):
        if isinstance(out, model_cls):
            return out
        raw = _as_text(out)
        data: dict | None = None
        fields: list[str] | None = None
        try:
            data = parse_llm_json(raw)
            return model_cls.model_validate(data)
        except LLMJsonError as exc:
            problem = str(exc)
        except ValidationError as exc:
            problem = _describe(exc)
            fields = _failed_fields(exc)

        log.warning("structured_invalid", model=model_cls.__name__, repair=repair, problem=problem)
        if repair == max_repairs:
            raise StructuredOutputError(f"{model_cls.__name__}: {problem}")

        if field_repair and data is not None and fields and len(fields) <= _MAX_POINT_FIELDS:
            try:
                for name in fields:
                    hint = problem
                    if name in unchanged:
                        hint += " Прошлая правка вернула то же значение — измени его."
                    fixed = await fix_field(data, name, hint)
                    if fixed == data.get(name):
                        unchanged.add(name)
                    data[name] = fixed
            except LLMJsonError as exc:
                raise StructuredOutputError(f"{model_cls.__name__}: {exc}") from exc
            out = json.dumps(data, ensure_ascii=False)
            continue

        out = await run_agent(
            agent,
            f"{prompt}\n\nТвой предыдущий ответ:\n{raw}\n\n"
            f"Он не прошёл проверку: {problem}\n"
            "Верни исправленный JSON целиком, без пояснений.",
        )
    raise StructuredOutputError(model_cls.__name__)  # недостижимо: цикл выходит раньше


class CardValidationError(StructuredOutputError):
    """Карточку не удалось привести к контракту CardDraft за отведённое число попыток."""


def parse_model(out, model_cls: type[T]) -> T:
    """Ответ агента (объект или текст с JSON) → модель; иначе LLMJsonError/ValidationError."""
    if isinstance(out, model_cls):
        return out
    return model_cls.model_validate(parse_llm_json(_as_text(out)))


async def validate_or_retry(prompt: str, *, max_fixes: int | None = None) -> CardDraft:
    """Генерация карточки с лечением: невалидный ответ уходит обратно в модель с ошибкой.

    Не разобралось или не прошло валидацию — модель получает текст ошибки; если сломалось
    одно-два поля, переделываются только они. Попыток не больше 1 + max_fixes, иначе
    CardValidationError.
    """
    try:
        return await call_structured(
            generator_agent(), prompt, CardDraft, max_repairs=max_fixes, field_repair=True
        )
    except StructuredOutputError as exc:
        raise CardValidationError(str(exc)) from exc


async def regenerate_field(card: CardDraft, field: str, problem: str) -> CardDraft:
    """Точечная переделка: модель правит одно поле, остальные поля карточки не трогаются."""
    prompt = (
        "Текущий черновик:\n"
        + card.model_dump_json(indent=2)
        + f"\n\nПоле «{field}» не прошло проверку: {problem}\n"
        f"Исправь ТОЛЬКО поле {field}. "
        f'Верни JSON вида {{"{field}": <новое значение>}} без пояснений.'
    )
    log.info("regenerate_field", field=field, problem=problem)
    out = await run_agent(field_fixer_agent(), prompt)
    fixed = parse_llm_json(_as_text(out))
    if field not in fixed:
        raise CardValidationError(f"в ответе нет поля {field!r}")
    try:
        return CardDraft.model_validate({**card.model_dump(), field: fixed[field]})
    except ValidationError as exc:
        raise CardValidationError(_describe(exc)) from exc


def confidence_gate(card: CardDraft) -> str:
    """Ниже порога уверенности карточка не «готова»: её смотрит человек."""
    return "needs_review" if card.confidence < settings.confidence_threshold else "done"
