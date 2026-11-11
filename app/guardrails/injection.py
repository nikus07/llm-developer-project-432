"""Детектор инъекций в два уровня: правила по тексту и узкий вопрос дешёвой модели.

Быстрый уровень бесплатный и ловит грубое. Точный платный и идёт только по тем фрагментам,
которые пометил первый. Если точный уровень недоступен, фрагмент считается подозрительным:
защита, которая при своём отказе открывает дверь, — не защита.
"""
import re
from dataclasses import dataclass, field

from app.agents.prompts import injection_detector_agent
from app.core.logging import get_logger
from app.schemas.guard import InjectionAnswer
from app.services.structured import call_structured

log = get_logger(__name__)

_I = re.IGNORECASE
_RULES: list[tuple[str, re.Pattern]] = [
    # прямые приказы
    ("приказ_игнорировать", re.compile(
        r"(?:игнорируй(?:те)?|забудь(?:те)?|отмени(?:те)?|не\s+учитывай(?:те)?)\s+(?:все\s+|всё\s+)?"
        r"(?:предыдущие|прошлые|прежние|вышеуказанные|системные|свои|любые)?\s*"
        r"(?:инструкци\w+|указани\w+|правил\w+|команд\w+)", _I)),
    ("приказ_игнорировать_en", re.compile(
        r"(?:ignore|disregard|forget|override)\s+(?:all\s+|any\s+)?(?:the\s+)?"
        r"(?:previous|prior|above|earlier|system|your)\s+(?:instructions?|prompts?|rules|directions)",
        _I)),
    # маркеры служебных ролей
    ("маркер_роли", re.compile(
        r"^\s*(?:system|assistant|developer|система|системное\s+сообщение)\s*[:>\]-]",
        _I | re.MULTILINE)),
    ("служебные_токены", re.compile(
        r"<\|(?:im_start|im_end|system|assistant|user)\|>|\[/?INST\]|<<SYS>>"
        r"|<\s*/?\s*(?:system|assistant)\s*>", _I)),
    ("смена_роли", re.compile(
        r"\b(?:ты\s+теперь|с\s+этого\s+момента\s+ты|отныне\s+ты|you\s+are\s+now|act\s+as)\b", _I)),
    # попытка вытащить промпт
    ("вытащить_промпт", re.compile(
        r"(?:покажи|выведи|повтори|раскрой|reveal|show|print|repeat)\s+(?:мне\s+)?"
        r"(?:свой\s+|твой\s+|your\s+|the\s+)?(?:системн\w+\s+)?(?:промпт|prompt|инструкци\w+|instructions?)",
        _I)),
    # подозрительно длинные закодированные вставки
    ("закодированная_вставка", re.compile(r"[A-Za-z0-9+/]{80,}={0,2}|(?:[0-9a-fA-F]{2}){40,}")),
    # требование вписать постороннее в результат
    ("подмена_результата", re.compile(
        r"\b(?:включи|добавь|укажи|впиши)(?:те)?\s+(?:\S+\s+){0,3}?(?:в\s+(?:описание|карточку|заголовок)|цену)",
        _I)),
]

_MAX_FRAGMENT_CHARS = 2500


def injection_rules(text: str) -> list[str]:
    """Имена сработавших правил: быстрый уровень, без обращения к модели."""
    return [name for name, pattern in _RULES if pattern.search(text)]


@dataclass
class InjectionVerdict:
    """Вердикт детектора: подозрителен ли фрагмент, почему и какие правила сработали."""

    suspicious: bool
    reason: str = ""
    rules: list[str] = field(default_factory=list)
    level: str = "regex"  # regex | model | fail_closed


def detect_injection_regex(text: str) -> InjectionVerdict:
    """Быстрый уровень: правила по тексту (приказы, маркеры ролей, закодированные вставки)."""
    rules = injection_rules(text)
    reason = "сработали правила: " + ", ".join(rules) if rules else ""
    return InjectionVerdict(bool(rules), reason, rules)


def _prompt(text: str) -> str:
    return "Фрагмент документа:\n<fragment>\n" + text[:_MAX_FRAGMENT_CHARS] + "\n</fragment>"


async def detect_injection_llm(text: str) -> InjectionVerdict:
    """Точный уровень: узкий вопрос дешёвой модели. Ошибку модели решает вызывающий.

    Платный уровень: зовётся только по фрагментам, которые пометил быстрый. Исключение
    (модель недоступна, ответ не разобрался) вызывающий трактует как «подозрителен».
    """
    answer = await call_structured(
        injection_detector_agent(), _prompt(text), InjectionAnswer, max_repairs=1
    )
    return InjectionVerdict(
        answer.is_injection, answer.reason[:300], injection_rules(text), "model"
    )
