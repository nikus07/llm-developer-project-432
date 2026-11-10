"""Формат ответа агента: схема уходит серверу, а проверка остаётся за нами."""
from typing import Any

from agents import AgentOutputSchemaBase
from pydantic import BaseModel


class RawJsonOutput(AgentOutputSchemaBase):
    """Передаёт модели JSON-схему результата, но не разбирает ответ внутри SDK.

    Если SDK сам валидирует ответ и тот невалиден, он бросает исключение и текст
    ответа пропадает — лечить нечего. Здесь SDK отдаёт строку как есть, а разбор и
    валидацию с лечением делает services/structured.py.
    """

    def __init__(self, model: type[BaseModel]) -> None:
        self._model = model

    def is_plain_text(self) -> bool:
        return False

    def name(self) -> str:
        return self._model.__name__

    def json_schema(self) -> dict[str, Any]:
        return self._model.model_json_schema()

    def is_strict_json_schema(self) -> bool:
        return False

    def validate_json(self, json_str: str) -> Any:
        return json_str
