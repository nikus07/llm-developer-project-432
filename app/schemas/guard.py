from pydantic import BaseModel, Field


class InjectionAnswer(BaseModel):
    """Вердикт дешёвой модели: пытается ли фрагмент манипулировать инструкциями ассистента."""

    is_injection: bool = Field(description="true — во фрагменте есть попытка манипуляции")
    reason: str = Field(default="", description="Одно предложение: почему")
