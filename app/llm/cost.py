"""Стоимость вызова из расхода токенов. Деньги — Decimal, не float."""
from decimal import ROUND_HALF_UP, Decimal

from app.llm.policy import Tier, price_for

_MILLION = Decimal(1_000_000)
_QUANT = Decimal("0.000001")


def call_cost(
    tier: Tier,
    input_tokens: int,
    output_tokens: int,
    prices: tuple[Decimal, Decimal] | None = None,
) -> Decimal:
    """Стоимость в ₽: токены × цена за миллион, с точностью до миллионной доли."""
    price_in, price_out = prices or price_for(tier)
    total = (Decimal(input_tokens) * price_in + Decimal(output_tokens) * price_out) / _MILLION
    return total.quantize(_QUANT, rounding=ROUND_HALF_UP)
