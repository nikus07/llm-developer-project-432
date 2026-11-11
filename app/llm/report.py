"""Отчёт по таблице вызовов: разбивка по моделям и стоимость задачи.

python -m app.llm.report            — разбивка по моделям
python -m app.llm.report --job ID   — вызовы и стоимость одной задачи
"""
import argparse

from app.core.db import close_pool, init_pool
from app.repositories import llm_calls as llm_calls_repo


def _money(value) -> str:
    return f"{value:.6f}"


def print_breakdown() -> None:
    rows = llm_calls_repo.breakdown_by_model()
    if not rows:
        print("Вызовов модели пока нет")
        return
    print(f"{'модель':<20}{'тариф':<8}{'вызовов':>8}{'токенов вх.':>13}{'токенов исх.':>14}"
          f"{'стоимость, ₽':>15}{'сред. мс':>10}")
    total_calls = 0
    total_cost = 0
    for r in rows:
        print(f"{r['model']:<20}{r['tier']:<8}{r['calls']:>8}{r['input_tokens']:>13}"
              f"{r['output_tokens']:>14}{_money(r['cost']):>15}{r['avg_ms']:>10}")
        total_calls += r["calls"]
        total_cost += r["cost"]
    print(f"{'итого':<28}{total_calls:>8}{'':>27}{_money(total_cost):>15}")


def print_job(job_id: str) -> None:
    total = llm_calls_repo.cost_for_job(job_id)
    print(f"задача {job_id}: вызовов {total['calls']}, токенов {total['input_tokens']} вх. / "
          f"{total['output_tokens']} исх., стоимость {_money(total['cost'])} ₽, "
          f"в модели {total['duration_ms'] / 1000:.1f} с")
    for call in llm_calls_repo.calls_for_job(job_id):
        print(f"  {call['created_at']:%H:%M:%S}  {call['agent']:<14}{call['tier']:<7}"
              f"{call['input_tokens']:>6} / {call['output_tokens']:<6}{_money(call['cost']):>12} ₽"
              f"{call['duration_ms']:>9} мс  {call['status']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Стоимость вызовов модели")
    parser.add_argument("--job", help="идентификатор задачи")
    args = parser.parse_args()
    init_pool()
    try:
        print_job(args.job) if args.job else print_breakdown()
    finally:
        close_pool()


if __name__ == "__main__":
    main()
