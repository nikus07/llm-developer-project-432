"""Калибровка порога релевантности векторного поиска на своих документах.

Запуск: python -m evals.calibrate_threshold
Для запросов с известным ответом печатает близость верного фрагмента, а для запросов
«мимо» (в документах ответа нет) — близость лучшего найденного фрагмента. Порог должен
лежать между худшим верным и лучшим мусорным результатом.
"""
from app.core.db import close_pool, connection, init_pool
from app.repositories import chunks as chunks_repo

# (запрос, подстрока, которая должна быть в верном фрагменте)
RELEVANT = [
    ("Какая мощность у блендера?", "Мощность: 800 Вт"),
    ("мотор 800 Вт", "Мощность: 800 Вт"),
    ("из какого материала сделана ножка", "Материал ножки"),
    ("сколько действует гарантия на блендер", "Гарантия 24 месяца"),
    ("объём стакана блендера", "Объём мерного стакана"),
    ("какая мощность у электрического чайника", "Мощность: 2200 Вт"),
    ("вес чайника", "Вес: 0,9 кг"),
    ("сколько стоит блендер оптом", "цена"),
    ("минимальная партия для заказа", "Минимальная партия"),
    ("характеристики чайников КеттлПро", "KTL-1700"),
    ("кому звонить по вопросам оптовых поставок", "менеджер"),
]
# в документах поставщиков ответа на эти запросы нет
IRRELEVANT = [
    "рецепт борща с говядиной",
    "курс доллара на завтра",
    "как выучить английский язык",
    "погода в Санкт-Петербурге",
    "расписание электричек Москва Тверь",
    "как заменить колесо на велосипеде",
]


def top_hit(query: str) -> tuple[float, str]:
    with connection() as conn:
        hits = chunks_repo.search_vector(conn, query, top_k=1, threshold=0.0)
    return (hits[0]["score"], hits[0]["text"]) if hits else (0.0, "")


def best_score_for(query: str, needle: str) -> float:
    """Близость самого близкого фрагмента, в тексте которого есть нужная подстрока."""
    with connection() as conn:
        hits = chunks_repo.search_vector(conn, query, top_k=50, threshold=0.0)
    scores = [h["score"] for h in hits if needle.lower() in h["text"].lower()]
    return max(scores, default=0.0)


def main() -> None:
    init_pool()
    try:
        with connection() as conn:
            total = conn.execute("SELECT count(*) FROM chunks WHERE embedding IS NOT NULL")
            print(f"фрагментов с векторами: {total.fetchone()[0]}\n")
        print("ВЕРНЫЕ (близость фрагмента с ответом):")
        relevant = []
        for query, needle in RELEVANT:
            score = best_score_for(query, needle)
            relevant.append(score)
            print(f"  {score:.3f}  {query}")
        print("\nМИМО (близость лучшего найденного фрагмента):")
        junk = []
        for query in IRRELEVANT:
            score, _ = top_hit(query)
            junk.append(score)
            print(f"  {score:.3f}  {query}")
        print(f"\nхудший верный:   {min(relevant):.3f}")
        print(f"лучший мусорный: {max(junk):.3f}")
    finally:
        close_pool()


if __name__ == "__main__":
    main()
