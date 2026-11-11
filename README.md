# ИИ-генератор карточки товара

[![hexlet-check](https://github.com/nikus07/llm-developer-project-432/actions/workflows/hexlet-check.yml/badge.svg)](https://github.com/nikus07/llm-developer-project-432/actions)

Бэкенд-сервис: принимает документы поставщиков (pdf, docx, xlsx), строит по ним поисковый
индекс и генерирует **черновик** карточки товара — с источником у каждой характеристики,
списком недостающих полей и уровнем уверенности. Сервис не обещает, что карточка готова к
публикации: он обещает, что человек проверит её за минуту, а не соберёт за час.

Работает полностью локально: модель — в LM Studio, эмбеддинги — на процессоре, векторы — в
PostgreSQL (pgvector). Ни внешних API, ни платных токенов.

```
документ ─► разбор ─► фрагменты ─► эмбеддинги ─► PostgreSQL (pgvector + полнотекстовый индекс)
                                                          │
 POST /generate ─► поиск ─► маскирование ПД ─► детектор инъекций ─► генерация ─► проверка ссылок
                    (гибрид)                    (правила + модель)     │         и критик
                                                                       ▼
                                         ожидание человека (сигнал approve / reject)
```

Долгая работа идёт не в HTTP-запросе, а в процессах [Temporal](https://temporal.io):
перезапуск воркера задачу не теряет, а статус для клиента живёт в базе.

## Требования

| | |
|---|---|
| uv | [astral.sh/uv](https://docs.astral.sh/uv/): ставит Python 3.12 и зависимости из `uv.lock` |
| Docker | с Compose v2 (PostgreSQL + pgvector и Temporal) |
| LM Studio | с загруженной чат-моделью (проверялось на `qwen3-14b`), сервер на `:1234` |
| Место и память | ~1,2 ГБ под модель эмбеддингов (скачивается один раз), ~1 ГБ под torch |

Порты должны быть свободны: `8000` (API), `5432` (PostgreSQL), `7233` и `8233` (Temporal),
`1234` (LM Studio). Если `5432` занят локальным PostgreSQL — см. «Если что-то не работает».

## Запуск с нуля

Команды одинаковы для bash и PowerShell. Утилита `make` необязательна — рядом указаны
прямые команды; `uv run` сам берёт окружение проекта, активировать его не нужно.

**1. Окружение**

```bash
git clone https://github.com/nikus07/llm-developer-project-432.git
cd llm-developer-project-432
uv sync --frozen            # зависимости; `make setup` делает это и применяет миграции
```

Зависимости ставятся строго из `uv.lock`, одной командой. На Linux torch берётся
CPU-сборкой с pytorch.org (~200 МБ вместо ~2 ГБ с CUDA): это настроено в `pyproject.toml`.

**2. Конфигурация**

```bash
cp .env.example .env
curl http://localhost:1234/v1/models     # возьмите оттуда "id" чат-модели
```

Впишите этот `id` в `LLM_MODEL` в `.env`. Остальные значения подходят как есть.

**3. Инфраструктура** — PostgreSQL с pgvector и Temporal в Compose

```bash
docker compose up -d        # make infra
docker compose ps           # postgres и temporal должны стать healthy (до минуты)
```

Веб-интерфейс Temporal: http://localhost:8233 — история каждого процесса.

**4. Миграции**

```bash
uv run python -m app.core.migrate  # make migrate
```

В первый раз пишет «Миграция применена: …» по каждой из шести, во второй: «Новых миграций нет».

**5. Приложение и воркер процессов** — каждый в своём терминале

```bash
uv run uvicorn app.main:app --port 8000   # make api
uv run python -m app.temporal_worker      # make temporal-worker
```

Проверка: `curl localhost:8000/health/ready` → `{"status":"ok","database":"up","vector":"0.8.x"}`.
Первый запрос поиска или генерации скачивает модель эмбеддингов (~1,1 ГБ, один раз) и
загружает её в память воркера — это около минуты.

**6. Документы и карточки**

```bash
uv run python scripts/demo.py
```

Скрипт загружает все файлы из `data/`, ждёт, пока они проиндексируются, и по каждому
генерирует карточку: печатает статус, заголовок, число характеристик и ссылок, недостающие
поля, уверенность, что замаскировано и исключено, и стоимость. Результаты сохраняются в
`evals/reports/cards-*.json`. Локальной модели нужно несколько минут на документ.

Карточка, которую ждёт человек (`awaiting_confirmation` или `needs_review`), подтверждается
или отклоняется сигналом:

```bash
curl -X POST localhost:8000/workflows/<job_id>/approve
curl -X POST localhost:8000/workflows/<job_id>/reject
```

## API

| Метод и путь | Что делает |
|---|---|
| `POST /documents` | Загрузка файла (pdf, docx, xlsx; до 10 МБ). 202 и `id`; тот же файл не создаёт дубль |
| `GET /documents/{id}` | Состояние (`new → parsing → parsed → indexed` или `failed` с причиной) и число фрагментов |
| `POST /generate` | `{document_ids, hint?}` → 202 и `id` задачи; заголовок `Idempotency-Key` защищает от дублей |
| `GET /jobs/{id}` | Статус, карточка с источниками (`result`), примечание (`error`) и отчёт защиты (`report`) |
| `GET /jobs/{id}/cost` | Токены и стоимость задачи одним запросом |
| `GET /workflows/{id}` · `POST /workflows/{id}/approve` · `…/reject` | Состояние процесса и решение человека |
| `GET /search?q=&mode=vector\|keyword\|hybrid` | Поиск по фрагментам с фильтрами `document`, `article`, `brand`, `page` |
| `GET /health/live` · `GET /health/ready` | Живость процесса / готовность (база и расширение `vector`) |
| `POST /cards` · `POST /jobs` | Прямые варианты по тексту, без поиска и защиты (учебные, из практики темы) |

Каждый ответ несёт `X-Request-ID`; тот же идентификатор и `job_id` попадают в логи.

**Состояния задачи:** `queued → parsing → indexing → retrieving → screening → generating →
verifying → awaiting_confirmation | needs_review → approved | rejected | failed`.
`needs_review` — карточку или документ должен посмотреть человек: низкая уверенность,
ссылки не прошли проверку, фильтр выхода что-то нашёл или документ уведён из-за инъекций.

## Команды

| | |
|---|---|
| `make setup` | поставить зависимости из `uv.lock` и применить миграции (база должна быть поднята: `make infra`) |
| `make infra` | поднять PostgreSQL и Temporal |
| `make migrate` | применить миграции из `db/migrations/` |
| `make api` · `make temporal-worker` | приложение и воркер процессов |
| `make reindex` | досчитать векторы фрагментам, у которых их нет (повтор ничего не делает) |
| `make search Q="мощность блендера"` | поиск из командной строки |
| `make costs` · `make costs JOB=<id>` | разбивка по моделям / вызовы и стоимость задачи |
| `make eval` · `make eval ALL=1` | метрики качества на эталоне: подмножество / весь набор |
| `make test` · `make lint` | тесты и линтер |

## Тесты

```bash
uv run pytest         # make test
uv run ruff check .   # make lint
```

Тесты не загружают модель эмбеддингов и не обращаются к LM Studio — модель подменена. Часть
тестов (репозитории, сквозные сценарии процесса) идёт на живых PostgreSQL и Temporal из
Compose; без них такие тесты пропускаются, а не падают.

## Структура

```
app/
  main.py                 сборка FastAPI, middleware с X-Request-ID
  core/                   настройки, пул соединений, миграции, логи, трассировка
  routers/                health, cards, jobs, documents, generate, search, workflows
  services/               pipeline (стадии), structured (разбор и лечение ответа модели),
                          rag_pipeline (поиск → контекст → карточка → проверка ссылок), documents
  agents/prompts.py       роли модели: промпт как код
  llm/                    client (таймаут, повторы, учёт), cost, policy (две модели), report
  parsers/                pdf, docx, xlsx, нормализатор, резка на фрагменты
  rag/                    embedder, retrieval, context, reindex
  guardrails/             pii, injection, output — три рубежа защиты
  repositories/           весь SQL: jobs, documents, chunks, llm_calls
  schemas/                контракты (Pydantic)
  temporal/               шаги, процессы, перехватчик трассировки
db/migrations/            SQL-миграции, применяются раннером app/core/migrate.py
data/                     выданные документы поставщиков и эталон golden_cards.json
evals/                    метрики качества, калибровка порога, отчёты
scripts/demo.py           сквозной прогон по выданным документам
docs/                     decisions.md — архитектурные решения, REPORT.md — отчёт, DEFENSE.md — защита
tests/
```

## Отчёт и решения

- **[docs/REPORT.md](docs/REPORT.md)** — метрики последнего прогона, стоимость карточки,
  самая слабая метрика и почему она слабее остальных.
- **[docs/decisions.md](docs/decisions.md)** — архитектурные решения и их цена: слои, миграции,
  эмбеддинги и порог, гибридный поиск, проверка ссылок, учёт стоимости, защита.
- **[docs/DEFENSE.md](docs/DEFENSE.md)** — ответы на вопросы защиты, со ссылками на код.

## Если что-то не работает

| Симптом | Причина и что делать |
|---|---|
| `password authentication failed for user "card"` | Порт `5432` занят локальным PostgreSQL. В `.env`: `PG_PORT=5433` и порт `5433` в `DATABASE_URL`, затем `docker compose up -d` |
| `/health/ready` отвечает 503 | Миграции не применены или контейнер `postgres` ещё не healthy: `docker compose ps`, затем `uv run python -m app.core.migrate` |
| Задача долго в `retrieving` | Первый раз воркер скачивает и загружает модель эмбеддингов (~1,1 ГБ) |
| Ошибка «unknown model» от модели | `LLM_MODEL` не совпадает с `id` из `curl localhost:1234/v1/models` |
| `404` от модели при генерации | LM Studio не запущен или не включён сервер на `:1234` |
| Задача зависла после перезапуска воркера | Это нормально: поднимите воркер снова, шаг продолжится сам (≈30 с на обнаружение) |
| Кириллица в теле запроса `curl` из Git Bash ломается | Используйте `scripts/demo.py` или `curl` из PowerShell/WSL |

Полный сброс состояния (тома, база, история процессов):

```bash
docker compose down -v && rm -rf uploads
```

---

<details>
<summary>Автоматические тесты Хекслета</summary>

Тесты запускаются на каждый коммит. За запуск отвечает файл `.github/workflows/hexlet-check.yml` — не удаляйте и не переименовывайте ни его, ни репозиторий.

</details>

## О Хекслете

[Хекслет](https://ru.hexlet.io/) — школа программирования: авторские программы обучения с практикой, поддержкой наставников и реальными проектами, которые остаются в резюме. Этот репозиторий — один из таких проектов.
