"""Прогон метрик качества генерации на эталонном наборе.

python -m evals.run_metrics          — подмножество (eval_defaults из эталона)
python -m evals.run_metrics --all    — весь набор (локальная модель медленная)

По каждому документу: генерация по его фрагментам, три метрики, стоимость. Отчёт
сохраняется файлом в evals/reports/, чтобы «до» и «после» правки промпта было с чем сравнить.
"""
import argparse
import asyncio
import hashlib
import json
import time
import uuid
from datetime import datetime
from pathlib import Path

import structlog

from app.agents import prompts
from app.core.config import settings
from app.core.db import close_pool, connection, init_pool
from app.core.logging import setup_logging
from app.rag.context import context_chunk_ids
from app.repositories import chunks as chunks_repo
from app.repositories import documents as documents_repo
from app.repositories import llm_calls as llm_calls_repo
from app.services import documents as documents_service
from app.services import rag_pipeline
from evals.judge import judge_card
from evals.metrics import (
    characteristics_accuracy,
    citation_report,
    judge_score,
    missing_fields_scores,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
GOLDEN = DATA / "golden_cards.json"
REPORTS = Path(__file__).resolve().parent / "reports"


def _stored_chunks(ids: list[str]) -> dict:
    with connection() as conn:
        return chunks_repo.get_chunks_by_ids(conn, ids)


def ensure_document(name: str) -> dict:
    """Документ из data/ в базе и в состоянии indexed (повторный прогон его переиспользует)."""
    content = (DATA / name).read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    document, created = documents_repo.create(
        name, digest, len(content), documents_service.kind_of(name)
    )
    if created:
        documents_service.save_upload(document["id"], name, content)
    if document["status"] == "failed":
        return document
    if document["status"] != "indexed":
        result = documents_service.process_document(document["id"])
        if result["status"] == "parsed":
            documents_service.index_document(document["id"])
    return documents_repo.get(document["id"])


def prompts_fingerprint() -> str:
    """Отпечаток промптов: отчёты с разным отпечатком — до и после правки промпта."""
    text = "".join(
        agent().instructions
        for agent in (
            prompts.extractor_agent,
            prompts.rag_generator_agent,
            prompts.judge_agent,
        )
    )
    return hashlib.sha256(text.encode()).hexdigest()[:10]


async def evaluate_document(name: str, golden: dict, run_id: str) -> dict:
    document = await asyncio.to_thread(ensure_document, name)
    row: dict = {"document": name, "status": document["status"]}
    if document["status"] == "failed":
        # ожидаемый отказ (скан): метрик нет, причина названа
        row["rejected_reason"] = document["reason"]
        return row

    label = f"eval-{run_id}-{name}"
    structlog.contextvars.bind_contextvars(job_id=label)
    started = time.perf_counter()
    product = golden["product"]

    context, guard = await rag_pipeline.retrieve_context_async(product, [document["id"]])
    card, attempts, critic, status, guard = await rag_pipeline.generate_from_context(
        product, context, guard
    )
    issues = guard.output_issues
    draft = card
    verdict = await judge_card(context, draft)

    cited = [s.chunk_id for s in draft.sources]
    stored = await asyncio.to_thread(_stored_chunks, cited)
    chunk_texts = {chunk_id: row["text"] for chunk_id, row in stored.items()}
    chunk_pages = {
        chunk_id: (row["metadata"] or {}).get("page") for chunk_id, row in stored.items()
    }
    citation = citation_report(draft, chunk_texts, golden.get("source_probes"), chunk_pages)
    recall, precision = missing_fields_scores(golden.get("missing_fields", []), card)
    cost = await asyncio.to_thread(llm_calls_repo.cost_for_job, label)

    row.update(
        accuracy=characteristics_accuracy(golden.get("characteristics", {}), card),
        citation=1.0 if citation["card_ok"] else 0.0,
        citation_detail=citation,
        judge=judge_score(draft, verdict),
        judge_verdict=verdict.model_dump(),
        missing_recall=recall,
        missing_precision=precision,
        reference_issues=issues,
        fragments=len(context_chunk_ids(context)),
        cost_rub=str(cost["cost"]),
        llm_calls=cost["calls"],
        tokens=cost["input_tokens"] + cost["output_tokens"],
        seconds=round(time.perf_counter() - started, 1),
        card={
            "characteristics": card.characteristics,
            "missing_fields": card.missing_fields,
            "confidence": card.confidence,
        },
    )
    return row


def _mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _fmt(value, spec: str = ".2f") -> str:
    return "—" if value is None else format(value, spec)


def print_table(rows: list[dict], averages: dict) -> None:
    head = (f"{'документ':<26}{'характер.':>10}{'цитиров.':>10}{'судья':>8}"
            f"{'недост.R':>10}{'недост.P':>10}{'₽':>10}{'сек':>7}")
    print("\n" + head)
    print("-" * len(head))
    for r in rows:
        if "rejected_reason" in r:
            print(f"{r['document']:<26}отказ при разборе (ожидаемо): {r['rejected_reason'][:60]}")
            continue
        print(f"{r['document']:<26}{_fmt(r['accuracy']):>10}{_fmt(r['citation']):>10}"
              f"{_fmt(r['judge']):>8}{_fmt(r['missing_recall']):>10}"
              f"{_fmt(r['missing_precision']):>10}{float(r['cost_rub']):>10.4f}{r['seconds']:>7}")
    print("-" * len(head))
    print(f"{'среднее':<26}{_fmt(averages['accuracy']):>10}{_fmt(averages['citation']):>10}"
          f"{_fmt(averages['judge']):>8}{_fmt(averages['missing_recall']):>10}"
          f"{_fmt(averages['missing_precision']):>10}")


async def run(names: list[str]) -> dict:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))["documents"]
    run_id = uuid.uuid4().hex[:8]
    rows = []
    for name in names:
        print(f"→ {name} ...", flush=True)
        rows.append(await evaluate_document(name, golden[name], run_id))

    scored = [r for r in rows if "rejected_reason" not in r]
    averages = {
        key: _mean([r[key] for r in scored])
        for key in ("accuracy", "citation", "judge", "missing_recall", "missing_precision")
    }
    return {
        "run_id": run_id,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "settings": {
            "llm_model": settings.llm_model,
            "llm_model_light": settings.llm_model_light or settings.llm_model,
            "confidence_threshold": settings.confidence_threshold,
            "context_max_chars": settings.context_max_chars,
            "prompts": prompts_fingerprint(),
        },
        "documents": rows,
        "averages": averages,
        "total_cost_rub": str(sum(float(r.get("cost_rub", 0)) for r in rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Метрики качества генерации")
    parser.add_argument("--all", action="store_true", help="весь эталонный набор")
    parser.add_argument("--docs", help="документы через запятую")
    parser.add_argument("--out", type=Path, help="файл отчёта")
    args = parser.parse_args()

    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    if args.docs:
        names = args.docs.split(",")
    elif args.all:
        names = list(golden["documents"])
    else:
        names = golden["eval_defaults"]

    setup_logging()
    init_pool()
    try:
        report = asyncio.run(run(names))
    finally:
        close_pool()

    print_table(report["documents"], report["averages"])
    REPORTS.mkdir(exist_ok=True)
    out = args.out or REPORTS / f"metrics-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nОтчёт: {out}")


if __name__ == "__main__":
    main()
