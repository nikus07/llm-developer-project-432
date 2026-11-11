"""Сквозной прогон: загрузить выданные документы и сгенерировать по каждому карточку.

    python scripts/demo.py                    # все документы из data/
    python scripts/demo.py --docs kettle_manual.pdf,kettle_spec.xlsx
    python scripts/demo.py --approve          # подтвердить готовые карточки сигналом

Нужны запущенные API и воркер процессов (см. README). Только стандартная библиотека: работает
одинаково в Linux, macOS и Windows (curl в Windows портит кириллицу в теле запроса).
"""
import argparse
import json
import mimetypes
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
EXTENSIONS = {".pdf", ".docx", ".xlsx"}
FINAL = {"awaiting_confirmation", "needs_review", "failed"}


def call(base: str, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(body, ensure_ascii=False).encode() if body is not None else None
    request = urllib.request.Request(
        base + path, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def upload(base: str, path: Path) -> dict:
    boundary = uuid.uuid4().hex
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
        f"filename=\"{path.name}\"\r\nContent-Type: {content_type}\r\n\r\n"
    ).encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(
        base + "/documents", data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def wait_document(base: str, document_id: str, timeout: float = 900) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        _, doc = call(base, "GET", f"/documents/{document_id}")
        if doc["status"] in ("indexed", "failed"):
            return doc
        time.sleep(3)
    raise TimeoutError(f"документ {document_id} не разобрался за {timeout:.0f} с")


def run_generation(base: str, document: dict, hint: str, approve: bool) -> dict:
    status, job = call(base, "POST", "/generate", {"document_ids": [document["id"]], "hint": hint})
    if status != 202:
        return {"document": document["filename"], "error": job.get("detail", job), "http": status}
    last = None
    while True:
        _, job = call(base, "GET", f"/jobs/{job['id']}")
        if job["status"] != last:
            print(f"    {time.strftime('%H:%M:%S')}  {job['status']}", flush=True)
            last = job["status"]
        if job["status"] in FINAL:
            break
        time.sleep(3)
    _, cost = call(base, "GET", f"/jobs/{job['id']}/cost")
    if approve and job["status"] == "awaiting_confirmation":
        call(base, "POST", f"/workflows/{job['id']}/approve")
    return {"document": document["filename"], "job_id": job["id"], "status": job["status"],
            "error": job["error"], "result": job["result"], "report": job["report"],
            "cost": cost}


def summarize(item: dict) -> str:
    if "http" in item:
        return f"не запущено ({item['http']}): {item['error']}"
    card = item["result"] or {}
    report = item["report"] or {}
    lines = [
        f"статус: {item['status']}",
        f"заголовок: {card.get('title', '—')}",
        f"характеристик: {len(card.get('characteristics', {}))}, "
        f"ссылок: {len(card.get('sources', []))}, "
        f"недостающие: {card.get('missing_fields', [])}, "
        f"уверенность: {card.get('confidence', '—')}",
        f"замаскировано: {report.get('masked', {})}, "
        f"исключено фрагментов: {len(report.get('excluded', []))}",
        f"стоимость: {item['cost']['cost']} ₽ за {item['cost']['calls']} вызовов",
    ]
    if item["error"]:
        lines.append(f"примечание: {item['error']}")
    return "\n    ".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--docs", help="имена файлов из data/ через запятую")
    parser.add_argument("--approve", action="store_true", help="подтвердить готовые карточки")
    parser.add_argument("--out", type=Path, help="куда сохранить результаты (JSON)")
    args = parser.parse_args()

    golden = json.loads((DATA / "golden_cards.json").read_text(encoding="utf-8"))["documents"]
    names = args.docs.split(",") if args.docs else sorted(
        p.name for p in DATA.iterdir() if p.suffix in EXTENSIONS
    )

    status, _ = call(args.api, "GET", "/health/ready")
    if status != 200:
        raise SystemExit(f"сервис не готов (GET /health/ready → {status}): проверьте README")

    documents = {}
    for name in names:
        document = upload(args.api, DATA / name)
        print(f"загружен {name}: {document['id']}")
        documents[name] = wait_document(args.api, document["id"])
        doc = documents[name]
        print(f"  → {doc['status']}, фрагментов: {doc['chunks']}"
              + (f", причина: {doc['reason']}" if doc["reason"] else ""))

    results = []
    for name, document in documents.items():
        print(f"\nгенерация: {name}")
        if document["status"] == "failed":
            print("    пропущено: документ отвергнут при разборе (см. причину выше)")
            results.append({"document": name, "status": "rejected_at_parsing",
                            "reason": document["reason"]})
            continue
        hint = golden.get(name, {}).get("product")
        results.append(run_generation(args.api, document, hint, args.approve))
        print("    " + summarize(results[-1]))

    out = args.out or ROOT / "evals" / "reports" / f"cards-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nрезультаты: {out}")


if __name__ == "__main__":
    main()
