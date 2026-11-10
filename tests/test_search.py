"""Поиск на заранее заданных векторах: модель эмбеддингов не загружается."""
import hashlib
import math
import uuid

import psycopg
import pytest

from app.core import db
from app.core.config import settings
from app.core.migrate import migrate
from app.parsers.models import Chunk
from app.rag import embedder, reindex, retrieval
from app.repositories import chunks as chunks_repo
from app.repositories import documents as documents_repo

DIM = settings.embedding_dim


def basis(index: int, weight: float = 1.0, other: dict[int, float] | None = None) -> list[float]:
    vector = [0.0] * DIM
    vector[index] = weight
    for i, w in (other or {}).items():
        vector[i] = w
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector]


# запрос: почти «блендер» (ось 1) и чуть-чуть «чайник» (ось 0)
QUERY_VECTOR = basis(1, 1.0, {0: 0.3})

CHUNKS = [
    Chunk(
        0, "text", "Чайник KTL-1700 мощность 2200 Вт", 1, "Характеристики", "KTL-1700", "КеттлПро"
    ),
    Chunk(1, "text", "Блендер погружной металлическая ножка", 1, "Описание", None, None),
    Chunk(2, "text", "Гарантия двенадцать месяцев", 2, "Гарантия", None, None),
    Chunk(3, "text", "Совсем другая тема", 2, "Прочее", None, None),
]
VECTORS = [basis(0), basis(1), basis(2), basis(3)]


@pytest.fixture(scope="module")
def live_db():
    try:
        with psycopg.connect(settings.database_url, connect_timeout=2):
            pass
    except psycopg.OperationalError:
        pytest.skip("база недоступна: поднимите make infra")
    migrate()
    db.init_pool()
    ids: list[str] = []
    yield ids
    with db.connection() as conn:
        conn.execute("DELETE FROM documents WHERE id = ANY(%s)", (ids,))
    db.close_pool()


def _document(ids: list[str], *, with_vectors: bool) -> dict:
    digest = hashlib.sha256(uuid.uuid4().bytes).hexdigest()
    document, _ = documents_repo.create("search-test.pdf", digest, 1)
    ids.append(document["id"])
    chunks_repo.replace_for_document(document["id"], "search-test.pdf", CHUNKS)
    rows = chunks_repo.list_without_embeddings(100, document["id"])
    if with_vectors:
        chunks_repo.write_embeddings([(r["id"], v) for r, v in zip(rows, VECTORS, strict=True)])
    return document


@pytest.fixture
def doc(live_db):
    return _document(live_db, with_vectors=True)


def _texts(hits: list[dict]) -> list[str]:
    return [h["text"] for h in hits]


def _filters(doc: dict) -> dict:
    return {"document_id": doc["id"]}


def test_vector_search_orders_by_cosine_similarity(doc):
    hits = chunks_repo.search_vector(
        QUERY_VECTOR, limit=4, min_score=0.0, filters=_filters(doc)
    )
    assert _texts(hits)[:2] == [CHUNKS[1].text, CHUNKS[0].text]
    assert hits[0]["score"] > 0.9 > hits[1]["score"] > 0.2


def test_threshold_cuts_irrelevant_results(doc):
    hits = chunks_repo.search_vector(
        QUERY_VECTOR, limit=10, min_score=0.5, filters=_filters(doc)
    )
    assert _texts(hits) == [CHUNKS[1].text]  # «чайник» близок на 0.29 — отсечён


def test_keyword_search_finds_exact_article_only(doc):
    hits = chunks_repo.search_keyword("KTL-1700", limit=10, filters=_filters(doc))
    assert _texts(hits) == [CHUNKS[0].text]
    assert chunks_repo.search_keyword("мотор", limit=10, filters=_filters(doc)) == []


def test_hybrid_returns_union_of_both_lists(doc):
    hits = chunks_repo.search_hybrid(
        QUERY_VECTOR, "KTL-1700", limit=10, min_score=0.2, candidates=20, rrf_k=60,
        filters=_filters(doc),
    )
    assert set(_texts(hits)) == {CHUNKS[0].text, CHUNKS[1].text}


def test_rrf_ranks_chunk_found_by_both_lists_first(doc):
    hits = chunks_repo.search_hybrid(
        QUERY_VECTOR, "KTL-1700", limit=10, min_score=0.2, candidates=20, rrf_k=60,
        filters=_filters(doc),
    )
    # чайник: 2-й в векторном списке и 1-й в словесном; блендер: 1-й только в векторном
    assert _texts(hits) == [CHUNKS[0].text, CHUNKS[1].text]
    assert float(hits[0]["score"]) == pytest.approx(1 / (60 + 2) + 1 / (60 + 1))
    assert float(hits[1]["score"]) == pytest.approx(1 / (60 + 1))


def test_hybrid_applies_threshold_to_vector_list(doc):
    hits = chunks_repo.search_hybrid(
        QUERY_VECTOR, "несуществующее слово", limit=10, min_score=0.5, candidates=20,
        rrf_k=60, filters=_filters(doc),
    )
    assert _texts(hits) == [CHUNKS[1].text]  # мусор до слияния не дошёл


def test_metadata_filters(doc):
    base = _filters(doc)
    by_article = chunks_repo.search_vector(
        QUERY_VECTOR, limit=10, min_score=0.0, filters={**base, "article": "ktl-1700"}
    )
    assert _texts(by_article) == [CHUNKS[0].text]
    by_page = chunks_repo.search_vector(
        QUERY_VECTOR, limit=10, min_score=0.0, filters={**base, "page": 2}
    )
    assert {h["page"] for h in by_page} == {2}
    by_brand = chunks_repo.search_keyword(
        "KTL-1700", limit=10, filters={**base, "brand": "КеттлПро"}
    )
    assert len(by_brand) == 1
    assert chunks_repo.search_keyword(
        "KTL-1700", limit=10, filters={**base, "brand": "Другой"}
    ) == []


def test_unknown_filter_keys_are_ignored(doc):
    hits = chunks_repo.search_vector(
        QUERY_VECTOR, limit=10, min_score=0.0,
        filters={**_filters(doc), "text; DROP TABLE chunks": "x"},
    )
    assert len(hits) == 4


def test_hit_has_id_text_score_and_metadata(doc):
    hit = chunks_repo.search_keyword("KTL-1700", limit=1, filters=_filters(doc))[0]
    for key in ("id", "text", "score", "document", "document_id", "page", "section", "article"):
        assert key in hit
    assert hit["document"] == "search-test.pdf"
    assert hit["section"] == "Характеристики"


def test_retrieval_modes_use_fake_query_embedder(doc, monkeypatch):
    monkeypatch.setattr(retrieval, "embed_query", lambda text: QUERY_VECTOR)
    f = _filters(doc)
    vector = retrieval.search("мотор", mode="vector", min_score=0.5, filters=f)
    keyword = retrieval.search("мотор", mode="keyword", filters=f)
    hybrid = retrieval.search("KTL-1700", mode="hybrid", min_score=0.2, filters=f)
    assert _texts(vector) == [CHUNKS[1].text]
    assert keyword == []
    assert _texts(hybrid)[0] == CHUNKS[0].text


def test_reindex_fills_missing_vectors_and_is_idempotent(live_db, monkeypatch):
    document = _document(live_db, with_vectors=False)
    calls: list[int] = []

    def fake_embed(texts: list[str]) -> list[list[float]]:
        calls.append(len(texts))
        return [basis(i % DIM) for i in range(len(texts))]

    monkeypatch.setattr(reindex, "embed_documents", fake_embed)

    assert reindex.index_chunks(document["id"], batch_size=3) == 4
    assert calls == [3, 1]  # пачками
    assert chunks_repo.list_without_embeddings(10, document["id"]) == []
    assert reindex.index_chunks(document["id"]) == 0  # повторный запуск — пустая работа
    assert calls == [3, 1]


# --- модуль эмбеддингов: без загрузки модели ---


class _FakeModel:
    def __init__(self):
        self.calls: list[tuple[list[str], dict]] = []

    def encode(self, texts, **kwargs):
        self.calls.append((texts, kwargs))

        class Result:
            def tolist(self_inner):
                return [[0.0] * DIM for _ in texts]

        return Result()


def test_embedder_uses_prefixes_batches_and_normalizes(monkeypatch):
    model = _FakeModel()
    monkeypatch.setattr(embedder, "_model", lambda: model)

    embedder.embed_documents(["один", "два"])
    embedder.embed_query("вопрос")

    (doc_texts, doc_kwargs), (query_texts, _) = model.calls
    assert doc_texts == ["passage: один", "passage: два"]  # документ — свой префикс
    assert query_texts == ["query: вопрос"]  # запрос — другой
    assert doc_kwargs["normalize_embeddings"] is True
    assert doc_kwargs["batch_size"] == settings.embedding_batch_size


def test_embedder_empty_input_does_not_touch_model(monkeypatch):
    def boom():
        raise AssertionError("модель не должна загружаться")

    monkeypatch.setattr(embedder, "_model", boom)
    assert embedder.embed_documents([]) == []
