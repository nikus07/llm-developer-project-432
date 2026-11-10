from app.core import migrate


def test_migrations_are_discovered_in_order():
    names = [p.name for p in migrate.discover()]
    assert names == sorted(names)
    assert names[0] == "0001_init.sql"


def test_first_migration_enables_vector_and_tracks_versions():
    sql = (migrate.MIGRATIONS_DIR / "0001_init.sql").read_text(encoding="utf-8")
    assert "CREATE EXTENSION IF NOT EXISTS vector" in sql
    assert "schema_migrations" in sql
