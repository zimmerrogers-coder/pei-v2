"""Stage 1 check: the app role can log in to the test database and the tables exist."""

import psycopg
import pytest

from app import db

EXPECTED = {
    ("core", "import_file"), ("core", "team"), ("core", "player"),
    ("core", "game"), ("core", "pitch"), ("core", "dartfish_pitch"),
}


@pytest.fixture
def conn():
    try:
        connection = db.connect(db.TEST_DB_NAME)
    except psycopg.OperationalError as err:
        pytest.skip(f"test database not reachable; run scripts/create_database.py ({err})")
    yield connection
    connection.close()


def test_schema_applies_and_is_repeatable(conn):
    db.apply_schema(conn)
    db.apply_schema(conn)      # a second run must not fail or duplicate anything
    rows = conn.execute(
        "SELECT table_schema, table_name FROM information_schema.tables "
        "WHERE table_schema IN ('core', 'pei')").fetchall()
    assert EXPECTED <= set(rows)


def test_same_file_cannot_be_imported_twice(conn):
    db.apply_schema(conn)
    insert = ("INSERT INTO core.import_file (source, filename, file_hash, row_count) "
              "VALUES ('trackman', %s, 'abc123', 10)")
    conn.execute(insert, ("game.csv",))
    with pytest.raises(psycopg.errors.UniqueViolation):
        conn.execute(insert, ("game (1).csv",))
    conn.rollback()
