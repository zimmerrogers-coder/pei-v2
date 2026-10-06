"""Database access: one place that knows how to connect and how to build the tables.

No password appears here. libpq (the library underneath psycopg) looks the
password up in %APPDATA%\\postgresql\\pgpass.conf, which
scripts/create_database.py writes once.
"""

import os
from pathlib import Path

import psycopg

DB_NAME = "advanced_metrics"
TEST_DB_NAME = "advanced_metrics_test"
DB_USER = "metrics_app"
DB_HOST = "localhost"
DB_PORT = 5432

SQL_DIR = Path(__file__).resolve().parent.parent / "sql"


def connect(dbname: str | None = None) -> psycopg.Connection:
    """Open a connection as the app role. AM_DB_NAME overrides the database,
    which is how the tests point at the test database."""
    return psycopg.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER,
        dbname=dbname or os.environ.get("AM_DB_NAME", DB_NAME),
    )


def apply_schema(conn: psycopg.Connection) -> list[str]:
    """Run every sql/NN_*.sql file in name order. Each file is written so that
    running it again changes nothing. Returns the file names it ran."""
    ran = []
    for path in sorted(SQL_DIR.glob("[0-9][0-9]_*.sql")):
        conn.execute(path.read_text(encoding="utf-8"))
        ran.append(path.name)
    conn.commit()
    return ran


if __name__ == "__main__":
    with connect() as conn:
        print("applied:", ", ".join(apply_schema(conn)))
