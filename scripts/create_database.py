"""One-time setup: create the app's role and its two databases.

Run from the project folder:

    .venv\\Scripts\\python.exe scripts\\create_database.py

It asks for the password of the `postgres` admin user (the one you chose when
PostgreSQL was installed). That password is used for this one connection and is
not saved anywhere.

What it does:
  1. creates the role `metrics_app` with a random password
  2. creates the databases `advanced_metrics` and `advanced_metrics_test`,
     both owned by that role
  3. adds two lines to %APPDATA%\\postgresql\\pgpass.conf so the app can log
     in without a password in the code
  4. builds the tables in both databases

Running it again is safe: anything that already exists is left alone.
"""

import getpass
import os
import secrets
import sys
from pathlib import Path

import psycopg
from psycopg import sql

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import db  # noqa: E402

PGPASS = Path(os.environ["APPDATA"]) / "postgresql" / "pgpass.conf"


def _pgpass_line(dbname: str, password: str) -> str:
    return f"{db.DB_HOST}:{db.DB_PORT}:{dbname}:{db.DB_USER}:{password}"


def _has_pgpass_entry() -> bool:
    if not PGPASS.exists():
        return False
    return any(f":{db.DB_USER}:" in line for line in PGPASS.read_text().splitlines())


def main() -> None:
    admin_password = os.environ.get("PGPASSWORD") or getpass.getpass("postgres admin password: ")
    admin = psycopg.connect(host=db.DB_HOST, port=db.DB_PORT, dbname="postgres",
                            user="postgres", password=admin_password, autocommit=True)

    role_exists = admin.execute(
        "SELECT 1 FROM pg_roles WHERE rolname = %s", (db.DB_USER,)).fetchone()

    if role_exists and _has_pgpass_entry():
        print(f"role {db.DB_USER} already exists and pgpass.conf has its password")
    else:
        # New role, or a role whose password we no longer have: set a fresh one.
        password = secrets.token_urlsafe(24)
        verb = "ALTER" if role_exists else "CREATE"
        admin.execute(sql.SQL("{} ROLE {} LOGIN PASSWORD {}").format(
            sql.SQL(verb), sql.Identifier(db.DB_USER), sql.Literal(password)))
        PGPASS.parent.mkdir(parents=True, exist_ok=True)
        with PGPASS.open("a", encoding="utf-8") as fh:
            for name in (db.DB_NAME, db.TEST_DB_NAME):
                fh.write(_pgpass_line(name, password) + "\n")
        print(f"{verb.lower()}d role {db.DB_USER}; password written to {PGPASS}")

    for name in (db.DB_NAME, db.TEST_DB_NAME):
        if admin.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone():
            print(f"database {name} already exists")
        else:
            admin.execute(sql.SQL("CREATE DATABASE {} OWNER {}").format(
                sql.Identifier(name), sql.Identifier(db.DB_USER)))
            print(f"created database {name}")
    admin.close()

    for name in (db.DB_NAME, db.TEST_DB_NAME):
        with db.connect(name) as conn:
            print(f"{name}: applied {', '.join(db.apply_schema(conn))}")


if __name__ == "__main__":
    main()
