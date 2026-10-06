"""The one-command run: game files in, projections out, on an empty database."""

import sys
import warnings

import psycopg
import pytest

from app import db
from app.pei import __main__ as cli
from tests.test_pei_features import DATA_DIR

warnings.filterwarnings("ignore")


@pytest.mark.skipif(not DATA_DIR.exists(), reason="real game files not on this machine")
def test_all_runs_from_an_empty_database(monkeypatch):
    try:
        conn = db.connect(db.TEST_DB_NAME)
    except psycopg.OperationalError as err:
        pytest.skip(f"test database not reachable ({err})")
    db.apply_schema(conn)
    conn.execute("TRUNCATE core.import_file, core.game, core.player, core.team, "
                 "pei.model_version CASCADE")
    conn.commit()

    monkeypatch.setenv("AM_DB_NAME", db.TEST_DB_NAME)
    monkeypatch.setattr(sys, "argv", ["pei", "all", str(DATA_DIR), "--version", "test-run"])
    assert cli._all() == 0

    def count(sql):
        return conn.execute(sql).fetchone()[0]

    assert count("SELECT count(*) FROM core.pitch") > 0
    assert count("SELECT count(*) FROM pei.plate_appearance") > 0
    assert count("SELECT count(*) FROM pei.projection WHERE version = 'test-run'") > 0
    low, projected, high = conn.execute(
        "SELECT min(woba_lo), avg(proj_woba), max(woba_hi) FROM pei.projection").fetchone()
    assert 0 < low < projected < high < 1

    conn.execute("TRUNCATE core.import_file, core.game, core.player, core.team, "
                 "pei.model_version CASCADE")
    conn.commit()
    conn.close()
