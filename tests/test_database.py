from lbm_lab import db


def test_database_initializes(tmp_path):
    db_path = tmp_path / "runs.sqlite"
    db.init_database(db_path)

    with db.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        ).fetchall()

    assert ("simulations",) in rows
    assert ("metrics",) in rows
    assert ("artifacts",) in rows

