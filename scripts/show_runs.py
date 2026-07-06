from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lbm_lab.db import init_database  # noqa: E402


def main() -> None:
    db_path = init_database("database/lbm_runs.sqlite")
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, name, kind, status, started_at, finished_at, output_dir
            FROM simulations
            ORDER BY started_at DESC
            LIMIT 20
            """
        ).fetchall()

    if not rows:
        print("No simulation runs found.")
        return

    for run_id, name, kind, status, started_at, finished_at, output_dir in rows:
        print(f"{run_id} | {status} | {name} | {kind}")
        print(f"  started : {started_at}")
        print(f"  finished: {finished_at}")
        print(f"  output  : {output_dir}")


if __name__ == "__main__":
    main()

