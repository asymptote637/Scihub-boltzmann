"""SQLite run registry for LBM simulations."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
import argparse
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterator


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS simulations (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    config_json TEXT NOT NULL,
    output_dir TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    error TEXT
);

CREATE TABLE IF NOT EXISTS metrics (
    simulation_id TEXT NOT NULL,
    step INTEGER NOT NULL,
    name TEXT NOT NULL,
    value REAL NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (simulation_id) REFERENCES simulations(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_metrics_simulation_step
ON metrics(simulation_id, step);

CREATE TABLE IF NOT EXISTS artifacts (
    simulation_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    path TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (simulation_id) REFERENCES simulations(id) ON DELETE CASCADE
);
"""


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def init_database(path: str | Path) -> Path:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA)
    return db_path


@contextmanager
def connect(path: str | Path) -> Iterator[sqlite3.Connection]:
    db_path = init_database(path)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def create_simulation(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    name: str,
    kind: str,
    config: dict[str, Any],
    output_dir: str | Path,
) -> None:
    conn.execute(
        """
        INSERT INTO simulations
        (id, name, kind, status, config_json, output_dir, started_at)
        VALUES (?, ?, ?, 'running', ?, ?, ?)
        """,
        (run_id, name, kind, json.dumps(config, sort_keys=True), str(output_dir), utc_now()),
    )


def finish_simulation(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    status: str = "completed",
    error: str | None = None,
) -> None:
    conn.execute(
        """
        UPDATE simulations
        SET status = ?, finished_at = ?, error = ?
        WHERE id = ?
        """,
        (status, utc_now(), error, run_id),
    )


def add_metric(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    step: int,
    name: str,
    value: float,
) -> None:
    conn.execute(
        """
        INSERT INTO metrics (simulation_id, step, name, value, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (run_id, int(step), name, float(value), utc_now()),
    )


def add_artifact(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    kind: str,
    path: str | Path,
) -> None:
    conn.execute(
        """
        INSERT INTO artifacts (simulation_id, kind, path, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (run_id, kind, str(path), utc_now()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize the LBM run database.")
    parser.add_argument("--database", default="database/lbm_runs.sqlite")
    args = parser.parse_args()
    db_path = init_database(args.database)
    print(f"Initialized database: {db_path}")


if __name__ == "__main__":
    main()

