"""Command-line runner for configured LBM simulations."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
import time
import uuid

import numpy as np

from lbm_lab.config import load_config
from lbm_lab import db
from lbm_lab.simulations import cavity


def make_run_id(name: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid.uuid4().hex[:8]
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
    return f"{stamp}_{safe_name}_{suffix}"


def run_from_config(config_path: str | Path) -> Path:
    config_path = Path(config_path)
    config = load_config(config_path)
    run_id = make_run_id(config.name)
    output_dir = config.output.results_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, output_dir / "config.toml")

    started = time.perf_counter()
    metrics_rows: list[cavity.MetricRow] = []
    final_state: cavity.CavityState | None = None

    with db.connect(config.output.database) as conn:
        db.create_simulation(
            conn,
            run_id=run_id,
            name=config.name,
            kind=config.kind,
            config=config.raw,
            output_dir=output_dir,
        )

        try:
            if config.kind != "lid_driven_cavity":
                raise ValueError(f"Unsupported simulation kind: {config.kind}")

            for state, row in cavity.run(config):
                final_state = state
                metrics_rows.append(row)
                db.add_metric(conn, run_id=run_id, step=row.step, name="max_speed", value=row.max_speed)
                db.add_metric(
                    conn, run_id=run_id, step=row.step, name="mean_density", value=row.mean_density
                )
                db.add_metric(
                    conn, run_id=run_id, step=row.step, name="mass_error", value=row.mass_error
                )

            metrics_path = output_dir / "metrics.csv"
            cavity.write_metrics_csv(metrics_path, metrics_rows)
            db.add_artifact(conn, run_id=run_id, kind="metrics_csv", path=metrics_path)

            if config.output.save_final_fields and final_state is not None:
                fields_path = output_dir / "fields_final.npz"
                np.savez_compressed(
                    fields_path,
                    rho=final_state.rho,
                    ux=final_state.ux,
                    uy=final_state.uy,
                    f=final_state.f,
                )
                db.add_artifact(conn, run_id=run_id, kind="fields_npz", path=fields_path)

            elapsed_s = time.perf_counter() - started
            summary = {
                "run_id": run_id,
                "name": config.name,
                "kind": config.kind,
                "config": str(config_path),
                "output_dir": str(output_dir),
                "elapsed_s": elapsed_s,
                "last_metric": metrics_rows[-1].__dict__ if metrics_rows else None,
            }
            summary_path = output_dir / "run_summary.json"
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            db.add_artifact(conn, run_id=run_id, kind="summary_json", path=summary_path)
            db.finish_simulation(conn, run_id=run_id)
        except Exception as exc:
            db.finish_simulation(conn, run_id=run_id, status="failed", error=str(exc))
            raise

    print(f"Completed run: {run_id}")
    print(f"Output: {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an LBM simulation from a TOML config.")
    parser.add_argument("--config", default="configs/default.toml", help="Path to a TOML config file.")
    args = parser.parse_args()
    run_from_config(args.config)


if __name__ == "__main__":
    main()

