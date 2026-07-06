from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lbm_lab.db import init_database  # noqa: E402


if __name__ == "__main__":
    path = init_database("database/lbm_runs.sqlite")
    print(f"Initialized database: {path}")
