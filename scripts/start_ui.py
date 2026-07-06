"""Start the Streamlit UI with a clean environment."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    env: dict[str, str] = {}
    for key, value in os.environ.items():
        normalized = "Path" if key.lower() == "path" else key
        env[normalized] = value
    env["PYTHONPATH"] = str(ROOT)
    env["MPLBACKEND"] = "Agg"

    log_path = LOG_DIR / "streamlit.last.log"
    with log_path.open("ab") as log:
        print("Starting Streamlit at http://localhost:8501")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "ui_app.py",
                "--server.headless",
                "true",
                "--server.port",
                "8501",
                "--server.address",
                "localhost",
                "--browser.gatherUsageStats",
                "false",
            ],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )


if __name__ == "__main__":
    main()
