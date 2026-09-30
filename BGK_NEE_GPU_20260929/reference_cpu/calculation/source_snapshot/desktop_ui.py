"""Launcher for the PySide6 desktop UI.

Run this file to open the non-web desktop frontend. It does not bind to any
local web port.
"""

from __future__ import annotations

from qt_ui import main


if __name__ == "__main__":
    raise SystemExit(main())
