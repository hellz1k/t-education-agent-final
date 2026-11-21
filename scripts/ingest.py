"""Thin standalone runner for KB ingestion without requiring editable package install."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from t_edu_agent.ingestion.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
