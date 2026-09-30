#!/usr/bin/env python3
"""Run the Adult dataset ETL pipeline."""  # noqa: EXE001

import sys
from pathlib import Path

# Make the src-layout package importable when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trustworthy_ai.etl import main

if __name__ == "__main__":
    raise SystemExit(main())
