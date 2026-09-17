"""Unit tests package. Ensures src and tools are on sys.path."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = ROOT / "src"
TOOLS_DIR = ROOT / "tools"

for p in (str(SRC_DIR), str(TOOLS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)
