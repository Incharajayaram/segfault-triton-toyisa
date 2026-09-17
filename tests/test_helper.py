"""Shared test helpers for unittest: fixtures, goldens, and test assertions.

Zero dependency on pytest. Works with standard library unittest.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
TOOLS_DIR = ROOT / "tools"

for p in (str(SRC_DIR), str(TOOLS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from fixtures_lib import FIXTURES, GOLDEN, normalize, read_fixture, scan


@dataclass(frozen=True)
class Tier:
    """One corpus kernel fixture: name, text, observations, and intent."""

    name: str
    text: str
    obs: dict
    intent: dict

    @property
    def ops(self) -> dict:
        return self.obs["ops"]


def load_golden_canon() -> dict:
    """Load the golden fixture specifications."""
    return json.loads(GOLDEN.read_text())


def get_all_tiers() -> dict[str, Tier]:
    """Retrieve all tiers defined in the canon."""
    golden = load_golden_canon()
    return {
        name: Tier(name, read_fixture(name), obs, golden["intent"][name])
        for name, obs in golden["observations"].items()
    }


def get_tier(name: str) -> Tier:
    """Retrieve one specific tier."""
    golden = load_golden_canon()
    return Tier(name, read_fixture(name), golden["observations"][name], golden["intent"][name])


__all__ = [
    "FIXTURES",
    "GOLDEN",
    "ROOT",
    "Tier",
    "get_all_tiers",
    "get_tier",
    "load_golden_canon",
    "normalize",
    "read_fixture",
    "scan",
]
