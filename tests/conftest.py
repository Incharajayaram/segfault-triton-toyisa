"""Shared test configuration: the canon, the fixture registry, the two input classes.

Nothing here imports Triton, torch or a GPU driver. If you find yourself needing
one of them, you are writing a `@pytest.mark.gpu` test and it will not run in CI.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for extra in (ROOT / "src", ROOT / "tools"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from fixtures_lib import FIXTURES, GOLDEN, normalize, read_fixture, scan

DIRECTORY_MARKERS = {
    "unit": "unit",
    "contract": "contract",
    "integration": "integration",
    "e2e": "e2e",
}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--strict-real",
        action="store_true",
        default=False,
        help="turn the two-input-class gate into a hard failure (deadline: end of day 3)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Derive the taxonomy marker from the directory.

    Markers are never written by hand, so a test cannot be filed in the wrong
    layer by forgetting to annotate it (docs/team/testing-ci.md R5).
    """
    for item in items:
        parts = set(Path(str(item.fspath)).parts)
        for directory, marker in DIRECTORY_MARKERS.items():
            if directory in parts:
                item.add_marker(getattr(pytest.mark, marker))
                break


@dataclass(frozen=True)
class Tier:
    """One corpus kernel: its name, its text, its observations and its intent."""

    name: str
    text: str
    obs: dict
    intent: dict

    @property
    def ops(self) -> dict:
        return self.obs["ops"]


@pytest.fixture(scope="session")
def golden() -> dict:
    """The canon. Every expectation in the suite reads from here or from `tier`."""
    return json.loads(GOLDEN.read_text())


@pytest.fixture(scope="session")
def tiers(golden: dict) -> dict[str, Tier]:
    return {
        name: Tier(name, read_fixture(name), obs, golden["intent"][name])
        for name, obs in golden["observations"].items()
    }


@pytest.fixture
def tier(request: pytest.FixtureRequest) -> Tier:
    """Parametrise by fixture name. Use indirectly:

        @pytest.mark.parametrize("tier", ["t1_matmul"], indirect=True)
        def test_x(tier): ...
    """
    name = getattr(request, "param", "t1_matmul")
    golden = json.loads(GOLDEN.read_text())
    return Tier(name, read_fixture(name), golden["observations"][name], golden["intent"][name])


def real_pipeline_available() -> tuple[bool, str]:
    """Is the real upstream (parser + IR builder) importable and callable?"""
    try:
        from tritonflow.ttir.parser import parse_raw  # noqa: F401
        from tritonflow.ttir.to_ir import build_ir  # noqa: F401
    except Exception as exc:  # pragma: no cover - the expected state until day 3
        return False, f"{type(exc).__name__}: {exc}"
    return True, "ok"


@pytest.fixture(params=["handbuilt", "real"])
def input_class(request: pytest.FixtureRequest) -> str:
    """The two input classes every contract test must be parametrised over.

    `handbuilt` builds its input as a literal, so the test runs from hour 0.
    `real` consumes the output of the real upstream stage; until that stage
    lands it is an xfail, and after the day-3 deadline `--strict-real` turns
    the xfail into a failure. The xfail is the deadline mechanism, not an
    excuse (docs/team/testing-ci.md R4).
    """
    if request.param == "real":
        available, why = real_pipeline_available()
        if not available:
            if request.config.getoption("--strict-real"):
                pytest.fail(
                    "STRICT_REAL=1 but the real pipeline is unavailable: "
                    f"{why}\nThe deadline for every `real` row is the end of day 3."
                )
            pytest.xfail(f"real-input class unavailable upstream: {why}")
    return request.param


@pytest.fixture(scope="session")
def fixture_dir() -> Path:
    return FIXTURES


__all__ = ["Tier", "normalize", "scan"]
