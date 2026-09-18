"""T004 / FR-003: the package imports and degrades with Triton absent.

`triton` is an *optional* extra. Dynamic extraction is the feature it buys, so the
question this file answers is not "does the code have an `if` in it" but "with
Triton removed from the process, does the package still import, does the probe
report *why*, and does extraction refuse with a typed error instead of a
`NameError` or a silent `None`".

The block is done by putting `None` in `sys.modules`, which makes `import triton`
raise `ImportError` inside the interpreter rather than merely failing a lookup —
closer to a machine that never installed it.
"""

from __future__ import annotations

import importlib
import sys

import pytest

from tritonflow.extract import dynamic_extract as de
from tritonflow.extract import flaggems_bridge as fg


@pytest.fixture
def without_triton(monkeypatch: pytest.MonkeyPatch):
    """Block `import triton` and reset both probe caches around the test."""
    saved_capability = de._CAPABILITY
    saved_flaggems = fg._CAPABILITY
    for name in ("triton", "triton.language", "triton.compiler", "triton.backends"):
        monkeypatch.setitem(sys.modules, name, None)
    de._CAPABILITY = None
    fg._CAPABILITY = None
    yield
    de._CAPABILITY = saved_capability
    fg._CAPABILITY = saved_flaggems


def test_package_imports_without_triton(without_triton) -> None:
    """`import tritonflow` must not need the compiler it can optionally drive."""
    module = importlib.import_module("tritonflow")
    assert module is not None


def test_extract_package_imports_without_triton(without_triton) -> None:
    module = importlib.import_module("tritonflow.extract")
    assert module.is_triton_available() is False


def test_capability_reports_the_reason(without_triton) -> None:
    capability = de.capability()
    assert capability.available is False
    assert "triton" in capability.reason.lower()
    assert capability.triton_version is None


def test_extraction_refuses_with_a_typed_error(without_triton) -> None:
    with pytest.raises(de.ExtractionUnavailable) as raised:
        de.extract_matmul(8, 8, 8)
    assert "triton" in str(raised.value).lower()


def test_compile_entry_point_refuses_with_a_typed_error(without_triton) -> None:
    with pytest.raises(de.ExtractionUnavailable):
        de.compile_triton_kernel(object(), {"x_ptr": "*fp32"})


def test_extract_for_op_does_not_swallow_unavailability(without_triton) -> None:
    """A missing compiler is not "this op is unsupported" and must not look like it."""
    with pytest.raises(de.ExtractionUnavailable):
        de.extract_for_op("matmul", [(8, 8), (8, 8)])


def test_flaggems_bridge_reports_unavailability(without_triton) -> None:
    capability = fg.capability()
    assert capability.available is False
    assert "flag_gems" in capability.reason
    assert fg.implemented_ops() == ()
    assert fg.resolve_kernel("add") is None
    assert fg.triton_source("add") is None


def test_compile_spy_survives_a_missing_compiler(without_triton) -> None:
    """The observer must not be the thing that breaks a Triton-free process."""
    with de.record_compilations() as spy:
        pass
    assert spy.records == []
    assert any("triton" in message.lower() for message in spy.errors)


def test_capability_is_reprobed_not_sticky(monkeypatch: pytest.MonkeyPatch) -> None:
    """`refresh=True` must actually re-probe, or a stale verdict outlives its cause."""
    de._CAPABILITY = de.Capability(available=False, reason="stale verdict from another test")
    try:
        refreshed = de.capability(refresh=True)
        # Whatever this environment has, the probe must not have returned the
        # planted value: that is the only thing this test can assert without
        # assuming Triton is installed.
        assert refreshed.reason != "stale verdict from another test"
    finally:
        de._CAPABILITY = None
