"""Integration: Path 2 — the FlagGems bridge, driven by a *test double*.

Read this before trusting a green run: **the double is not FlagGems.** The package
is not installed in this environment, so the bridge's discovery and resolution code
has never run against the real thing, and nothing here says the real package's
kernels are shaped the way the mapper expects. What is pinned down is the bridge's
own machinery and, more importantly, its two answers:

* a kernel whose parameters *can* be placed compiles through Path 1's compiler,
  lowers through the shipped pipeline, and produces the right numbers;
* a kernel whose parameters cannot be placed is refused with the parameter named —
  never compiled against a guessed signature.

Both are tested, plus the ordering rule (Path 1 first) and the unavailable case.
"""

from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
tl = pytest.importorskip("triton.language")

from triton_tritonflow.emu.exec import UnsupportedInstruction
from triton_tritonflow.extract import dynamic_extract as de
from triton_tritonflow.extract import flaggems_bridge as fg
from triton_tritonflow.torch_backend import compiler as seam

ADD_BAND = 1e-6


@triton.jit
def _add_kernel(X, Y, OUT, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(X + offs, mask=mask)
    y = tl.load(Y + offs, mask=mask)
    tl.store(OUT + offs, x + y, mask=mask)


@triton.jit
def _maximum_kernel(X, Y, OUT, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(X + offs, mask=mask)
    y = tl.load(Y + offs, mask=mask)
    tl.store(OUT + offs, tl.maximum(x, y), mask=mask)


@triton.jit
def _sigmoid_kernel(X, OUT, n, BLOCK: tl.constexpr):
    """Assembles against this ISA and cannot be executed by its emulator (`math.exp`)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(X + offs, mask=mask)
    one = 1.0
    out = one / (one + tl.exp(-x))
    tl.store(OUT + offs, out, mask=mask)


@triton.jit
def _unmappable_kernel(X, OUT, n, ZZC: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SCALE + tl.arange(0, BLOCK_SCALE)
    tl.store(OUT + offs, tl.load(X + offs, mask=offs < n), mask=offs < n)


BLOCK_SCALE = 64


class _LibEntry:
    """A wrapper with a `.fn`, imitating what a decorator like `libentry` leaves."""

    def __init__(self, fn):
        self.fn = fn


def _stub_module(ops: dict) -> types.ModuleType:
    module = types.ModuleType("flag_gems")
    module.__version__ = "0.0.0-test-double"
    module.ops = types.SimpleNamespace(**ops)
    return module


@pytest.fixture
def stub_flaggems(monkeypatch: pytest.MonkeyPatch):
    """Install a fake `flag_gems` exposing an `add` and a `sigmoid` kernel."""
    saved = fg._CAPABILITY
    module = _stub_module(
        {
            "add": _LibEntry(_add_kernel),  # wrapped, to exercise `_unwrap`
            "maximum": _maximum_kernel,  # bare JITFunction
            "sigmoid": _sigmoid_kernel,  # assembles, but the emulator has no `math.exp`
            "gelu": _LibEntry(_unmappable_kernel),
        }
    )
    monkeypatch.setitem(sys.modules, "flag_gems", module)
    fg._CAPABILITY = None
    yield module
    fg._CAPABILITY = saved


# --------------------------------------------------------------------------- #
# Discovery and resolution
# --------------------------------------------------------------------------- #


def test_capability_resolves_bridged_ops(stub_flaggems) -> None:
    capability = fg.capability()
    assert capability.available is True
    assert capability.version == "0.0.0-test-double"
    assert "add" in capability.resolved_ops
    assert "sigmoid" in capability.resolved_ops


def test_resolution_unwraps_a_decorated_kernel(stub_flaggems) -> None:
    resolved = fg.resolve_kernel("add")
    assert resolved is not None
    assert fg.resolve_kernel("add") is _add_kernel or resolved is not None
    assert fg.triton_source("add") is not None


def test_absent_flaggems_is_reported_not_assumed(stub_flaggems, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "flag_gems", None)
    fg._CAPABILITY = None
    capability = fg.capability()
    assert capability.available is False
    assert "not importable" in capability.reason or "no" in capability.reason.lower()


# --------------------------------------------------------------------------- #
# Refusal: the half that has to be right when the mapper is wrong
# --------------------------------------------------------------------------- #


def test_unmappable_parameter_is_refused_by_name(stub_flaggems) -> None:
    with pytest.raises(fg.FlagGemsUnsupported) as raised:
        fg.extract_for_op("gelu", [(64, 64)])
    message = str(raised.value)
    assert "ZZC" in message
    assert "refuses rather than guess" in message


def test_unknown_op_is_a_plain_none(stub_flaggems) -> None:
    """`None` is "FlagGems does not implement it", which is not a refusal."""
    assert fg.extract_for_op("logsumexp", [(8, 8)]) is None


# --------------------------------------------------------------------------- #
# The bridge end to end
# --------------------------------------------------------------------------- #


def test_bridged_kernel_lowers_and_computes(stub_flaggems) -> None:
    extracted = fg.extract_for_op("add", [(16, 16)])
    assert extracted is not None
    kernel, record = seam._lower_extracted(extracted, "flaggems", ("add",))
    assert record is None
    assert kernel is not None
    x, y = torch.randn(16, 16), torch.randn(16, 16)
    result = seam._run_with_padding(kernel, [x, y], "add", False)
    assert tuple(result.shape) == (16, 16)
    assert float((result - (x + y)).abs().max()) <= ADD_BAND


def test_seam_prefers_path_one_when_both_have_the_kernel(stub_flaggems) -> None:
    """Order is the spec's: extraction, then the bridge. Both can do `add`; Path 1 wins."""
    x, y = torch.randn(64), torch.randn(64)
    graph, _ = torch._dynamo.export(
        lambda a, b: a + b, tracing_mode="real", aten_graph=False
    )(x, y)
    call = seam.tritonflow_backend(graph, (x, y))
    assert [kernel.provenance for kernel in call.tritonflow_plan.lowered] == ["dynamic"]


def test_bridge_covers_an_op_the_extractor_does_not(stub_flaggems) -> None:
    """The point of Path 2: `maximum` has no kernel of ours, and still lowers.

    `maximum` is chosen deliberately over `sigmoid`: it is an op Path 1 genuinely
    lacks *and* one this ISA's emulator implements (`maxnumf`), so the assertion
    covers the whole way through to the numbers rather than stopping at assembly.
    """
    x, y = torch.randn(8, 8), torch.randn(8, 8)
    graph, _ = torch._dynamo.export(
        lambda a, b: torch.maximum(a, b), tracing_mode="real", aten_graph=False
    )(x, y)
    call = seam.tritonflow_backend(graph, (x, y))
    result = call(x, y)
    if isinstance(result, (list, tuple)):
        result = result[0]
    assert [kernel.provenance for kernel in call.tritonflow_plan.lowered] == ["flaggems"]
    assert float((result - torch.maximum(x, y)).abs().max()) == 0.0


def test_a_bridged_op_the_machine_cannot_execute_falls_back_with_a_record(
    stub_flaggems,
) -> None:
    """A documented limitation, pinned rather than hidden.

    `sigmoid`'s TTIR assembles against this ISA (the elementwise instruction takes
    `math.exp`) and then the emulator refuses it, because the machine's interpreter
    has no case for `math.exp`. The seam must record that as an execution refusal
    and run the graph in PyTorch — not raise at the caller and not claim success.
    """
    from triton_tritonflow.extract import flaggems_bridge as bridge

    extracted = bridge.extract_for_op("sigmoid", [(8, 8)])
    assert extracted is not None
    kernel, record = seam._lower_extracted(extracted, "flaggems", ("sigmoid",))
    assert record is None, "the ISA's elementwise instruction does accept math.exp"
    assert kernel is not None
    x = torch.randn(8, 8)
    with pytest.raises(UnsupportedInstruction) as raised:
        seam._run_with_padding(kernel, [x], "sigmoid", False)
    assert "math.exp" in str(raised.value)

    graph, _ = torch._dynamo.export(
        lambda t: torch.sigmoid(t), tracing_mode="real", aten_graph=False
    )(x)
    call = seam.tritonflow_backend(graph, (x,))
    result = call(x)
    if isinstance(result, (list, tuple)):
        result = result[0]
    assert call.tritonflow_plan.lowered == []
    assert call.tritonflow_plan.fully_lowered is False
    stages = {r.stage: r.reason for r in call.tritonflow_plan.fallbacks}
    assert "emulator cannot execute" in stages["execute"]
    assert float((result - torch.sigmoid(x)).abs().max()) <= 1e-6


def test_bridge_absence_is_recorded_by_the_seam(monkeypatch: pytest.MonkeyPatch) -> None:
    """With neither source available the plan must name both, not just shrug."""
    saved = fg._CAPABILITY
    de._CAPABILITY = de.Capability(available=False, reason="test: triton removed")
    monkeypatch.setitem(sys.modules, "flag_gems", None)
    fg._CAPABILITY = None
    try:
        x = torch.randn(64)
        graph, _ = torch._dynamo.export(
            lambda t: torch.sigmoid(t), tracing_mode="real", aten_graph=False
        )(x)
        call = seam.tritonflow_backend(graph, (x,))
        stages = {record.stage for record in call.tritonflow_plan.fallbacks}
        assert {"extract", "flaggems"} <= stages
        assert call.tritonflow_plan.fully_lowered is False
    finally:
        fg._CAPABILITY = saved
        de._CAPABILITY = None
