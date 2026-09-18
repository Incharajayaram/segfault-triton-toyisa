"""Path 2: the FlagGems bridge — op coverage bought rather than written.

`flag_gems` implements a large slice of the ATen surface as `@triton.jit`
kernels. If those kernels are reachable as functions, then their *source* can be
handed to the same ahead-of-time compiler Path 1 uses and the TTIR read back —
which is the entire value of the bridge: coverage without re-authoring kernels
here, and without teaching the selector a single new instruction.

**What this module actually does, in order, and what it refuses.**

1. *Discover* — import `flag_gems`, enumerate the op names in
   :data:`OP_KERNELS` that it implements, and report the count. This is real
   introspection of the installed package and the only part that can be checked
   without compiling anything.
2. *Resolve* — find the op's `@triton.jit` function, unwrapping the `fn` /
   `__wrapped__` wrappers that decorators like `libentry` put around it.
3. *Map* — build a Triton signature and a launch environment from the op's
   operand shapes. This is where the bridge is honest about its limits: every
   parameter must be recognised — a pointer that corresponds to an operand, or a
   scalar/extent/strides the env can supply, or a `constexpr` with a known value.
   A parameter it cannot place is a :class:`FlagGemsUnsupported` refusal naming
   the parameter, *not* a guess. FlagGems kernels come in many shapes and the
   mapper is deliberately strict, because a kernel compiled against a mismapped
   signature computes the wrong thing and reports success.
4. *Compile* — `dynamic_extract.compile_triton_kernel`, the same entry point and
   the same explicit `GPUTarget` Path 1 uses.

**Evidence, stated plainly.** `flag_gems` is not installed in this repository's
environment, so steps 1–2 have never run against the real package here, and the
mapper has been exercised only against a test double that presents kernels the
way the mapper expects to find them (`tests/integration/test_flaggems_bridge.py`).
The refusal path — the part that has to be right when the mapper is wrong — is
what the tests can and do pin down. This module is therefore *wired and
unverified against upstream*, and calling it "180+ ops supported" on the basis of
this file would be exactly the kind of claim this project exists not to make.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .dynamic_extract import (
    DEFAULT_TILE,
    Extracted,
    compile_triton_kernel,
)

__all__ = [
    "OP_KERNELS",
    "FlagGemsCapability",
    "FlagGemsUnsupported",
    "capability",
    "extract_for_op",
    "implemented_ops",
    "is_available",
    "resolve_kernel",
    "triton_source",
]

#: Op names the bridge looks for, and the attribute spellings FlagGems may use
#: for them. Read off the package's own naming convention: `flag_gems.ops.<name>`
#: is the public surface, `flag_gems.<name>` is the older one, and several ops
#: carry an underscore-suffixed in-place spelling or a longer alias
#: (`true_divide` for `div`, `cross_entropy_loss` for `cross_entropy`).
OP_KERNELS: Mapping[str, tuple[str, ...]] = {
    # elementwise
    "add": ("add", "add_"),
    "sub": ("sub", "sub_", "subtract"),
    "mul": ("mul", "mul_", "multiply"),
    "div": ("div", "div_", "true_divide"),
    "relu": ("relu", "relu_"),
    "neg": ("neg", "negative"),
    "exp": ("exp", "exp_"),
    "log": ("log", "log_"),
    "sqrt": ("sqrt",),
    "sigmoid": ("sigmoid",),
    "tanh": ("tanh",),
    "gelu": ("gelu", "gelu_"),
    "silu": ("silu", "silu_", "swish"),
    # matmul family
    "matmul": ("mm", "matmul"),
    "mm": ("mm", "matmul"),
    "linear": ("linear", "matmul"),
    # reduction / norm
    "sum": ("sum", "sum_dim"),
    "mean": ("mean",),
    "max": ("max",),
    "maximum": ("maximum", "max", "maximum_"),
    "minimum": ("minimum", "min", "minimum_"),
    "softmax": ("softmax",),
    "layer_norm": ("layer_norm",),
    "rms_norm": ("rms_norm",),
    "cross_entropy": ("cross_entropy_loss", "cross_entropy"),
    # conv / norm
    "conv2d": ("conv2d",),
    "batch_norm": ("batch_norm",),
}

#: Names that mean "the op's extents", in the order a kernel usually declares
#: them. Used to build the launch env; a kernel with an unrecognised spelling in
#: this role is refused rather than guessed at.
_EXTENT_NAMES = ("M", "N", "K")

#: Names a kernel may use for its flat lane count.
_FLAT_NAMES = ("n", "N", "numel", "num_elements", "n_elements", "total", "N_total")

#: Parameter names that mean "a tensor operand". Unambiguous substrings only:
#: the earlier version also matched the single letters `a b c x y z` *anywhere* in
#: a name, so `BLOCK` was a pointer because it contains a `b` — measured, not
#: imagined, and it made Triton type `pid * BLOCK` as `int * pointer<fp32>`.
_POINTER_HINTS = (
    "ptr", "input", "output", "weight", "bias", "src", "dst", "grad", "acc",
    "result", "tensor", "buffer",
)

#: Whole names that mean an operand, including the one-letter spellings that are
#: only safe as complete names (`X`, `A`, `B`, `C`) rather than as substrings.
_POINTER_NAMES = frozenset(
    {
        "a", "b", "c", "x", "y", "z", "in", "out", "inp", "outp", "res",
        "ret", "src", "dst", "lhs", "rhs", "left", "right", "self", "other",
    }
)

#: Constexpr names and the value to bind them to when a kernel leaves the choice
#: to the caller. Block sizes follow the extraction tile so a bridged kernel and
#: an extracted one of the same shape agree.
_CONSTEXPR_VALUES: Mapping[str, Any] = {
    "BLOCK": 64, "BLOCK_SIZE": 64, "BLOCK_N": 64, "BLOCK_M": 64,
    "BM": 64, "BN": 64, "BK": 32, "BLOCK_K": 32,
    "HAS_BIAS": True, "USE_BIAS": True, "IS_BIAS": True,
    "num_warps": 4, "num_stages": 1,
}


class FlagGemsUnsupported(RuntimeError):
    """FlagGems is present but this op cannot be turned into TTIR.

    The message names the parameter or the missing convention, because "the
    bridge did not work" is not a reason a reader can act on.
    """


@dataclass(frozen=True)
class FlagGemsCapability:
    """Whether FlagGems can be used here, and which of our ops it implements."""

    available: bool
    reason: str
    version: str | None = None
    resolved_ops: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return self.available


_CAPABILITY: FlagGemsCapability | None = None


def _probe_version(module: Any) -> str | None:
    for attr in ("__version__", "version"):
        value = getattr(module, attr, None)
        if isinstance(value, str) and value:
            return value
    return None


def capability(*, refresh: bool = False) -> FlagGemsCapability:
    """Probe for an importable `flag_gems` and list the ops it implements."""
    global _CAPABILITY
    if _CAPABILITY is not None and not refresh:
        return _CAPABILITY
    try:
        import flag_gems  # type: ignore[import-not-found]
    except Exception as exc:
        _CAPABILITY = FlagGemsCapability(
            available=False,
            reason=f"flag_gems is not importable: {type(exc).__name__}: {exc}",
        )
        return _CAPABILITY
    found = tuple(sorted(name for name in OP_KERNELS if resolve_kernel(name) is not None))
    reason = (
        f"flag_gems resolves {len(found)} of the {len(OP_KERNELS)} op names this bridge knows"
        if found
        else "flag_gems imported but none of the bridged op names resolved to a kernel"
    )
    _CAPABILITY = FlagGemsCapability(
        available=bool(found),
        reason=reason,
        version=_probe_version(flag_gems),
        resolved_ops=found,
    )
    return _CAPABILITY


def is_available() -> bool:
    """Whether the bridge can resolve anything. Never raises."""
    return capability().available


def implemented_ops() -> tuple[str, ...]:
    """The op names this bridge can resolve inside the installed FlagGems."""
    return capability().resolved_ops


def _unwrap(obj: Any) -> Any:
    """Peel the decorator wrappers off a kernel until a `JITFunction` shows.

    Bounded on purpose: a cycle in `__wrapped__` would otherwise be a hang, and
    "the wrapper chain ends somewhere that is not a Triton kernel" is a refusal
    with a name, not a stack overflow.
    """
    seen: set[int] = set()
    current = obj
    for _ in range(8):
        if current is None or id(current) in seen:
            return None
        seen.add(id(current))
        if hasattr(current, "params") and hasattr(current, "fn"):
            return current  # a triton JITFunction
        for attr in ("fn", "__wrapped__", "kernel", "jit_fn", "func"):
            inner = getattr(current, attr, None)
            if inner is not None and inner is not current and callable(inner):
                current = inner
                break
        else:
            return None
    return None


def resolve_kernel(op_name: str, attr: str | None = None) -> Any | None:
    """The `@triton.jit` kernel implementing `op_name`, or `None`.

    `attr` pins a specific spelling; otherwise every spelling in
    :data:`OP_KERNELS` is tried against `flag_gems.ops` first and the package
    root second.
    """
    try:
        import flag_gems  # type: ignore[import-not-found]
    except Exception:
        return None
    ops_ns = getattr(flag_gems, "ops", None)
    names = (attr,) if attr else OP_KERNELS.get(op_name, ())
    for name in names:
        for namespace in (ops_ns, flag_gems):
            if namespace is None:
                continue
            kernel = _unwrap(getattr(namespace, name, None))
            if kernel is not None:
                return kernel
    return None


def triton_source(op_name: str) -> str | None:
    """The Python source of the resolved kernel, or `None` when unavailable.

    Source is what makes the bridge possible at all: a compiled Triton kernel
    cannot be re-signed, so a *different* signature has to be compiled from the
    same text.
    """
    kernel = resolve_kernel(op_name)
    if kernel is None:
        return None
    try:
        return inspect.getsource(kernel.fn)
    except (OSError, TypeError):  # pragma: no cover - depends on install layout
        return None


# --------------------------------------------------------------------------- #
# Signature and environment mapping
# --------------------------------------------------------------------------- #


def _is_constexpr(kernel: Any, name: str) -> bool:
    """Whether a parameter is a `tl.constexpr` in this kernel.

    Read from `JITFunction.params[i].is_constexpr`, which is the compiler's own
    per-parameter flag. Two plausible-looking sources are wrong, and one of them
    was actually used: a `JITFunction`'s `__annotations__` is a dict about
    *Triton's* internals — `{'run': 'T'}` on 3.8.0 — so reading it first classified
    every `tl.constexpr` as a runtime value and then typed it as a pointer. The
    Python function's annotations are a decent second source and are used only if
    the parameter list is unavailable.
    """
    for param in getattr(kernel, "params", ()) or ():
        if str(getattr(param, "name", "")) == name:
            return bool(getattr(param, "is_constexpr", False))
    annotations = getattr(getattr(kernel, "fn", None), "__annotations__", None) or {}
    annotation = annotations.get(name)
    if annotation is None:
        return False
    text = getattr(annotation, "__name__", None) or str(annotation)
    return "constexpr" in text.lower()


def _parameter_names(kernel: Any) -> tuple[str, ...]:
    params = getattr(kernel, "params", None)
    if params:
        return tuple(
            str(getattr(param, "name", param)) for param in params
        )
    try:  # pragma: no cover - fallback for a JITFunction without `params`
        return tuple(inspect.signature(kernel.fn).parameters)
    except (TypeError, ValueError):
        return ()


def _defaults(kernel: Any) -> Mapping[str, Any]:
    try:
        signature = inspect.signature(kernel.fn)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return {}
    return {
        name: parameter.default
        for name, parameter in signature.parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }


def _looks_like_pointer(name: str) -> bool:
    """Whether a parameter name means a tensor operand.

    A whole-name match, or a match on a substring that does not appear in the
    scalar vocabulary (`ptr`, `input`, `weight`, ...). Deliberately not a
    substring test against single letters.
    """
    low = name.lower()
    if low in _POINTER_NAMES:
        return True
    return any(hint in low for hint in _POINTER_HINTS)


def _looks_like_extent(name: str) -> bool:
    low = name.lower()
    return low in ("m", "n", "k") or "stride" in low


def _unmappable(name: str, op_name: str) -> FlagGemsUnsupported:
    return FlagGemsUnsupported(
        f"flag_gems.{op_name}: parameter {name!r} could not be mapped to an operand, an "
        "extent, a stride or a known constexpr. The bridge refuses rather than guess a "
        "signature, because a kernel compiled against the wrong one computes the wrong "
        "answer without failing."
    )


def _map_signature(
    kernel: Any,
    op_name: str,
    shapes: Sequence[Sequence[int]],
    *,
    has_bias: bool | None,
) -> tuple[dict[str, str], dict[str, Any]]:
    """`(signature, constexprs)` for the kernel, or a refusal naming a parameter."""
    names = _parameter_names(kernel)
    if not names:
        raise FlagGemsUnsupported(
            f"flag_gems.{op_name}: the kernel exposes no parameter list, so no signature "
            "can be built for it"
        )
    defaults = _defaults(kernel)
    signature: dict[str, str] = {}
    constexprs: dict[str, Any] = {}
    for name in names:
        if _is_constexpr(kernel, name):
            if name.upper() in _CONSTEXPR_VALUES:
                constexprs[name] = _CONSTEXPR_VALUES[name.upper()]
                continue
            if name in defaults and isinstance(defaults[name], (int, bool)):
                constexprs[name] = defaults[name]
                continue
            raise _unmappable(name, op_name)
        if _looks_like_pointer(name):
            signature[name] = "*fp32"
            continue
        if _looks_like_extent(name) or name.lower() in _FLAT_NAMES:
            signature[name] = "i32"
            continue
        raise _unmappable(name, op_name)
    if not any(value == "*fp32" for value in signature.values()):
        raise FlagGemsUnsupported(
            f"flag_gems.{op_name}: the mapped signature has no pointer parameter, so there "
            "is nothing for the op's tensors to bind to"
        )
    return signature, constexprs


def _map_env(
    signature: Mapping[str, str],
    op_name: str,
    shapes: Sequence[Sequence[int]],
    tile: tuple[int, int, int],
) -> tuple[dict[str, int], tuple[int, int, int], tuple[int, int, int]]:
    """`(env, problem, padded)` for the mapped parameter names.

    Matmul-family ops get `M`/`N`/`K` and the padded contiguous strides, matching
    what Path 1 compiles against. Elementwise ops get the padded flat extent,
    because the pipeline's 1-D buffer derivation reads it from there. A parameter
    the caller cannot supply (an extent with no shape to read it from) is a
    refusal.
    """
    bm, bn, bk = tile
    is_matmul = op_name in ("mm", "matmul", "linear")
    env: dict[str, int] = {}
    if is_matmul:
        two = [tuple(s) for s in shapes if len(s) == 2]
        if len(two) < 2:
            raise FlagGemsUnsupported(
                f"flag_gems.{op_name}: matmul-family extraction needs two 2-D operands, "
                f"got {[tuple(s) for s in shapes]}"
            )
        (m, k), (k2, n) = two[0], two[1]
        if k != k2:
            raise FlagGemsUnsupported(
                f"flag_gems.{op_name}: inner dimensions disagree ({k} vs {k2})"
            )
        rows = max(bm, math.ceil(m / bm) * bm)
        inner = max(bk, math.ceil(k / bk) * bk)
        cols = max(bn, math.ceil(n / bn) * bn)
        env.update({
            "M": rows, "N": cols, "K": inner,
            "%M": rows, "%N": cols, "%K": inner,
            "sam": inner, "sak": 1, "sbk": cols, "sbn": 1, "scm": cols, "scn": 1,
            "%sam": inner, "%sak": 1, "%sbk": cols, "%sbn": 1, "%scm": cols, "%scn": 1,
        })
        problem, padded = (m, k, n), (rows, inner, cols)
    else:
        if not shapes:
            raise FlagGemsUnsupported(f"flag_gems.{op_name}: no operand shape was supplied")
        flat = 1
        for dim in shapes[0]:
            flat *= int(dim)
        padded_flat = max(64, math.ceil(flat / 64) * 64)
        env.update({
            "n": flat, "N": padded_flat, "M": padded_flat,
            "%n": flat, "%__flat_width__": padded_flat, "%__block__": 64,
        })
        problem, padded = (flat, 0, 0), (padded_flat, 0, 0)
    for name, kind in signature.items():
        if kind != "i32" or name in env:
            continue
        lowered = name.lower()
        if lowered in _FLAT_NAMES and "n" not in env:
            env[name] = problem[0]
            continue
        # A scalar the env has no source for: refuse rather than default it to 0,
        # because a stride of 0 is a legal-looking value that reads the same row.
        raise FlagGemsUnsupported(
            f"flag_gems.{op_name}: no value is derivable for scalar parameter {name!r}; "
            "supplying one would be inventing a launch, not reading one"
        )
    env.update({"%__tile_m__": bm, "%__tile_n__": bn, "%__tile_k__": bk})
    return env, problem, padded


def extract_for_op(
    op_name: str,
    shapes: Sequence[Sequence[int]],
    *,
    has_bias: bool | None = None,
    tile: tuple[int, int, int] = DEFAULT_TILE,
) -> Extracted | None:
    """Extract TTIR for a FlagGems-implemented op, or `None` when it cannot.

    `None` means *FlagGems does not implement it* (or is not installed) — which
    the seam records as a reason and moves past. :class:`FlagGemsUnsupported`
    means FlagGems implements it and the bridge cannot sign it, which is a
    different fact and is reported as its own reason. Neither is ever a silent
    success.
    """
    global _CAPABILITY
    try:
        import flag_gems  # type: ignore[import-not-found]  # noqa: F401
    except Exception:
        return None
    kernel = resolve_kernel(op_name)
    if kernel is None:
        return None
    signature, constexprs = _map_signature(kernel, op_name, shapes, has_bias=has_bias)
    env, problem, padded = _map_env(signature, op_name, shapes, tile)
    ttir = compile_triton_kernel(kernel, signature, constexprs)
    return Extracted(
        name=f"flaggems_{op_name}",
        kind="matmul" if op_name in ("mm", "matmul", "linear") else "elementwise",
        op=op_name,
        ttir=ttir,
        env=env,
        tile=tile if op_name in ("mm", "matmul", "linear") else (64, 1, 1),
        problem=problem,
        padded=padded,
        has_bias=bool(has_bias),
    )
