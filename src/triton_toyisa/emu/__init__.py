"""emu: NumPy execution of the emitted stream, with a derived tolerance policy.

No NumPy dtype magic is hidden here: the precision policy is explicit and
recorded per dtype (FR-021, FR-022).
"""

# The optional C++ backend. `HAS_CPP` is the single place anything asks whether
# it is available, so a caller never has to guess from an ImportError.
#
# The extension is NOT built by `pip install .`: `pyproject.toml` uses the plain
# setuptools backend, so `CMakeLists.txt` and `emu/cpp/` are never compiled, and
# `emu.exec.emulate` has no `use_cpp` parameter to route to them. The C++ sources
# are therefore present but unwired — see KNOWN_GAPS.md G5. This flag is
# deliberately still defined rather than hardcoded to False, so that wiring the
# build up later is a build-system change and not a source change.
try:  # pragma: no cover - exercised only where the extension is built
    from ._emu_cpp import HAS_CPP as HAS_CPP
except ImportError:
    HAS_CPP = False

__all__ = ["HAS_CPP"]
