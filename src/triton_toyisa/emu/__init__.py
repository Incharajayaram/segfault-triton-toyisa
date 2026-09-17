"""emu: NumPy execution of the emitted stream, with a derived tolerance policy.

No NumPy dtype magic is hidden here: the precision policy is explicit and
recorded per dtype (FR-021, FR-022).

The C++ backend (_emu_cpp) is preferred when available. It is an optional
accelerator: the pure-Python path stays untouched and is used as fallback.
"""

try:
    from ._emu_cpp import HAS_CPP as HAS_CPP  # noqa: F401
except ImportError:
    HAS_CPP = False
