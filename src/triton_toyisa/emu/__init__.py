"""emu: NumPy execution of the emitted stream, with a derived tolerance policy.

No NumPy dtype magic is hidden here: the precision policy is explicit and
recorded per dtype (FR-021, FR-022).
"""

try:
    from triton_toyisa.emu import _emu_cpp  # noqa: F401
    HAS_CPP = True
except ImportError:
    HAS_CPP = False

from .hardware import BankConflictModel, BankConflictUnit, CoalescingUnit, HardwarePerformanceStats
from .tcu import TcuEmulator, TcuPerformanceCounters
from .dxa import DxaEmulator, DxaDescriptor, DxaPerformanceCounters

__all__ = [
    "HAS_CPP",
    "BankConflictModel",
    "BankConflictUnit",
    "CoalescingUnit",
    "HardwarePerformanceStats",
    "TcuEmulator",
    "TcuPerformanceCounters",
    "DxaEmulator",
    "DxaDescriptor",
    "DxaPerformanceCounters",
]
