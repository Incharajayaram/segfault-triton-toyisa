"""Hardware performance modeling: memory coalescing, banked scratchpad conflict analysis,

and cycle estimation inspired by open-source RISC-V GPGPUs (Vortex / RV-GPGPU).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class CoalescingReport:
    """Analysis report for a warp memory request."""

    active_threads: int
    element_bytes: int
    requested_bytes: int
    transacted_bytes: int
    num_transactions: int
    coalescing_efficiency: float
    is_fully_coalesced: bool


class CoalescingUnit:
    """Models global memory coalescing in a GPGPU memory controller.

    Intercepts memory access requests from threads in a warp and merges
    adjacent accesses falling into the same cache line / burst sector.
    Default line size: 32 bytes (or 64 bytes).
    """

    def __init__(self, cache_line_bytes: int = 32, warp_size: int = 32) -> None:
        self.cache_line_bytes = cache_line_bytes
        self.warp_size = warp_size

    def analyze(
        self,
        base_address: int,
        stride_elements: int,
        element_bytes: int = 4,
        active_mask: int | None = None,
    ) -> CoalescingReport:
        """Analyze memory access pattern for a warp."""
        if active_mask is None:
            active_threads = self.warp_size
            thread_indices = list(range(self.warp_size))
        else:
            thread_indices = [i for i in range(self.warp_size) if (active_mask & (1 << i))]
            active_threads = len(thread_indices)

        if active_threads == 0:
            return CoalescingReport(
                active_threads=0,
                element_bytes=element_bytes,
                requested_bytes=0,
                transacted_bytes=0,
                num_transactions=0,
                coalescing_efficiency=1.0,
                is_fully_coalesced=True,
            )

        requested_bytes = active_threads * element_bytes
        sectors_accessed: set[int] = set()

        for tid in thread_indices:
            addr = base_address + (tid * stride_elements * element_bytes)
            sector_id = addr // self.cache_line_bytes
            sectors_accessed.add(sector_id)

        num_transactions = len(sectors_accessed)
        transacted_bytes = num_transactions * self.cache_line_bytes
        efficiency = min(1.0, requested_bytes / max(1, transacted_bytes))
        is_fully_coalesced = num_transactions == math.ceil(requested_bytes / self.cache_line_bytes)

        return CoalescingReport(
            active_threads=active_threads,
            element_bytes=element_bytes,
            requested_bytes=requested_bytes,
            transacted_bytes=transacted_bytes,
            num_transactions=num_transactions,
            coalescing_efficiency=efficiency,
            is_fully_coalesced=is_fully_coalesced,
        )


@dataclass(frozen=True)
class BankConflictReport:
    """Analysis report for multi-banked scratchpad / LDS memory access."""

    num_banks: int
    bank_width_bytes: int
    max_conflict_degree: int
    total_conflicts: int
    stall_cycles: int
    bank_access_counts: dict[int, int]


class BankConflictUnit:
    """Models multi-banked scratchpad (shared memory / LDS) bank arbitration.

    Detects bank conflicts when multiple threads within a warp simultaneously access
    different addresses located within the same memory bank.
    Formula: bank_id = (address // bank_width_bytes) % num_banks.
    """

    def __init__(self, num_banks: int = 16, bank_width_bytes: int = 4, stall_per_conflict: int = 1) -> None:
        self.num_banks = num_banks
        self.bank_width_bytes = bank_width_bytes
        self.stall_per_conflict = stall_per_conflict

    def analyze(self, addresses: Sequence[int]) -> BankConflictReport:
        """Analyze a list of concurrent thread addresses accessing scratchpad memory."""
        # Maps bank_id -> set of unique addresses (accesses to identical address broadcast without conflict)
        bank_unique_addrs: dict[int, set[int]] = {b: set() for b in range(self.num_banks)}

        for addr in addresses:
            bank_id = (addr // self.bank_width_bytes) % self.num_banks
            bank_unique_addrs[bank_id].add(addr)

        bank_counts = {b: len(addrs) for b, addrs in bank_unique_addrs.items() if addrs}
        max_conflict = max(bank_counts.values()) if bank_counts else 1
        total_conflicts = sum(max(0, count - 1) for count in bank_counts.values())
        stall_cycles = (max_conflict - 1) * self.stall_per_conflict

        return BankConflictReport(
            num_banks=self.num_banks,
            bank_width_bytes=self.bank_width_bytes,
            max_conflict_degree=max_conflict,
            total_conflicts=total_conflicts,
            stall_cycles=stall_cycles,
            bank_access_counts=bank_counts,
        )


@dataclass
class HardwarePerformanceStats:
    """Complete hardware metrics collected during kernel execution."""

    instructions_executed: int = 0
    compute_cycles: int = 0
    global_memory_cycles: int = 0
    scratchpad_cycles: int = 0
    bank_stall_cycles: int = 0
    total_cycles: int = 0
    dram_bytes_requested: int = 0
    dram_bytes_transacted: int = 0
    dram_transactions: int = 0
    coalescing_efficiency: float = 1.0
    total_bank_conflicts: int = 0


class BankConflictModel:
    """Vortex GPGPU multi-bank scratchpad conflict cost model.

    Evaluates concurrent addresses against 16 banks with 4-byte interleaving.
    Returns the exact number of stall cycles incurred due to bank arbitration.
    """

    def __init__(self, num_banks: int = 16, bank_width_bytes: int = 4) -> None:
        self.num_banks = num_banks
        self.bank_width_bytes = bank_width_bytes

    def analyze_access_pattern(self, addresses: Sequence[int], num_banks: int | None = None) -> int:
        """Return number of stall cycles due to bank conflicts.
        Formula: max_conflicts_per_bank - 1 (or 0 if no accesses).
        """
        if not addresses:
            return 0
        banks = num_banks if num_banks is not None else self.num_banks
        from collections import defaultdict
        bank_accesses: dict[int, int] = defaultdict(int)
        for addr in addresses:
            bank = (addr // self.bank_width_bytes) % banks
            bank_accesses[bank] += 1
        return max(0, max(bank_accesses.values()) - 1)
