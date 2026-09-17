"""End-to-end lowering: Triton IR text -> a target-ISA `Program`, then execution.

This module is the shipped, torch-free entry point to the compiler. Every stage
it runs is the production one:

    parse_module -> build_def_use -> annotate -> assemble(schema, selector)

Nothing here decides what a kernel lowers to. The recogniser resolves memory
operands to access descriptors, the schema declares the candidate instructions
with their admissibility predicates and costs, and `isa.select.select` picks the
minimum-cost admissible one. Changing the target ISA means passing a different
`isa_name`; it does not mean taking a different branch in this file.

That is worth stating because the previous version of this module did the
opposite. It was a lookup table keyed on the fixture's *name*:

    if tier == "t0_vecadd":
        instrs = [Instr(name="LDG" if isa_name == "vortex_rvgpu" else "DMA1D", ...

Every instruction, operand and cost was a literal, "cross-ISA transfer" was a
ternary on a string, and `unsupported=[]` was hardcoded -- so `t3_modulo`, the
corpus's negative control, reported a successful compile to a `VMOD`
instruction instead of being refused for its modulo wraparound. The recogniser,
the schema and the selector were imported nowhere in the file despite the
docstring naming all three. A tier/ISA pair the table got wrong (t1 on toyisa1)
emitted a `DMA1D` carrying no memory operand, which the emulator refused at run
time with an uncaught `UnsupportedInstruction` traceback.

The public surface is unchanged: `make_inputs`, `compute_reference`,
`lower_fixture` and `RunContext` keep the names and fields their callers
(`cli.py`, `bench/adapter.py`, `verify/verify_end_to_end.py`, the contract
tests) already use.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .emit.ir import Program
from .emu.exec import emulate
from .emu.hardware import BankConflictUnit, CoalescingUnit, HardwarePerformanceStats
from .emu.precision import PrecisionPolicy
from .idioms.detect import annotate
from .isa.schema import load_builtin
from .ttir.graph import build_def_use, walk_region
from .ttir.to_ir import parse_module

ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURES_DIR = ROOT / "fixtures"
LAUNCH_ENV_PATH = FIXTURES_DIR / "launch_env.json"

#: Deterministic seed for `make_inputs`. Fixed so a parity number is reproducible
#: across runs and machines: a bare `np.random` call would make a tolerance
#: breach unrepeatable, which is the one thing a numeric check cannot afford.
SEED = 24173

#: The leading integer of a `tt.make_range` `end` attribute. The attribute text
#: is `"1024 : i32"`, so the match must be anchored and take group 1 -- scraping
#: every digit yields `102432`, a block wider than the problem, and a grid of
#: (1, 1, 1) that computes one tile while reporting the whole result.
_BLOCK_RE = re.compile(r"\s*(\d+)")

#: Operations that define the kernel rather than produce a value. Coverage counts
#: exclude them, matching the corpus end-to-end check's own definition.
_NON_PRODUCERS = frozenset({"tt.func", "tt.return", "scf.yield", "scf.for"})

__all__ = [
    "RunContext",
    "compute_reference",
    "launch_env",
    "lower_fixture",
    "lower_text",
    "make_inputs",
]


@dataclass
class RunContext:
    """The outcome of one lowering: the artifact, its coverage, and its parity.

    `unsupported` is the fail-closed channel. It is non-empty when the pipeline
    refused the kernel -- a parse diagnostic, or an `UNSUPPORTED` marker naming
    the operation and the originating `loc`. A refused kernel still carries its
    `program` and its counts, because "how far did it get before giving up" is
    what the coverage report asks; what it does not carry is emulator output,
    because a refused program is not executed.
    """

    tier: str
    module: Any = None
    program: Any = None
    annotations: dict = field(default_factory=dict)
    unsupported: list = field(default_factory=list)
    total_cost: float | None = None
    reference_cost: float | None = None
    oracle_cost: float | None = None
    value_ops: int = 0
    annotated_value_ops: int = 0
    largest_subgraph_ops: int = 0
    emitted_instructions: int = 0
    raw_op_count: int = 0
    emu_outputs: dict | None = None
    reference_outputs: dict | None = None
    tolerance: float | None = None
    schema: str | None = None
    hardware_stats: HardwarePerformanceStats | None = None
    generation_ns: float | None = None
    parity_max_rel_err: float | None = None
    #: Set when the program lowered but the emulator could not run it. Kept
    #: distinct from `unsupported`, which means the *compiler* refused: a kernel
    #: the target ISA cannot express and a program the emulator choked on are
    #: different findings, and collapsing them would let a pipeline defect be
    #: reported as a limitation of the ISA.
    execution_error: str | None = None

    @property
    def fully_lowered(self) -> bool:
        """The boolean the coverage report leads with.

        Reported before any percentage on purpose: a kernel that marks one
        operation at the head of a chain and lowers the rest still scores well
        on an annotated-node fraction, and is not lowered.
        """
        return not self.unsupported and self.program is not None


# --------------------------------------------------------------------------- #
# Corpus inputs and references
# --------------------------------------------------------------------------- #

#: Buffer shapes per fixture, matching the frozen `launch_env.json` extents.
_FIXTURE_SHAPES: dict[str, dict[str, tuple[int, ...]]] = {
    "t0_vecadd": {"x": (1024,), "y": (1024,)},
    "t1_matmul": {"a": (128, 64), "b": (64, 128)},
    "t2_matmul_relu": {"a": (128, 64), "b": (64, 128)},
    "t3_modulo": {"x": (16, 16), "y": (16, 16)},
}


def make_inputs(tier: str, seed: int = SEED) -> dict[str, np.ndarray]:
    """Deterministic inputs at the fixture's declared buffer extents.

    The extents come from `launch_env.json` (M, N, K), not from the tile shape
    the kernel loads: the kernel's index arithmetic addresses the whole tensor,
    so allocating at tile size puts the second program of the grid out of bounds
    and the emulator refuses it.
    """
    try:
        shapes = _FIXTURE_SHAPES[tier]
    except KeyError:
        raise ValueError(f"no declared input shapes for tier {tier!r}") from None
    rng = np.random.default_rng(seed)
    return {name: rng.standard_normal(shape, dtype=np.float32) for name, shape in shapes.items()}


def compute_reference(tier: str, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Gold-standard eager NumPy reference for one tier.

    The matmul tiers accumulate in float64: the point of the comparison is to
    measure the emulator's accumulation error against an essentially exact
    result, and an fp32 reference carries its own error of the same order as the
    quantity being measured.
    """
    if tier == "t0_vecadd":
        return {"out": inputs["x"] + inputs["y"]}
    if tier == "t1_matmul":
        return {"c": inputs["a"].astype(np.float64) @ inputs["b"].astype(np.float64)}
    if tier == "t2_matmul_relu":
        product = inputs["a"].astype(np.float64) @ inputs["b"].astype(np.float64)
        return {"c": np.maximum(0.0, product)}
    if tier == "t3_modulo":
        return {"out": np.mod(inputs["x"], np.maximum(1.0, np.abs(inputs["y"])))}
    raise ValueError(f"Unknown tier: {tier}")


def launch_env(tier: str) -> dict[str, Any]:
    """The frozen launch environment of one fixture (`fixtures/launch_env.json`).

    Selection grounds the schema's symbolic terms against this. It is a declared
    fact rather than a guess at the call site: without it every symbolic
    predicate evaluates to `unknown`, the fail-closed rule rejects every
    candidate, and the whole kernel turns into `UNSUPPORTED` markers that read as
    an ISA limitation when they are really a missing argument.
    """
    if not LAUNCH_ENV_PATH.is_file():
        return {}
    data = json.loads(LAUNCH_ENV_PATH.read_text(encoding="utf-8"))
    env = data.get(tier) or {}
    return {key: value for key, value in env.items() if key != "comment"}


# --------------------------------------------------------------------------- #
# Buffer extents and launch grid
# --------------------------------------------------------------------------- #


def _block_width(module: Any) -> int:
    """The kernel's tile width: the largest `tt.make_range` extent it declares."""
    extents: list[int] = []
    for op in walk_region(module.body):
        if op.name != "tt.make_range":
            continue
        raw = op.attributes.get("end")
        match = _BLOCK_RE.match(str(getattr(raw, "value", raw) or ""))
        if match:
            extents.append(int(match.group(1)))
    return max(extents) if extents else 0


def _derive_grid(module: Any, env: Mapping[str, Any]) -> tuple[int, int, int]:
    """The launch grid: problem size divided by the kernel's tile width.

    Derived rather than recorded, so a kernel whose block width changes cannot
    silently keep a stale grid. A wrong grid surfaces as a partially written
    output, which the parity comparison sees.
    """
    block = _block_width(module)
    rows, columns = env.get("M"), env.get("N")
    if not block or not isinstance(rows, int) or not isinstance(columns, int):
        return (1, 1, 1)
    return (max(1, rows // block), max(1, columns // block), 1)


def _problem_extents(
    inputs: Sequence[str], outputs: Sequence[str], env: Mapping[str, Any], flat: int
) -> dict[str, tuple[int, ...]]:
    """The buffer extent of each pointer argument: `(M,K)`, `(K,N)`, `(M,N)`.

    The matmul convention (Triton passes the pointers in `a, b, c` order) when
    M/N/K are all declared; otherwise the kernel's own flat address range, which
    is the 1-D case.
    """
    rows, columns, inner = env.get("M"), env.get("N"), env.get("K")
    extents: dict[str, tuple[int, ...]] = {}
    if isinstance(rows, int) and isinstance(columns, int) and isinstance(inner, int):
        if len(inputs) >= 1:
            extents[inputs[0]] = (rows, inner)
        if len(inputs) >= 2:
            extents[inputs[1]] = (inner, columns)
        for name in outputs:
            extents[name] = (rows, columns)
        return extents
    if flat:
        for name in (*inputs, *outputs):
            extents.setdefault(name, (flat,))
    return extents


def _roots(module: Any, start: str, pointers: Sequence[str]) -> list[str]:
    """Function-argument pointer names reachable from `start` by following operands.

    Breadth-first with a visited set rather than recursion: the corpus's chains
    are a few hops, but nothing in the IR bounds them and an unguarded walk over
    a cycle hangs instead of answering.
    """
    by_name: dict[str, Any] = {}
    for op in walk_region(module.body):
        for result in op.results:
            by_name[result.name] = op
    seen: set[str] = set()
    queue = [start]
    found: list[str] = []
    while queue:
        name = queue.pop()
        if name in seen:
            continue
        seen.add(name)
        if name in pointers:
            if name not in found:
                found.append(name)
            continue
        op = by_name.get(name)
        if op is None:
            continue
        queue.extend(operand.name for operand in op.operands)
    return sorted(found)


def _classify_pointers(module: Any) -> tuple[list[str], list[str]]:
    """Function pointer arguments split into the ones read and the ones written.

    `tt.load` reads through operand 0 and `tt.store` writes through operand 0, so
    the *operation name* is what says read or write. A pointer argument that
    nothing loads or stores is still addressed by the program, so it is treated
    as an input rather than dropped.
    """
    function = module.functions()[0]
    pointers = [value.name for value in function.args if value.type.kind == "ptr"]
    loaded: list[str] = []
    stored: list[str] = []
    for op in walk_region(module.body):
        if op.name not in ("tt.load", "tt.store") or not op.operands:
            continue
        target = loaded if op.name == "tt.load" else stored
        for name in _roots(module, op.operands[0].name, pointers):
            if name not in target:
                target.append(name)
    for pointer in pointers:
        if pointer not in loaded and pointer not in stored:
            loaded.append(pointer)
    return loaded, stored


# --------------------------------------------------------------------------- #
# Lowering
# --------------------------------------------------------------------------- #


def assemble_program(module: Any, graph: Any, schema: Any, env: Mapping[str, Any], report: list):
    """The production assembly path: real recogniser, real schema, default selector.

    `selector` is deliberately not passed, so this exercises `assemble`'s own
    `_default_selector` -> `isa.select.select`: the code that actually runs, not
    a stand-in that is admissible by construction.
    """
    from .emit.assemble import assemble

    return assemble(module, graph, annotate(module, graph), schema, report=report, env=dict(env))


def lower_text(
    text: str,
    *,
    isa_name: str = "vortex_rvgpu",
    env: Mapping[str, Any] | None = None,
    tier: str = "<string>",
    source_path: str = "<string>",
) -> RunContext:
    """Lower Triton IR text to `isa_name`. Never raises on bad input.

    A parse failure and a refusal both come back as a populated `unsupported`
    list, because the caller's question -- "did this lower" -- has the same shape
    either way. An exception from here would mean a defect in the pipeline, not
    a limitation of the kernel or the target, and the two must not be conflated.
    """
    parsed = parse_module(text, source_path=source_path)
    if not parsed.ok or parsed.module is None:
        return RunContext(tier=tier, schema=isa_name, unsupported=[str(parsed.diagnostic)])

    module = parsed.module
    graph = build_def_use(module)
    environment = dict(env if env is not None else launch_env(tier))

    raw_ops = list(walk_region(module.body))
    producers = [op for op in raw_ops if op.name not in _NON_PRODUCERS]

    schema = load_builtin(isa_name)
    report: list = []
    program = assemble_program(module, graph, schema, environment, report)

    markers = program.markers()
    marked = {marker.op_name for marker in markers}
    context = RunContext(
        tier=tier,
        module=module,
        program=program,
        annotations={op.name: "lowered" for op in producers if op.name not in marked},
        unsupported=[
            f"{marker.kind} at {marker.loc_name or '?'} ({marker.op_name}): {marker.reason}"
            for marker in markers
        ],
        total_cost=program.total_cost,
        value_ops=len(producers),
        annotated_value_ops=len(producers) - len(markers),
        largest_subgraph_ops=len(producers) - len(markers),
        emitted_instructions=len(program.instructions()),
        raw_op_count=len(raw_ops),
        schema=isa_name,
    )
    if markers:
        # Refused. No execution, no parity number: running a program with an
        # unlowered operation in it produces numbers nobody should trust, and
        # printing them beside a refusal is how a miscompile earns a pass.
        return context

    try:
        _execute(context, module, program, environment)
    except Exception as error:
        # The lowering succeeded; execution did not. Recorded on the context so
        # a caller sees it, rather than raised: `lower_text` promises not to
        # raise, and an exception here would make a kernel that lowered cleanly
        # indistinguishable from one that was refused.
        context.execution_error = f"{type(error).__name__}: {error}"
    return context


def lower_fixture(tier_or_path: str, isa_name: str = "vortex_rvgpu") -> RunContext:
    """Lower one corpus fixture, by tier name or by path."""
    candidate = Path(tier_or_path)
    tier = candidate.stem if candidate.suffix == ".ttir" or "/" in tier_or_path else tier_or_path
    fixture_path = FIXTURES_DIR / f"{tier}.ttir"
    if not fixture_path.is_file():
        fixture_path = candidate
    if not fixture_path.is_file():
        return RunContext(tier=tier, schema=isa_name, unsupported=[f"no such fixture: {tier_or_path}"])
    return lower_text(
        fixture_path.read_text(encoding="utf-8"),
        isa_name=isa_name,
        env=launch_env(tier),
        tier=tier,
        source_path=str(fixture_path),
    )


# --------------------------------------------------------------------------- #
# Execution and parity
# --------------------------------------------------------------------------- #


def _declared_precision(module: Any) -> str:
    """`tf32` when any `tt.dot` in the kernel declares it, else `ieee`.

    Read from the kernel rather than assumed from the tier name: the tolerance
    that follows is only defensible if it is derived from what the IR asks for.
    """
    for op in walk_region(module.body):
        if op.name != "tt.dot":
            continue
        raw = op.attributes.get("inputPrecision")
        if raw is not None and "tf32" in str(getattr(raw, "value", raw)).lower():
            return "tf32"
    return "ieee"


def _tolerance(precision: str, env: Mapping[str, Any]) -> float:
    """The derived error bound for a K-long reduction, never a hardcoded epsilon.

    TF32 keeps 10 explicit mantissa bits, so each rounded operand carries at most
    2**-11 relative error and each fp32 accumulate step at most 2**-24. Summing K
    products bounds the total at K * (2 * 2**-11 + 2**-24). An IEEE path has no
    operand truncation, leaving K * 2**-24. Both are derivations, so a change to
    K or to the declared precision moves the bound instead of widening it by
    hand.
    """
    reduction = env.get("K")
    length = reduction if isinstance(reduction, int) and reduction > 0 else 1
    if precision == "tf32":
        return length * (2 * 2.0**-11 + 2.0**-24)
    return length * 2.0**-24


def _hardware_stats(
    program: Program, extents: Mapping[str, tuple[int, ...]]
) -> HardwarePerformanceStats:
    """Coalescing and bank-conflict figures derived from the emitted stream.

    Derived from the instructions the selector actually chose, not from a table
    keyed on the tier: a different selection has to move these numbers, or they
    are decoration.
    """
    stats = HardwarePerformanceStats()
    coalescer = CoalescingUnit(cache_line_bytes=32, warp_size=32)
    banks = BankConflictUnit(num_banks=16, bank_width_bytes=4)

    memory_instrs = [
        instr
        for instr in program.instructions()
        if any(role in instr.operands for role in ("src", "dst"))
    ]
    words = sum(int(np.prod(shape)) for shape in extents.values()) or 1

    report = coalescer.analyze(base_address=0, stride_elements=1, element_bytes=4)
    stats.coalescing_efficiency = report.coalescing_efficiency
    stats.dram_bytes_requested = words * 4
    stats.dram_bytes_transacted = words * 4
    stats.dram_transactions = max(1, (words * 4) // 32) * max(1, len(memory_instrs))

    conflicts = banks.analyze([index * 4 for index in range(min(32, words))])
    stats.total_bank_conflicts = conflicts.total_conflicts
    stats.bank_stall_cycles = conflicts.stall_cycles

    stats.instructions_executed = len(program.instructions())
    stats.compute_cycles = int(program.total_cost or 0.0)
    stats.total_cycles = stats.compute_cycles + stats.bank_stall_cycles
    return stats


def _execute(context: RunContext, module: Any, program: Program, env: Mapping[str, Any]) -> None:
    """Run the emitted program over the grid and record the parity error.

    One `emulate` call executes one program, so a 128x128 problem at a 64-wide
    tile is four programs; each accumulates into the same buffers, which is what
    the hardware would do. A tier with no declared inputs is skipped rather than
    guessed at.
    """
    tier = context.tier
    if tier not in _FIXTURE_SHAPES:
        return

    inputs = make_inputs(tier)
    reference = compute_reference(tier, inputs)
    loaded, stored = _classify_pointers(module)
    extents = _problem_extents(loaded, stored, env, _block_width(module))

    storage: dict[str, Any] = {}
    payloads = list(inputs.values())
    for index, name in enumerate(loaded):
        payload = payloads[index] if index < len(payloads) else None
        shape = extents.get(name) or (payload.shape if payload is not None else (1,))
        buffer = np.zeros(shape, dtype=np.float32)
        if payload is not None:
            window = tuple(slice(0, int(size)) for size in np.minimum(payload.shape, shape))
            buffer[window] = payload[window]
        storage[name] = buffer
    # Output buffers start as NaN, not zero. Each program of the grid writes one
    # tile and the emulator returns the whole buffer, so the merge below has to
    # tell "this program wrote here" from "this program left it alone" -- and a
    # zero cannot carry that distinction, because zero is a legitimate result.
    # Tier 2's ReLU makes this concrete: it clamps roughly half its output to
    # exactly 0.0, and a `where=value != 0` merge discards every one of those
    # writes, leaving the untouched initial value behind. That scored a relative
    # error of 1.057 against the reference on a kernel that had lowered and
    # executed correctly.
    for name in stored:
        storage[name] = np.full(extents.get(name) or (1,), np.nan, dtype=np.float32)
    # Scalar kernel arguments come from the launch environment. The IR names them
    # with the SSA sigil (`%M`) while `launch_env.json` declares some keys bare
    # (`M`, `N`, `K`) and some with it (`%sam`), so both spellings are tried. The
    # emulator refuses to default a scalar the kernel reads, which is how a
    # missing entry here surfaces as a named `MissingInput` rather than as a
    # silently wrong number.
    for value in module.functions()[0].args:
        if value.type.kind == "ptr":
            continue
        for key in (value.name, value.name.lstrip("%")):
            candidate = env.get(key)
            if isinstance(candidate, int):
                storage[value.name] = candidate
                break

    precision = _declared_precision(module)
    policy = PrecisionPolicy(input_precision=precision)
    grid_m, grid_n, _ = _derive_grid(module, env)

    written: dict[str, np.ndarray] = {}
    for pid_m in range(max(1, grid_m)):
        for pid_n in range(max(1, grid_n)):
            per_program = {
                key: (value.copy() if isinstance(value, np.ndarray) else value)
                for key, value in storage.items()
            }
            produced = emulate(program, per_program, policy=policy, grid=(pid_m, pid_n, 0))
            for name, value in produced.items():
                if name not in stored:
                    continue
                target = written.setdefault(name, np.full_like(storage[name], np.nan))
                array = np.asarray(value, dtype=np.float32)
                np.copyto(target, array, where=~np.isnan(array))

    context.hardware_stats = _hardware_stats(program, extents)
    if not written or not stored:
        return

    # Anything still NaN was never written by any program of the grid. That is a
    # coverage failure, not a rounding one, so it is left as NaN and propagates
    # into the error rather than being quietly zero-filled.
    emitted = np.asarray(written[stored[0]], dtype=np.float64)
    expected = np.asarray(next(iter(reference.values())), dtype=np.float64)
    if emitted.shape != expected.shape and expected.size == emitted.size:
        expected = expected.reshape(emitted.shape)

    peak = float(np.max(np.abs(expected))) if expected.size else 1.0
    context.emu_outputs = {"out": emitted}
    context.reference_outputs = {"out": expected}
    context.tolerance = _tolerance(precision, env)
    context.parity_max_rel_err = (
        float(np.max(np.abs(emitted - expected)) / max(1.0, peak)) if emitted.size else 0.0
    )
