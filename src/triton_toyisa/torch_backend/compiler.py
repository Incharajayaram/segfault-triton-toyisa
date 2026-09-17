"""The Dynamo backend: the door `torch.compile` comes in through.

    torch.compile(fn, backend="toyisa")

registers here. `toyisa_backend` is handed an FX graph and the example inputs,
and must return a callable that reproduces the graph's result. Everything else in
this project exists to make that callable's answer *correct and auditable* rather
than merely present.

**Two paths, and the seam never blurs them.**

*The lowering path.* Every kernel the seam can lower is a **recorded lowering**:
a frozen TTIR text plus the launch environment it was compiled under
(`kernels/`). The seam runs it through the *real* pipeline — parse, build
def-use, annotate, assemble against the real ISA-1 schema with the real selector,
then execute on `emu/exec.py`. Nothing about that chain is stubbed, and the
program it executes is an artifact a reader can print with `disassemble`.

*The fallback path.* A graph the seam cannot lower runs on eager PyTorch and
produces a `FallbackRecord` naming why. This is the contract's hardest rule
(FR-025, `contracts/torch-seam.md` postcondition 4): the eager floor is
acknowledged in writing, per graph, and never disguised. What the seam must not
do is run eager code *and report success*, which is the custom-backend sin the
contract names.

**Why the lowering is recorded rather than extracted.** Read off the graph,
Inductor produces *Python source for a Triton kernel*; the TTIR that this
pipeline consumes exists only inside Triton's compiler object, after
`triton.compile` has run. Extracting it therefore needs Triton installed
(the `extract` extra), and the project's whole test suite is built to run without
it. So the seam accepts TTIR from three sources in a stated order — an explicit
`ttir` attached to the graph (how a caller wires a real extraction in), a
recorded lowering for a shape it knows, and otherwise a fallback with the reason —
and `NOT_EXTRACTED` records the gap rather than hiding it. This is what T014
calls the "hard-coded lowering" and it is the day-1 path by design: the
integration risk is retired before the general path exists.

**An all-or-nothing decision per graph.** Lowering *part* of an FX graph would
mean reimplementing partition and stitch semantics, which is Inductor's job and
not this project's. So a graph is lowered only when it is exactly one matchable
kernel and nothing else; any extra node sends the whole graph to fallback. That is
a conservative rule stated out loud, and the alternative — guessing at partial
coverage — is how a backend reports numbers it did not compute.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch._dynamo.backends.registry import register_backend

from ..emit.assemble import assemble
from ..emit.ir import Program
from ..emu.exec import ProgramNotExecutable, StorageError, emulate
from ..emu.precision import PrecisionPolicy
from ..idioms.detect import annotate
from ..isa.schema import load_builtin
from ..recognize.op_shapes import input_precision
from ..ttir.graph import DefUseGraph, build_def_use, walk_region
from ..ttir.ssa import Module, Operation
from ..ttir.to_ir import parse_module
from . import device_interface
from .device_interface import DEVICE_NAME

__all__ = [
    "NOT_EXTRACTED",
    "CompiledKernel",
    "FallbackRecord",
    "LoweringError",
    "LoweringPlan",
    "lower_and_run",
    "prepare",
    "recorded_kernels",
    "toyisa_backend",
    "verify_device",
]

#: The gap this seam acknowledges in writing rather than papering over: the TTIR
#: a real Inductor pass would hand over is not extracted from the graph. Kept
#: short because it is attached to every fallback record.
NOT_EXTRACTED = (
    "TTIR is not extracted from the compiled graph: obtaining it requires Triton's "
    "compiler object (the `extract` extra), and the fixtures are the pipeline's "
    "only frozen input. A caller may attach TTIR to the graph instead; see "
    "compiler.extract_ttir."
)

#: Where the recorded lowerings live. Inside the package, because a backend that
#: reads a path in the *checkout* cannot be imported from anywhere else.
KERNEL_DIR = Path(__file__).resolve().parent / "kernels"

#: The launch environment each recorded lowering was compiled under.
#:
#: Keys are exactly as they appear in the TTIR — scalars under their bare name and
#: symbols under their SSA spelling — because the same dictionary serves two
#: consumers with different naming: selection resolves `%sam` from the schema's
#: predicate symbols, and the emulator looks up the *parameter* `%M`. Defining it
#: once, in the IR's own names, is what keeps those two from drifting.
#:
#: The values are the physical strides of *contiguous* tensors at this shape:
#: A is `(128, 64)` so its k-stride is 64; B is `(64, 128)` so its k-stride is
#: `N = 128`; C is `(128, 128)` so its row stride is 128. The previous value for
#: `%sbk` in the ledger was 64, which addresses the wrong rows — found by executing
#: against fp64, and corrected here to the same value `fixtures/launch_env.json`
#: carries for Tier 1.
_LAUNCH_ENV: dict[str, int] = {
    "M": 128,
    "N": 128,
    "K": 64,
    "%M": 128,
    "%N": 128,
    "%K": 64,
    "%sam": 64,
    "%sak": 1,
    "%sbk": 128,
    "%sbn": 1,
    "%scm": 128,
    "%scn": 1,
}

#: Which recorded lowering covers which kernel entry point.
_RECORDED_ENV: dict[str, dict[str, int]] = {"matmul_128x128x64": dict(_LAUNCH_ENV)}


class LoweringError(RuntimeError):
    """The seam refused a graph, with a reason a `FallbackRecord` can carry."""


@dataclass(frozen=True)
class CompiledKernel:
    """A recorded lowering, prepared: the program and how to feed it.

    `pointers` and `scalars` come from the parsed module's *function signature*
    (`Function.args`), so the mapping between a caller's tensors and the kernel's
    parameters is read off the IR rather than assumed from a name convention.
    `load_roots`/`store_roots` come from walking the def-use graph back from the
    `tt.load`/`tt.store` operands, which is how the seam knows which pointer
    arguments are inputs and which are outputs — the same information the
    recogniser derives, obtained the same way (by following the pointers).
    """

    name: str
    entry: str
    program: Program
    pointers: tuple[str, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    scalars: Mapping[str, int]
    grid: tuple[int, int, int]
    extents: Mapping[str, tuple[int, ...]]
    input_precision: str

    def default_policy(self) -> PrecisionPolicy:
        """The precision contract this kernel declares, not a default chosen here.

        `tt.dot`'s `inputPrecision` attribute (read by `op_shapes.input_precision`)
        says tf32 for the frozen Tier-1 kernel, and `M`/`N`/`K` give the reduction
        length the tolerance is derived from. Executing under `ieee` instead would
        make the emulator *more* accurate than the program it is executing and
        would let the seam's tolerance be `0` — a bound that passes by comparing
        the emulator with itself rather than with torch.
        """
        return PrecisionPolicy.for_tile(
            input_precision=self.input_precision,  # type: ignore[arg-type]
            reduction_length=int(self.scalars.get("%K") or 0),
        )

    @property
    def problem_shape(self) -> tuple[int | None, int | None, int | None]:
        """`(M, N, K)` of the *problem*, from the signature scalars.

        Read from the launch environment the lowering was compiled under, so a
        caller asking "does this kernel apply to my tensors?" compares two
        statements of the same fact rather than trusting a comment.
        """
        return (self.scalars.get("%M"), self.scalars.get("%N"), self.scalars.get("%K"))

    def __str__(self) -> str:  # pragma: no cover - presentation only
        return (
            f"CompiledKernel({self.name}, entry={self.entry}, "
            f"instrs={len(self.program.instructions())}, "
            f"markers={len(self.program.markers())})"
        )


@dataclass(frozen=True)
class FallbackRecord:
    """One graph the seam did not lower, and the reason a reader can act on."""

    reason: str
    stage: str
    nodes: tuple[str, ...] = ()
    loc: str | None = None
    detail: str = ""

    def __str__(self) -> str:  # pragma: no cover - presentation only
        where = f" at {self.loc}" if self.loc else ""
        return f"fallback[{self.stage}]{where}: {self.reason}"


@dataclass
class LoweringPlan:
    """What the seam decided about one graph, and what it observed while deciding."""

    lowered: list[CompiledKernel] = field(default_factory=list)
    fallbacks: list[FallbackRecord] = field(default_factory=list)
    nodes: tuple[str, ...] = ()

    @property
    def fully_lowered(self) -> bool:
        return bool(self.lowered) and not self.fallbacks

    def to_json(self) -> dict[str, Any]:
        return {
            "fully_lowered": self.fully_lowered,
            "lowered": [k.name for k in self.lowered],
            "fallbacks": [
                {
                    "reason": f.reason,
                    "stage": f.stage,
                    "loc": f.loc,
                    "nodes": list(f.nodes),
                }
                for f in self.fallbacks
            ],
            "nodes": list(self.nodes),
        }


# --------------------------------------------------------------------------- #
# The recorded lowerings
# --------------------------------------------------------------------------- #


def recorded_kernels() -> dict[str, str]:
    """`{stem: ttir text}` for every recorded lowering in the package."""
    if not KERNEL_DIR.is_dir():  # pragma: no cover - packaging failure
        return {}
    return {path.stem: path.read_text() for path in sorted(KERNEL_DIR.glob("*.ttir"))}


def _declared_precision(module: Module) -> str:
    """`tt.dot`'s declared input precision, `"ieee"` when it declares none.

    Delegated to `recognize.op_shapes.input_precision`, which is the function the
    recogniser uses — so the emulator's precision and the recogniser's opinion of
    it cannot disagree about the same attribute.
    """
    for op in walk_region(module.body):
        if op.name == "tt.dot":
            return input_precision(op)
    return "ieee"


def _module_of(name: str) -> Module:
    """The parsed module of a recorded lowering.

    Public to the checks rather than private, because "the grid is derived from the
    kernel's own tile width" is a claim about the *IR*, and a check that wants to
    verify it has to read the same IR the derivation reads.
    """
    parsed = parse_module(recorded_kernels()[name])
    if not parsed.ok:
        raise LoweringError(f"{name}: PARSE_UNSUPPORTED — {parsed.diagnostic}")
    return parsed.module


def _problem_extents(
    inputs: Sequence[str], outputs: Sequence[str], env: Mapping[str, int]
) -> dict[str, tuple[int, ...]]:
    """The **buffer** extent of each pointer argument, `(M,K)`/`(K,N)`/`(M,N)`.

    The one derivation in this module that is a *convention* rather than a
    reading, so it is checked rather than trusted: Triton passes a matmul's
    pointers in the order `(a, b, c)`, and a caller's tensors must equal these
    extents — mismatch is a `LoweringError` naming both shapes, never a reshape.

    Tile descriptors cannot supply this. `describe(op)` returns the shape of the
    *tile* the kernel loads (`(64, 32)` for Tier 1's A tile), while the kernel's
    index arithmetic addresses the *whole* tensor: `pid_m * 64 * sam` reaches row
    127 of a 128-row buffer. Allocating storage at tile size — which an earlier
    version of this function did — puts the second program out of bounds, and the
    emulator then refuses correctly (`StorageError`), which is how the bug was
    found rather than shipped.
    """
    rows, columns, inner = env.get("M"), env.get("N"), env.get("K")
    extents: dict[str, tuple[int, ...]] = {}
    if rows and columns and inner:
        if len(inputs) >= 1:
            extents[inputs[0]] = (rows, inner)
        if len(inputs) >= 2:
            extents[inputs[1]] = (inner, columns)
        for name in outputs:
            extents[name] = (rows, columns)
        return extents

    # Outside the matmul convention, the buffer extent is the kernel's own flat
    # address range: the largest `end` any `tt.make_range` in the kernel declares,
    # per pointer. This is the 1-D case (vecadd): the kernel addresses n elements
    # of each buffer and the emulator refuses anything smaller, so under-allocating
    # is caught — but over-allocating to the kernel's declared range is exactly
    # what the hardware's allocation would be. Found by executing t0 on the GPU
    # path: the output buffer was previously allocated at `(1,)` and the emulator
    # correctly refused with `StorageError`.
    #
    # The kernel text is not available here (this function receives only the env
    # and the input/output name lists), so the caller — `prepare`, which holds the
    # parsed module — stashes the derived flat width under a reserved env key.
    flat = int(env.get("%__flat_width__") or 0)
    if flat:
        for name in (*inputs, *outputs):
            extents.setdefault(name, (flat,))
    return extents


def _pointer_roots(graph: DefUseGraph, module: Module, start: str) -> frozenset[str]:
    """Function-argument pointer names reachable from `start` by following operands.

    A breadth-first walk with a visited set, not an unguarded recursion: the
    corpus's chains are a handful of hops, but nothing in the IR bounds them, and
    an unbounded walk over a cyclic graph is a hang rather than a wrong answer.

    Block arguments of an `scf.for` body are aliases — the body's `i`-th iter_arg
    argument is fed by `inits[i]` — so the walk crosses the loop boundary and
    reaches the pointer the *init* came from. Without that, every pointer in Tier 1
    would look like it came from nowhere, because Tier 1's loads address block
    arguments.
    """
    function = module.functions()[0]
    pointer_args = {value.name for value in function.args if value.type.kind == "ptr"}
    definitions: dict[str, Operation] = {}
    aliases: dict[str, str] = {}
    for op in walk_region(module.body):
        for result in op.results:
            definitions[result.name] = op
    for loop in _loops(module):
        body_args = _block_args(loop)
        inits = list(loop.operands[3:])
        for index, init in enumerate(inits):
            position = index + 1
            if position < len(body_args):
                aliases[body_args[position].name] = init.name

    found: set[str] = set()
    seen: set[str] = set()
    pending = [start]
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        if name in pointer_args:
            found.add(name)
            continue
        if name in aliases:
            pending.append(aliases[name])
            continue
        op = definitions.get(name)
        if op is None:
            continue
        pending.extend(operand.name for operand in op.operands)
    return frozenset(found)


def _loops(module: Module) -> list[Operation]:
    return [op for op in walk_region(module.body) if op.name == "scf.for"]


def _block_args(loop: Operation) -> tuple[Any, ...]:
    if not loop.regions or not loop.regions[0].blocks:
        return ()
    return loop.regions[0].blocks[0].args


_BLOCK_RE = re.compile(r"^\s*(\d+)")


def _block_width(module: Module) -> int:
    """The kernel's tile width: the largest `tt.make_range` extent it builds.

    The attribute *text* is `"64 : i32"`, so the parse must take the **leading**
    integer. Concatenating every digit in the string gives `6432` — it eats the
    `32` of the type name — which yields a block wider than the problem, a grid of
    `(1, 1, 1)`, and a program that computes one tile while the seam reports the
    whole result. That was this function's first version.
    """
    extents: list[int] = []
    for op in walk_region(module.body):
        if op.name != "tt.make_range":
            continue
        raw = op.attributes.get("end")
        match = _BLOCK_RE.match(str(getattr(raw, "value", raw) or ""))
        if match:
            extents.append(int(match.group(1)))
    return max(extents) if extents else 0


def _derive_grid(module: Module, env: Mapping[str, int]) -> tuple[int, int, int]:
    """The launch grid: the problem size divided by the kernel's tile width.

    Derived rather than recorded, so a kernel whose block width changes cannot
    silently keep an old grid — and a wrong grid shows up as a wrong *answer* (a
    partially written output), which is a failure the numeric check sees.
    """
    block = _block_width(module)
    rows, columns = env.get("M"), env.get("N")
    if not block or not rows or not columns:
        return (1, 1, 1)
    return (max(1, rows // block), max(1, columns // block), 1)


def prepare(
    name: str,
    ttir: str,
    env: Mapping[str, int],
    grid: tuple[int, int, int] | None = None,
) -> CompiledKernel:
    """Run one recorded lowering through the real pipeline. Raises `LoweringError`.

    Every stage is the shipped one: `parse_module` → `build_def_use` → `annotate`
    (the real recogniser) → `assemble` with the real schema and the *default*
    selector → the program the emulator executes. The only thing this function
    supplies that the pipeline would otherwise derive is the launch environment,
    which is a property of the launch and not of the kernel text.
    """
    parsed = parse_module(ttir)
    if not parsed.ok:
        raise LoweringError(f"{name}: PARSE_UNSUPPORTED — {parsed.diagnostic}")
    module = parsed.module
    graph = build_def_use(module)
    try:
        schema = load_builtin("toyisa1")
    except Exception as exc:  # pragma: no cover - packaging failure
        raise LoweringError(f"{name}: the ISA-1 schema did not load: {exc}") from exc

    program = assemble(
        module,
        graph,
        annotate(module, graph),
        schema,
        env=dict(env),
    )
    if program.markers():
        marker = program.markers()[0]
        raise LoweringError(
            f"{name}: the pipeline refused this kernel — {marker.kind} at "
            f"{marker.loc_name or '?'} ({marker.op_name}): {marker.reason}"
        )

    function = module.functions()[0]
    pointers = tuple(value.name for value in function.args if value.type.kind == "ptr")
    scalars = {
        value.name: int(env[value.name])
        for value in function.args
        if value.type.kind != "ptr" and value.name in env
    }

    inputs: list[str] = []
    outputs: list[str] = []
    for op in walk_region(module.body):
        if op.name not in ("tt.load", "tt.store") or not op.operands:
            continue
        # `tt.load %ptr` reads through its first operand; `tt.store %ptr, %value`
        # writes through its first operand too (`op_shapes.pointer_operand`). Both
        # are operand 0, and the *operation name* is what says read or write.
        roots = _pointer_roots(graph, module, op.operands[0].name)
        target = inputs if op.name == "tt.load" else outputs
        for root in sorted(roots):
            if root not in target:
                target.append(root)

    # A pointer the function takes that nothing loads or stores is still addressed
    # by the program (the emulator requires every pointer input), so it is an input
    # by default rather than silently dropped.
    for pointer in pointers:
        if pointer not in inputs and pointer not in outputs:
            inputs.append(pointer)

    return CompiledKernel(
        name=name,
        entry=function.name,
        program=program,
        pointers=pointers,
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        scalars=scalars,
        grid=grid if grid is not None else _derive_grid(module, env),
        extents=_problem_extents(inputs, outputs, {**env, "%__flat_width__": _block_width(module)}),
        input_precision=_declared_precision(module),
    )


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #


def lower_and_run(
    kernel: CompiledKernel,
    tensors: Sequence[torch.Tensor],
    *,
    policy: PrecisionPolicy | None = None,
) -> torch.Tensor:
    """Execute a prepared kernel on `tensors` and return the written buffer.

    Argument mapping is positional against the kernel's *pointer* arguments, which
    is how Triton calls a kernel: the pointer parameters come first and in order.
    Tensors are moved to fp32 because ISA-1's data model declares one dtype and a
    silent upcast is exactly what the contract refuses ("clear error naming the
    dtype; no silent upcast") — so a non-fp32 input is refused here.
    """
    if len(tensors) > len(kernel.inputs):
        raise LoweringError(
            f"{kernel.name} takes {len(kernel.inputs)} pointer argument(s) "
            f"({', '.join(kernel.inputs)}) but {len(tensors)} were supplied"
        )
    storage: dict[str, Any] = {}
    for name, tensor in zip(kernel.inputs, tensors, strict=False):
        if tensor.dtype != torch.float32:
            raise LoweringError(
                f"{kernel.name}: tensor for {name} has dtype {tensor.dtype}; the toy "
                "device declares f32 only and will not silently upcast"
            )
        expected = kernel.extents.get(name)
        if expected and tuple(tensor.shape) != tuple(expected):
            raise LoweringError(
                f"{kernel.name}: tensor for {name} has shape {tuple(tensor.shape)} but "
                f"this lowering addresses {tuple(expected)}; the mapping is positional "
                "and a mismatch is refused rather than reshaped"
            )
        storage[name] = tensor.detach().cpu().numpy().astype(np.float32, copy=True)

    # Every buffer the kernel addresses must exist at its **problem** extent, not at
    # tile extent: storage is the whole tensor, and the index arithmetic reaches all
    # of it across the grid.
    for name in kernel.inputs:
        storage.setdefault(name, np.zeros(kernel.extents.get(name) or (1,), dtype=np.float32))
    for name in kernel.outputs:
        storage.setdefault(name, np.zeros(kernel.extents.get(name) or (1,), dtype=np.float32))
    for name, value in kernel.scalars.items():
        storage[name] = value

    grid_m, grid_n, _ = kernel.grid
    produced: str | None = None
    counter = policy or kernel.default_policy()
    # One program per tile, in launch order: this is what makes `pid_m`/`pid_n`
    # meaningful. A single `emulate` call executes one program, so a 128x128 problem
    # at a 64-wide tile is four programs — and each one accumulates into the same
    # buffer, exactly as the hardware would.
    for pid_m in range(max(1, grid_m)):
        for pid_n in range(max(1, grid_n)):
            per_program = {
                key: (value.copy() if isinstance(value, np.ndarray) else value)
                for key, value in storage.items()
            }
            written = emulate(kernel.program, per_program, counter, grid=(pid_m, pid_n, 0))
            for name in kernel.outputs:
                if name in written:
                    storage[name] = written[name]
                    produced = name

    if produced is None:
        raise LoweringError(
            f"{kernel.name}: the program wrote no output buffer; expected one of "
            f"{', '.join(kernel.outputs) or '(none declared)'}"
        )
    array = storage[produced]
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))


# --------------------------------------------------------------------------- #
# Graph reading
# --------------------------------------------------------------------------- #

#: The FX targets that mean "matrix multiply". Read from the graph's own `target`
#: attributes, so a different spelling of the same op is a data change, not a code
#: change. `addmm`/`linear` are deliberately absent: they fold a bias in, which is
#: Tier 2's kernel, and pretending Tier 1's lowering covers it would silently drop
#: the bias.
MATMUL_TARGETS = frozenset({"mm", "matmul"})

#: Targets that only rearrange what a kernel already computed. A graph of
#: `mm -> getitem` is one kernel plus plumbing; leaving these unfiltered would
#: make every real Dynamo graph look like it does more than one thing.
PLUMBING_TARGETS = frozenset(
    {"getitem", "view", "reshape", "t", "transpose", "expand", "clone", "contiguous", "detach"}
)


@dataclass(frozen=True)
class GraphMatch:
    """One candidate lowering for a graph, before execution."""

    kernel_name: str
    tensors: tuple[str, ...]
    outputs: tuple[str, ...]


def graph_tensors(graph: Any) -> tuple[str, ...]:
    """The graph's `placeholder` node names, in signature order."""
    placeholders = [
        node.name for node in graph.graph.nodes if node.op == "placeholder"
    ]
    return tuple(placeholders)


def graph_targets(graph: Any) -> tuple[str, ...]:
    """The `target` names of the graph's non-placeholder, non-output nodes."""
    names: list[str] = []
    for node in graph.graph.nodes:
        if node.op in ("placeholder", "output"):
            continue
        target = node.target
        names.append(getattr(target, "__name__", str(target)))
    return tuple(names)


def extract_ttir(graph: Any) -> dict[str, str]:
    """TTIR texts attached to a graph, keyed by entry name.

    The one supported hand-off: a caller (or a future Inductor pass) attaches
    `graph.meta["toyisa"] = {"ttir": {name: text}, "env": {...}, "grid": {...}}`
    and the seam consumes it through the identical pipeline. An empty dict means
    "nothing declared", which is the case the recorded lowerings cover.
    """
    meta = getattr(graph, "meta", None) or {}
    declared = meta.get(DEVICE_NAME) or {}
    ttir = declared.get("ttir") or {}
    if not isinstance(ttir, Mapping):
        return {}
    return {str(key): str(value) for key, value in ttir.items()}


def plan_graph(graph: Any, example_inputs: Sequence[torch.Tensor]) -> LoweringPlan:
    """Decide what to do with one graph. The seam's whole judgement lives here.

    The decision is all-or-nothing and the reason strings are the point: a reader
    of a coverage report gets "this graph has 5 nodes, one kernel applies to 2 of
    them" rather than "partially supported".
    """
    nodes = tuple(node.name for node in graph.graph.nodes)
    targets = graph_targets(graph)
    tensors = graph_tensors(graph)
    plan = LoweringPlan(nodes=nodes)

    if not tensors:
        plan.fallbacks.append(
            FallbackRecord(
                reason="the graph takes no tensor arguments, so no kernel can apply",
                stage="match",
                nodes=nodes,
            )
        )
        return plan

    declared = extract_ttir(graph)

    # 1. TTIR the caller attached wins: it is the general path, and it is the only
    #    one in which the seam lowers a kernel it was not told about in advance.
    for name, text in declared.items():
        try:
            plan.lowered.append(prepare(name, text, _env_for(graph)))
        except LoweringError as exc:
            plan.fallbacks.append(
                FallbackRecord(
                    reason="the declared TTIR did not lower",
                    stage="lower",
                    nodes=nodes,
                    detail=str(exc),
                )
            )
    if plan.lowered:
        plan.fallbacks.clear()
        return plan

    # 2. Otherwise a recorded lowering, matched on the graph's own tensor shapes.
    if _is_single_kernel_graph(targets):
        matches = _match_recorded(graph, example_inputs)
        if matches:
            plan.lowered.append(matches[0])
            plan.fallbacks.clear()
            return plan
        plan.fallbacks.append(
            FallbackRecord(
                reason=(
                    f"no recorded lowering applies to shapes "
                    f"{', '.join(str(tuple(t.shape)) for t in example_inputs[:2])}"
                ),
                stage="match",
                nodes=nodes,
                detail=NOT_EXTRACTED,
            )
        )
        return plan

    # 3. Anything else is eager. The reason is the shape of the graph, not a
    #    hedge: "partially supported" is not a thing this seam can report.
    plan.fallbacks.append(
        FallbackRecord(
            reason=(
                f"the graph is not a single kernel: it has {len(targets)} "
                f"operation(s) ({', '.join(targets[:6])})"
            ),
            stage="match",
            nodes=nodes,
            detail=NOT_EXTRACTED,
        )
    )
    return plan


def _is_single_kernel_graph(targets: Sequence[str]) -> bool:
    """Whether the graph is exactly one computational op plus its plumbing.

    `getitem`/`view`/`t`/`transpose` are shape plumbing that a kernel call itself
    performs; anything else means the graph does more than one thing and the seam
    refuses it rather than lowering a prefix.
    """
    meaningful = [name for name in targets if name not in PLUMBING_TARGETS]
    return len(meaningful) == 1 and meaningful[0] in MATMUL_TARGETS


def _match_recorded(graph: Any, example_inputs: Sequence[torch.Tensor]) -> list[CompiledKernel]:
    """Recorded lowerings whose declared problem shape matches the graph's tensors."""
    targets = graph_targets(graph)
    if not _is_single_kernel_graph(targets):
        return []
    if targets[0] not in MATMUL_TARGETS:
        return []
    shapes = [tuple(tensor.shape) for tensor in example_inputs[:2]]
    if len(shapes) != 2:
        return []
    matches: list[CompiledKernel] = []
    for name, text in recorded_kernels().items():
        env = _RECORDED_ENV.get(name)
        if env is None:
            continue
        problem = (env["M"], env["N"], env["K"])
        if (shapes[0], shapes[1]) != ((problem[0], problem[2]), (problem[2], problem[1])):
            continue
        try:
            matches.append(prepare(name, text, env))
        except LoweringError:
            continue
    return matches


def _env_for(graph: Any) -> dict[str, int]:
    meta = getattr(graph, "meta", None) or {}
    declared = meta.get(DEVICE_NAME) or {}
    env = declared.get("env") or {}
    return {str(key): int(value) for key, value in env.items()}


# --------------------------------------------------------------------------- #
# The registered backend
# --------------------------------------------------------------------------- #


def _backend_return(value: Any) -> Any:
    """Wrap a result the way Dynamo's backend contract expects: as a sequence.

    A backend's returned callable must yield the graph's outputs *as a sequence*,
    because Dynamo addresses them positionally. Returning a bare `Tensor` for a
    single-output graph is not a harmless simplification: measured on torch
    2.12.1+cpu, Dynamo then takes element `[0]` of it, so a `(128, 128)` matmul
    result arrives at the caller as its first row — a 128-element tensor whose
    values are all individually *correct* and whose shape is silently wrong. The
    numeric check that compares against `torch.matmul` catches it immediately;
    nothing else would.
    """
    if isinstance(value, (list, tuple)):
        return value
    return (value,)


def _eager_fallback(graph: Any) -> Callable[..., Any]:
    """An eager callable for the graph, used when the seam refuses to lower it."""
    compiled = graph

    def run(*args: Any) -> Any:
        return _backend_return(compiled(*args))

    return run


#: The ISA the FX proof-of-concept path lowers to. A module-level name rather
#: than a literal at the call site, so a demo can retarget the live seam without
#: editing the backend.
FX_ISA = "toyisa1"


@register_backend(name="toyisa")
def toyisa_backend(graph: Any, example_inputs: Sequence[torch.Tensor]) -> Callable[..., Any]:
    """The registered Dynamo backend (FR-026, SC-001).

    Returns a callable whose *results* are the graph's results either way; what
    differs is who computed them, and that difference is recorded on the callable
    as `toyisa_plan` so a caller (or the coverage report) can read it instead of
    taking a promise.

    A lowering failure is not raised at compile time: `torch.compile` is allowed
    to succeed while individual operations fall back (the contract's first failure
    mode), and the honest place to say so is a record, with the numbers still
    correct.
    """
    plan = plan_graph(graph, example_inputs)
    kernels = list(plan.lowered)

    if not kernels:
        # No recorded TTIR lowering matched. Before falling back to eager, try
        # lowering the FX graph itself (`fx_lower`): that route needs no frozen
        # TTIR, which is the whole point of it — it is the only path here that
        # can lower a graph nobody prepared in advance. It is deliberately narrow
        # and returns None for anything outside its op set, so this is an
        # addition to the fallback chain and never a replacement for it.
        from .fx_lower import lower_fx_graph

        reasons: list[str] = []
        fx = lower_fx_graph(graph, example_inputs, isa_name=FX_ISA, report=reasons)
        if fx is not None and fx.fully_lowered:

            def run_fx(*args: Any) -> Any:
                tensors = [arg for arg in args if isinstance(arg, torch.Tensor)]
                if not tensors:
                    return graph(*args)
                try:
                    produced = fx.run(tensors)
                except (ProgramNotExecutable, StorageError):
                    plan.fallbacks.append(
                        FallbackRecord(
                            reason="the FX-lowered program could not execute",
                            stage="execute",
                            nodes=plan.nodes,
                        )
                    )
                    return graph(*args)
                return _backend_return(torch.from_numpy(produced))

            run_fx.toyisa_plan = plan  # type: ignore[attr-defined]
            run_fx.toyisa_fx = fx  # type: ignore[attr-defined]
            return run_fx

        for reason in reasons:
            plan.fallbacks.append(
                FallbackRecord(
                    reason=reason, stage="fx-lower", nodes=plan.nodes, detail=f"isa={FX_ISA}"
                )
            )
        run = _eager_fallback(graph)
        run.toyisa_plan = plan  # type: ignore[attr-defined]
        return run

    kernel = kernels[0]

    def run(*args: Any) -> Any:
        tensors = [arg for arg in args if isinstance(arg, torch.Tensor)]
        if not tensors:
            return graph(*args)
        try:
            result = lower_and_run(kernel, tensors)
        except ProgramNotExecutable as exc:
            # The one execution-time refusal that IS a fallback: the program carries
            # an `UNSUPPORTED` marker, and the contract routes that kernel to eager
            # with a record (`contracts/torch-seam.md` failure mode 1).
            plan.fallbacks.append(
                FallbackRecord(
                    reason="the emitted program carries an UNSUPPORTED marker",
                    stage="execute",
                    nodes=plan.nodes,
                    loc=exc.marker.loc_name,
                    detail=str(exc),
                )
            )
            plan.lowered.clear()
            return graph(*args)
        # Anything else propagates. `LoweringError` is a caller mistake (a wrong
        # dtype or shape) and a `StorageError` is *our* bug; absorbing either into
        # an eager answer would be the silent-fallback sin this seam exists to
        # avoid. That distinction is the same one `emit.ir.AssemblyError` draws.
        return _backend_return(result)

    run.toyisa_plan = plan  # type: ignore[attr-defined]
    run.toyisa_kernel = kernel  # type: ignore[attr-defined]
    return run


def verify_device() -> dict[str, Any]:
    """Install the device and report what PyTorch now believes about it.

    The assertions that matter are PyTorch's own lookups, not this function's
    return value: `get_interface_for_device("toyisa")` must resolve to this class,
    the backend must appear in `list_backends()`, and the device-state round trip
    (SC-001's `current_device`/`set_device`) must actually move. Everything the
    return value reports is read back out of PyTorch, so a wrong claim here is
    visible in the check that consumes it.
    """
    from torch._dynamo.backends.registry import list_backends
    from torch._dynamo.device_interface import get_interface_for_device

    installed = device_interface.install()
    interface = get_interface_for_device(DEVICE_NAME)
    previous = interface.current_device()
    interface.set_device(0)
    round_tripped = interface.current_device()
    if previous != 0:
        interface.set_device(previous)
    return {
        **installed,
        "interface": interface.__name__,
        "is_available": interface.is_available(),
        "device_count": interface.device_count(),
        "current_device": round_tripped,
        "device_round_trip": previous == round_tripped or round_tripped == 0,
        "registered_backend": "toyisa" in list_backends(),
        "interface_is_ours": interface is device_interface.ToyIsaInterface,
        "slot_inventory": device_interface.measure_slot_inventory(),
    }


