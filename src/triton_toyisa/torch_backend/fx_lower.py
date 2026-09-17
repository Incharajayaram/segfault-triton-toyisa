"""Proof of concept: lower a live `torch.compile` FX graph to the target ISA.

**Scope.** This is a rudimentary path, deliberately. It exists to answer one
question that the recorded-lowering route cannot: *can an arbitrary graph that
Dynamo hands us be lowered to a declaratively described ISA at compile time,
with no frozen TTIR anywhere?* It handles a handful of aten operations on 1-D
and 2-D float32 tensors. Anything else returns `None` and the caller falls back
to eager, which is the same contract the rest of the seam keeps.

**What it is not.** It is not a replacement for the TTIR route. The TTIR path
gets tiling, masking, loop structure and `#loc` provenance from Triton; this one
gets none of those, because an FX graph does not carry them. A graph lowered
here runs one program over whole tensors rather than a grid over tiles. Where
the two disagree, the TTIR path is the one to trust.

**What it does not do — and this is the point.** It does not choose instruction
names. Every instruction is selected by `isa.select.select` against the loaded
schema, from a synthesised access descriptor, exactly as the TTIR path does. So:

    lower_fx_graph(gm, inputs, isa_name="toyisa1")   -> DMA1D / MAC16 / EPI
    lower_fx_graph(gm, inputs, isa_name="toyisa2")   -> LDG / OPU32 / VPU

with no branch on `isa_name` anywhere in this file. That is the property worth
demonstrating; a version of this module that mapped `aten.add -> "EPI"` would
prove nothing, and is the exact mistake `lower.py` used to make.

The emulator still decides *what arithmetic to perform* from each instruction's
`source_ops` provenance, so the aten op is recorded there under the `arith.*`
name the machine knows. That is the known abstraction leak (KNOWN_GAPS G-EPI),
not something this module introduces.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..emit.ir import Imm, Instr, MemRef, Program, SourceRef, SsaRef
from ..emu.exec import ProgramNotExecutable, emulate
from ..emu.precision import PrecisionPolicy
from ..isa.schema import load_builtin
from ..isa.select import select
from ..recognize.descriptor import AccessDescriptor

__all__ = ["FxLowering", "lower_fx_graph", "supported_targets"]

#: aten target -> (schema rule, the `arith.*`/`tt.*` name the emulator dispatches
#: on, the short alias the schemas use in their `op:` lists). None of these is an
#: instruction name: they name the *operation*, and the schema decides which of
#: its own instructions serves it.
#:
#: Two spellings are carried because the schemas disagree: toyisa1 writes
#: `op: [add, relu]` while vortex writes `op: [arith.addf]`. Matching either is
#: the alternative to editing three schemas to agree on a convention that the
#: selector never required in the first place.
_ATEN: dict[str, tuple[str, str, str]] = {
    "add": ("elementwise", "arith.addf", "add"),
    "add_": ("elementwise", "arith.addf", "add"),
    "mul": ("elementwise", "arith.mulf", "mul"),
    "sub": ("elementwise", "arith.subf", "sub"),
    "relu": ("elementwise", "arith.maxnumf", "relu"),
    "mm": ("mac", "tt.dot", "dot"),
    "matmul": ("mac", "tt.dot", "dot"),
}


def supported_targets() -> tuple[str, ...]:
    """The aten operations this proof of concept lowers, for a coverage report."""
    return tuple(sorted(_ATEN))


@dataclass
class FxLowering:
    """One lowered FX graph: the program, how to feed it, and why anything failed."""

    program: Program
    inputs: tuple[str, ...]
    output: str
    isa_name: str
    node_count: int
    #: Shape of the output buffer. The emulator refuses to default a buffer the
    #: program writes, so `run` has to allocate it at the right extent.
    output_shape: tuple[int, ...] = ()
    lowered_nodes: tuple[str, ...] = ()
    refusals: list[str] = field(default_factory=list)
    #: Every selection made, as `(node, rule, chosen, rejected...)`. Kept because
    #: "the generator chose" is only a claim if the alternatives are visible.
    decisions: list[str] = field(default_factory=list)

    @property
    def fully_lowered(self) -> bool:
        return not self.refusals and not self.program.markers()

    def run(self, tensors: Sequence[Any]) -> np.ndarray:
        """Execute the program over `tensors`, in placeholder order."""
        storage: dict[str, Any] = {}
        for name, tensor in zip(self.inputs, tensors, strict=False):
            storage[name] = np.asarray(
                tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else tensor,
                dtype=np.float32,
            )
        storage[self.output] = np.zeros(self.output_shape or (1,), dtype=np.float32)
        produced = emulate(self.program, storage, policy=PrecisionPolicy(input_precision="ieee"))
        return produced[self.output]


def _descriptor(shape: tuple[int, ...], base: str = "%fx") -> AccessDescriptor:
    """A contiguous row-major descriptor for a whole tensor.

    Synthesised from the tensor shape rather than recovered from index
    arithmetic, because an FX graph has no index arithmetic to recover: the
    operands are whole tensors. That is the honest difference between this path
    and the TTIR one, and it is why the descriptor here is always contiguous.
    """
    sizes = shape or (1,)
    strides: list[int] = []
    running = 1
    for extent in reversed(sizes):
        strides.append(running)
        running *= extent
    strides.reverse()
    # `base` is load-bearing, not cosmetic: `emu.exec._pointer_inputs` reads the
    # buffer name from the *descriptor's* base field, not from `MemRef.base`, so
    # a descriptor left at a placeholder name sends the emulator looking for a
    # buffer nobody supplied.
    return AccessDescriptor(
        base=base,
        sizes=tuple(sizes),
        strides=tuple(strides),
        offsets=tuple(0 for _ in sizes),
        shape=tuple(sizes),
        order=tuple(range(len(sizes) - 1, -1, -1)),
        dtype="f32",
        loop_carried=False,
        increment=None,
    )


def _choose(
    schema: Any,
    rule: str,
    shape: tuple[int, ...],
    direction: str | None = None,
    base: str = "%fx",
    names: frozenset[str] = frozenset(),
):
    """Select an instruction of `rule` for a whole-tensor access. Returns a report.

    The selector's second parameter is named `kind` but enumerates by *rule*:
    `schema.of_kind("mac")` returns MAC8/MAC16 while `schema.of_kind("compute")`
    returns nothing. The rule is therefore what gets passed, and no rule->kind
    translation happens here — doing one is how this function first returned
    "no admissible compute instruction" for every graph.
    """
    descriptor = _descriptor(shape, base)
    tile = tuple(shape) if rule == "mac" else None
    env = {"words": int(np.prod(shape or (1,)))}
    if rule == "mac" and len(shape) == 2:
        env.update({"m": shape[0], "n": shape[1], "k": shape[1]})
    report = select(schema, rule, descriptor, tile, env, direction)
    admissible = [c for c in report.candidates if c.admissible and c.cost is not None]
    # An instruction that declares `op:` can only serve the operations it names.
    # Vortex's elementwise units all share one rule and one cost, so without this
    # the minimum-cost rule broke the tie by declaration order and chose VADD for
    # relu -- and the emulator, which dispatches those by instruction name,
    # computed an addition and returned a wrong answer with no diagnostic. An
    # instruction that declares no `op:` serves anything, so toyisa1's catch-all
    # EPI and toyisa2's VPU are unaffected.
    if names:
        declared = [c for c in admissible if getattr(c.instruction, "ops", ())]
        if declared:
            admissible = [c for c in declared if names & set(c.instruction.ops)]
    if not admissible:
        return None, report
    best = min(admissible, key=lambda c: (c.cost, c.instruction.declaration_index))
    return best, report


def _decision_line(node_name: str, rule: str, best: Any, report: Any) -> str:
    rejected = [
        f"{c.instruction.name}({c.rejected_by or c.cost})"
        for c in report.candidates
        if c.instruction.name != best.instruction.name
    ]
    return (
        f"{node_name}: {rule} -> {best.instruction.name} cost={best.cost:g}"
        + (f"; rejected {', '.join(rejected)}" if rejected else "")
    )


def lower_fx_graph(
    graph_module: Any,
    example_inputs: Sequence[Any],
    *,
    isa_name: str = "toyisa1",
    report: list[str] | None = None,
) -> FxLowering | None:
    """Lower one FX graph to `isa_name`, or return `None` if it cannot be lowered.

    `report`, when given, collects the reason for every refusal -- the same
    out-parameter convention `emit.assemble.assemble` uses. Without it a `None`
    return says "fell back" and not "fell back *because* this ISA declares no
    multiply", which is the difference between a coverage report a reader can act
    on and one that only counts.

    `None` rather than an exception: an unlowerable graph is the eager-fallback
    case the seam already handles, not a defect. A graph that lowers *partially*
    is also `None` — all-or-nothing, because a half-lowered program would have to
    hand intermediate values back to eager and this path has no mechanism for
    that.
    """
    try:
        schema = load_builtin(isa_name)
    except Exception:  # pragma: no cover - a missing schema is a packaging fault
        return None

    nodes = list(graph_module.graph.nodes)
    shapes: dict[str, tuple[int, ...]] = {}
    input_names: list[str] = []
    instrs: list[Instr] = []
    decisions: list[str] = []
    refusals: list[str] = report if report is not None else []
    lowered: list[str] = []
    total_cost = 0.0
    tensor_inputs = [t for t in example_inputs if hasattr(t, "shape")]
    position = 0
    output_value: str | None = None

    for node in nodes:
        if node.op == "placeholder":
            if position >= len(tensor_inputs):
                refusals.append(f"{node.name}: no example tensor supplied")
                return None
            name = f"%{node.name}"
            shapes[name] = tuple(int(d) for d in tensor_inputs[position].shape)
            input_names.append(name)
            position += 1
            continue

        if node.op == "output":
            arg = node.args[0]
            arg = arg[0] if isinstance(arg, (tuple, list)) else arg
            output_value = f"%{getattr(arg, 'name', arg)}"
            continue

        if node.op != "call_function":
            refusals.append(f"{node.name}: node kind {node.op!r} is not lowered")
            return None

        target = getattr(node.target, "__name__", str(node.target)).split(".")[0]
        entry = _ATEN.get(target)
        if entry is None:
            refusals.append(f"{node.name}: aten target {target!r} is not in the supported set")
            return None
        rule, source_op, alias = entry

        operands: dict[str, Any] = {}
        operand_shapes: list[tuple[int, ...]] = []
        for index, arg in enumerate(node.args):
            arg_name = f"%{getattr(arg, 'name', arg)}"
            if arg_name in shapes:
                operand_shapes.append(shapes[arg_name])
                role = ("a", "b")[index] if rule == "mac" and index < 2 else f"in{index}"
                operands[role] = SsaRef(arg_name)
            elif isinstance(arg, (int, float)):
                operands[f"in{index}"] = Imm(value=float(arg))
            else:
                refusals.append(f"{node.name}: operand {arg_name} has no known shape")
                return None

        if rule == "mac":
            if len(operand_shapes) != 2 or len(operand_shapes[0]) != 2:
                refusals.append(f"{node.name}: matmul needs two 2-D operands")
                return None
            result_shape = (operand_shapes[0][0], operand_shapes[1][1])
            operands["acc"] = Imm(value=0.0)
        else:
            result_shape = operand_shapes[0] if operand_shapes else (1,)
            if source_op == "arith.maxnumf" and len(operands) == 1:
                # relu is max(x, 0) — the second operand is the clamp, supplied
                # here because aten.relu is unary while the machine's maxnumf is
                # binary. Making it explicit keeps the emulator's arithmetic
                # table honest instead of adding a relu special case to it.
                operands["in1"] = Imm(value=0.0)

        best, report = _choose(schema, rule, result_shape, names=frozenset({source_op, alias}))
        if best is None:
            refusals.append(
                f"{node.name}: no admissible {rule} instruction in {isa_name} for shape {result_shape}"
            )
            return None

        name = f"%{node.name}"
        shapes[name] = result_shape
        decisions.append(_decision_line(node.name, rule, best, report))
        lowered.append(node.name)
        total_cost += float(best.cost or 0.0)
        instrs.append(
            Instr(
                name=best.instruction.name,
                operands=operands,
                cost=float(best.cost or 0.0),
                defs=(name,),
                source_ops=(SourceRef(source_op, len(instrs) + 1, 0, node.name),),
                constraint=str(getattr(best.instruction.constraint, "text", "")),
            )
        )

    if output_value is None or output_value not in shapes:
        return None

    # The result is copied to a named output buffer by a selected *memory*
    # instruction, so the program ends with a store the schema chose rather than
    # with a dangling value.
    store_shape = shapes[output_value]
    out_name = "%fx_out"
    store, store_report = _choose(
        schema, "memory", store_shape, direction="store", base=out_name
    )
    if store is None:
        refusals.append(f"output: no admissible memory instruction in {isa_name}")
        return None
    decisions.append(_decision_line("output", "memory", store, store_report))
    total_cost += float(store.cost or 0.0)
    instrs.append(
        Instr(
            name=store.instruction.name,
            operands={
                "dst": MemRef.of("global", out_name, _descriptor(store_shape, out_name)),
                "value": SsaRef(output_value),
            },
            cost=float(store.cost or 0.0),
            source_ops=(SourceRef("tt.store", len(instrs) + 1, 0, "output"),),
        )
    )

    program = Program(
        isa_name=isa_name,
        schema_version=schema.schema_version,
        kernel_name=getattr(graph_module, "_get_name", lambda: "fx_graph")(),
        instrs=tuple(instrs),
        inputs=(*input_names, out_name),
        total_cost=total_cost,
    )
    return FxLowering(
        program=program,
        inputs=tuple(input_names),
        output=out_name,
        isa_name=isa_name,
        node_count=len(nodes),
        output_shape=store_shape,
        lowered_nodes=tuple(lowered),
        refusals=refusals,
        decisions=decisions,
    )


def try_lower_and_run(
    graph_module: Any,
    example_inputs: Sequence[Any],
    tensors: Sequence[Any],
    *,
    isa_name: str = "toyisa1",
) -> tuple[np.ndarray | None, FxLowering | None]:
    """Lower and execute, returning `(result, lowering)`; `(None, _)` means fall back."""
    lowering = lower_fx_graph(graph_module, example_inputs, isa_name=isa_name)
    if lowering is None or not lowering.fully_lowered:
        return None, lowering
    try:
        return lowering.run(tensors), lowering
    except ProgramNotExecutable:
        return None, lowering
