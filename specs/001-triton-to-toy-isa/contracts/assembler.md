# Contract: assembler and serialiser

**Module**: `src/triton_tritonflow/emit/` (`ir.py`, `assemble.py`, `disasm.py`)
**Consumers**: the emulator and the PyTorch seam. **Requirements**: FR-019, FR-020, FR-005.

## Interface

```python
assemble(module: Module, graph: DefUseGraph, annotations: AnnotationSet,
         schema: IsaSchema) -> Program

serialize(program: Program) -> str            # stable text form
deserialize(text: str) -> Program             # inverse of serialize
disassemble(program: Program) -> str          # human-readable, same information

class Program:
    isa_name: str
    schema_version: int
    kernel_name: str
    loops: list[Loop]
    instrs: list[Instr]
    epilogue: list[Instr]
    total_cost: float
    unsupported: list[UnsupportedMarker]
```

## Preconditions

- Every memory operand has already been resolved or marked `UNSUPPORTED`.
- The schema validated, and `Program.schema_version` matches the schema that produced it.

## Postconditions

1. **Region structure is preserved (FR-019).** An operation inside `scf.for` becomes an `Instr` inside the
   corresponding `Loop`. Both `tt.load`s of Tier 1 stay inside the loop; the accumulator stays loop-carried
   through `Loop.iter_args`; `scf.yield` maps to the re-threading of `iter_args`, not to an instruction.
   A flat topological order is neither used nor produced.
2. **Complete coverage of the input (Principle II).** Every value-producing operation of the input module is
   either represented by an `Instr` or listed in `unsupported`. The union is exact: nothing is dropped, and
   nothing is invented.
3. **Round-trip (FR-020).** `deserialize(serialize(p)) == p` for every `Program` produced from the corpus. The
   serialised form is byte-stable across runs (FR-004).
4. **Constraint re-check.** Each `Instr` re-validates its instruction's constraint against the operand it was
   emitted for. A violation is an internal error naming the instruction, the operand and the failing predicate —
   it is not silently emitted.
5. **Cost is recorded, not recomputed.** `total_cost` is the sum of the selected instructions' costs, stored in
   the program and in the serialised header, so a consumer does not need the schema to interpret the file.
6. **`UNSUPPORTED` is a first-class instruction**, serialised like any other, carrying the originating `loc`
   name and the reason. It is never an omission (FR-005).

## Expected results on the corpus

| Kernel | Expectation |
|---|---|
| T0 vector add | no `Loop`; one or more `DMA*`/`EPI`; `unsupported == []` |
| T1 matmul | exactly one `Loop` with `iter_args` of size 3; one `MAC*` inside it; `unsupported == []` |
| T2 matmul+relu | as T1, plus one `EPI relu` after the loop |
| T3 modulo | `unsupported` contains at least one marker naming the originating `loc`; the program still serialises and round-trips |

## Failure modes

| Case | Required behaviour |
|---|---|
| Annotation covers an operation that no instruction can express | `UnsupportedMarker` with reason; program still valid |
| Two annotations claim the same operation | internal error naming both, at assembly time — not a silently dropped one |
| `serialize` given a `Program` with a dangling operand reference | raises `AssemblyError` naming the operand (programmer error, distinct from user-input failure) |
| Deserialising text whose `schema_version` differs from the loaded schema | refuse with a clear error; never reinterpret |

## Tests

- **Contract** (`tests/contract/test_assembler.py`): the table above.
- **Round-trip property** (`tests/unit/test_serialize.py`): corpus programs, plus 200 generated programs,
  assert `deserialize(serialize(p)) == p`.
- **Region property**: for every fixture, the set of operations inside the input's `scf.for` equals the set of
  `Instr`s inside the output's `Loop` (this is the regression against T-4).
- **Coverage property**: `annotated_ops ∪ unsupported_ops == value_producing_ops` exactly, for every fixture.
