# Contract: ISA schema and constraint evaluator

**Module**: `src/triton_tritonflow/isa/schema.py`
**Consumers**: the selector, the assembler's validation step, the transfer report.
**Requirements**: FR-013 … FR-016, FR-017 (fail-closed half).

## Interface

```python
load_schema(path: str | os.PathLike[str]) -> IsaSchema        # raises SchemaError, never silently defaults
validate_schema(schema: IsaSchema) -> list[SchemaViolation]   # empty list == valid

class IsaSchema:
    schema_version: int
    name: str
    data_model: DataModel
    instructions: dict[str, Instruction]

class Instruction:
    name: str
    kind: Literal["memory", "compute"]
    addressing: AddressingForm
    computation_attrs: list[str]
    addressing_attrs: list[str]
    constraint: Predicate
    cost: CostExpr
    accumulate: AccumulateSpec | None      # required for kind == "compute"
    semantics: str

def evaluate(predicate: Predicate, descriptor: AccessDescriptor) -> Literal[True, False, "unknown"]
def cost_of(instr: Instruction, descriptor: AccessDescriptor, tile: TileShape) -> float
```

## Preconditions

- The schema file is trusted input (it is part of this repository), but it is still validated.

## Postconditions

1. **Schema validity is enforced, not assumed.** `validate_schema` rejects a schema that has fewer than two
   `memory` instructions or fewer than two `compute` MAC-kind instructions (FR-015), or an instruction missing
   `constraint`, `cost`, or (for compute) `accumulate` (FR-014, FR-016). A single-variant schema is a *schema
   error*, because a schema with no alternatives cannot drive a decision (T-8).
2. **Fail-closed evaluation.** `evaluate` returns `True`, `False`, or `unknown`; the selector treats `unknown`
   as inadmissible. An unknown is never a pass (FR-017).
3. **Costs are total.** Every instruction has a cost expression that evaluates for every descriptor it is ever
   considered against. A cost that cannot be evaluated is a schema error, not a runtime surprise.
4. **Accumulator semantics are mandatory** for compute instructions: `precision`, `reduce_dim`, `order`. A
   missing `order` is a schema error, because without it the tolerance of the emulator is unfalsifiable (T-5).
5. **Versioned.** `schema_version` is required and recorded in every emitted program header, so a program can
   never be interpreted under a different schema than the one that produced it.
6. **Determinism.** Two loads of the same file produce equal objects; no dict-ordering dependence leaks into
   output ordering.

## Schema errors (all must be reported with file + key path)

| Case | Required behaviour |
|---|---|
| One DMA instruction only | `SchemaViolation("instructions.dma.count must be >= 2")` |
| Instruction with no `cost` | violation naming the instruction |
| Predicate syntactically valid but references an undefined term | violation naming the term |
| `accumulate` absent on a compute instruction | violation |
| Duplicate instruction name | violation |
| Unknown `schema_version` | violation (forward compatibility is not assumed) |

## Tests

- **Contract** (`tests/contract/test_schema.py`): ISA-1 and ISA-2 both validate; every row of §1.2's predicate
  table evaluates correctly on a hand-built descriptor, including the `unknown` cases.
- **Negative**: six intentionally broken schemas, each asserting the specific violation.
- **Property**: for random descriptors, `evaluate` never raises; it returns one of the three literals.
