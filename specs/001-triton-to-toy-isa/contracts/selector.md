# Contract: instruction selection

**Module**: `src/triton_tritonflow/isa/select.py`
**Consumers**: the emitter. **Requirements**: FR-017, FR-018, SC-005.

## Interface

```python
enumerate_candidates(schema: IsaSchema, kind: str, descriptor: AccessDescriptor,
                     tile: TileShape) -> list[Candidate]

select(schema: IsaSchema, kind: str, descriptor: AccessDescriptor,
       tile: TileShape) -> SelectionReport

class Candidate:
    instruction: Instruction
    admissible: bool
    rejected_by: str | None        # the predicate text that failed, or None
    cost: float | None

class SelectionReport:
    chosen: Instruction | None
    chosen_cost: float | None
    candidates: list[Candidate]
    oracle_min_cost: float | None
    oracle_chosen: Instruction | None
    gap: float
    no_admissible_lowering: bool
```

## Preconditions

- The schema validated (see `isa-schema.md`) and the descriptor is `Ok` (unstructured operands never reach
  the selector; they are already `UNSUPPORTED`).

## Postconditions

1. **Everything is enumerated.** For a given `kind`, every instruction of that kind appears in
   `candidates`, including rejected ones. The schema is small by design, so enumeration is also the oracle.
2. **Rejections are attributed.** `rejected_by` contains the specific predicate that failed, never `"unknown"`
   and never `None` for a rejected candidate. This is what makes "the generator chose" auditable rather than
   asserted (FR-018).
3. **Minimum cost among the admissible.** `chosen` is the admissible candidate with the lowest `cost`. Ties
   break deterministically by schema declaration order (determinism, FR-004).
4. **No default fallback.** If no candidate is admissible, `chosen is None`,
   `no_admissible_lowering is True`, and the emitter produces `UNSUPPORTED`. Selecting an inadmissible
   instruction "because something must be emitted" is a contract violation (FR-017).
5. **Oracle agreement.** `oracle_min_cost` equals `chosen_cost` whenever the greedy rule is optimal, and
   `gap ≥ 0` always. The gap is reported rather than hidden (SC-005).
6. **Re-validated downstream.** The emitter independently checks the chosen instruction's constraint against
   the operand. Selection is not trusted transitively (data-model invariant 3).

## Expected results on the corpus

| Operand | Expected |
|---|---|
| Tier-1 `%a_ptrs_34` (stride multiple of 4, in bounds) | both DMA variants admissible → `DMA2D` chosen (0.60 < 1.00); `DMA1D` rejected with `"cost(1.00*words) > 0.60*words"` |
| An operand whose `stride[0]` is not a multiple of 4 | `DMA2D` rejected by `stride[0] % 4 == 0` → `DMA1D` chosen |
| A 64×64×32 tile | `MAC16` admissible and cheaper → chosen; `MAC8` rejected on cost |
| A 8×8 tile with a unit-stride operand row | `MAC16` rejected by `m % 16 == 0` → `MAC8` chosen |
| An operand failing *every* constraint | `no_admissible_lowering = True` → `UNSUPPORTED`, no crash |

## Failure modes

| Case | Required behaviour |
|---|---|
| Cost expression divides by zero for a degenerate tile | schema error at load time, not a runtime `ZeroDivisionError` |
| Two candidates with equal cost | deterministic tie-break; both recorded with equal cost |
| Predicate returns `unknown` | treated as inadmissible; the `unknown` is recorded in `rejected_by` so the reason is visible |

## Tests

- **Contract** (`tests/contract/test_selector.py`): the table above, asserting the rejected list contents.
- **Property**: for random rasters of descriptor values, `gap >= 0` and `chosen` satisfies its own constraint.
- **Negative**: a descriptor that satisfies nothing yields `no_admissible_lowering`, never a default.
- **Oracle**: a brute-force enumeration over the schema agrees with `select` on 1,000 random cases.
