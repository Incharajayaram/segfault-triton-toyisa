# Contract: structured-access recogniser

**Module**: `src/tritonflow/recognize/` (`descriptor.py`, `op_shapes.py`, `walk.py`)
**Consumers**: idiom detection, selection, emission. **Requirements**: FR-006 … FR-009, FR-012.

## Interface

```python
MAX_HOPS: int = 32                     # named constant, not a magic number

resolve_operand(value: SsaValue, graph: DefUseGraph) -> DescriptorResult

DescriptorResult = Ok(descriptor) | Unstructured(reason, ops) | BudgetExhausted(hops, limit)

describe(ptr: SsaValue, access: tuple[Stream, ...], graph) -> DescriptorResult
canonicalize(module: Module) -> Module            # idempotent; no reduction claim
conformance_check(module: Module, oracle) -> ConformanceReport
```

## Preconditions

- The module parsed successfully and its def-use graph is built.
- `value` is the operand of a memory operation (`tt.load`, `tt.store`, `tt.addptr` chain head).

## Postconditions

1. **Total and typed.** Every operand resolves to exactly one of the three results. No exception (FR-008).
2. **Loop-carried operands are resolved, not skipped.** An operand that is an `scf.for` `iter_arg` is resolved
   by following the recurrence: the value defined *before* the loop plus the per-iteration advance from the
   loop body, with the advance substituted from `scf.yield` (FR-006). This is the case v1's operation list
   omitted.
3. **The increment must be decidable.** If the per-iteration advance depends on a value that cannot be reduced
   to a constant or a symbol, the result is `Unstructured(reason="non-decidable increment")`. Returning `Ok`
   with an unknown increment is a contract violation.
4. **All descriptor fields populated.** `base`, `sizes`, `strides`, `offsets`, `shape`, `order`, `dtype`,
   `loop_carried`, `increment`, `provenance` — not v1's `{base, stride, shape}` (FR-007).
5. **`shape` means wraparound boundary.** `0` means "no wrap in this dimension", following `tts.make_tptr`'s
   definition of the field.
6. **Bounded.** The walk visits at most `MAX_HOPS` operations; on the 33rd it returns `BudgetExhausted`. The
   budget is reported, and exhausting it is never a crash and never an unbounded recursion.
7. **Canonicalisation is idempotent** and asserts only idempotence and vocabulary-closure properties. It does
   **not** assert a node-count reduction — the measured data has 13 splats / 6 broadcasts that are *required*
   broadcast expansion and 0 sign-extension round-trips (FR-012, T-8-adjacent correction).
8. **Conformance.** For every fixture operand where the oracle produces a descriptor, our descriptor's
   `base/sizes/strides/offsets` agree, or the disagreement is recorded in `ConformanceReport` with the operand
   name. The oracle is `tts.make_tptr` semantics (FR-009) and is never a runtime dependency.

## Expected results on the corpus

| Tier | Operand | Required result |
|---|---|---|
| T0 | vector-add pointers | `Ok`, `strides == [1]` |
| T1 | `%a_ptrs_34`, `%b_ptrs_35` | `Ok`, `loop_carried=True`, `increment == 32`, `provenance` shows `scf.for iter_arg → tt.addptr → arith.muli` |
| T2 | same as T1 | `Ok` |
| T3 | modulo pointer | `Unstructured(reason=~"modulo wraparound")` — must **not** be `Ok` |

## Failure modes

| Input | Required behaviour |
|---|---|
| Pointer advanced by the loop induction variable times a *symbolic* stride | `Unstructured` (non-decidable increment); still no crash |
| Gather / scatter (non-affine index) | `Unstructured(reason="non-affine index")` |
| Deep pointer chain (40 `tt.addptr`s) | `BudgetExhausted(hops=32, limit=32)` |
| Pointer in a nested `scf.if` inside `scf.for` | resolve or `Unstructured`; never silently assume the fall-through path |

## Tests

- **Contract** (`tests/contract/test_descriptor.py`): the table above, field-by-field, on frozen fixtures.
- **Unit**: recurrence resolution with a constant advance; refusal with a symbolic advance; hop budget.
- **Conformance** (`tests/contract/test_make_tptr_conformance.py`): oracle comparison, discrepancies listed.
- **Negative**: Tier 3 and the fuzz kernel must not yield `Ok`.
