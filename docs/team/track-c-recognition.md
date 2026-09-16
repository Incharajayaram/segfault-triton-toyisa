# Track C — Recognition, idioms & coverage

**Owner**: C · **Load**: ~3.0 days · **Critical path: the middle**

## Mission

Resolve every memory operand to a canonical descriptor — or explicitly refuse — and find the reduction pattern
in the loop structure the IR already has. Then report coverage honestly, which is the part that decides whether
anyone believes the rest.

## You own

```
src/triton_toyisa/recognize/descriptor.py   AccessDescriptor, describe, descriptor_key, conformance_check
src/triton_toyisa/recognize/op_shapes.py    OP_SHAPES, is_memory_op, is_index_op
src/triton_toyisa/recognize/walk.py         BoundedWalker, resolve_operand, substitute_iter_arg, fold_constant
src/triton_toyisa/idioms/patterns.py        MAC_REQUIRED, EPILOGUE_REQUIRED, MatchResult
src/triton_toyisa/idioms/detect.py          detect_mac, detect_epilogue, detect_all, annotate
src/triton_toyisa/report/coverage.py        fully_lowered, largest_lowered_subgraph, coverage_report, render_markdown
```

**Your contracts**: `contracts/access-descriptor.md` (the "expected results on the corpus" table is your
definition of done), `contracts/coverage-report.md`.
**Your tasks**: T008, T029–T033, T040–T046, T067–T071.

## Day 1 — the descriptor is frozen, and everything you need is a dataclass

1. **Commit `AccessDescriptor` and `DescriptorResult` field definitions before lunch** (freeze item 3). D's
   selector evaluates constraints against this type and B's emitter carries it in `Operand` — they are writing
   against your field names from hour one.
2. **The expectation table.** `tests/contract/test_descriptor.py`, written against a **hand-built `Module`**.
   You do not need A's parser or B's `build_ir` to write these — B's `ssa.py` dataclasses are frozen at lunch
   and you construct `Module`/`Operation`/`SsaValue` literals directly.

   | Tier | Operand | Required |
   |---|---|---|
   | T0 | vector-add pointers | `Ok`, `strides == [1]` |
   | T1 | `%a_ptrs_34`, `%b_ptrs_35` | `Ok`, `loop_carried=True`, `increment == 32`, provenance `scf.for iter_arg → tt.addptr → arith.muli` |
   | T2 | same | `Ok` |
   | T3 | modulo pointer | `Unstructured("modulo wraparound")` — **must not be `Ok`** |

3. **The conformance harness.** `tests/contract/test_make_tptr_conformance.py`: read triton-shared's
   `TTS_Op<"make_tptr">` dialect definition (`repos/triton-shared/include/triton-shared/Dialect/…/TritonStructuredDialect.td`) —
   it has **nine** arguments, not three — and pin our field semantics against it. `shape` means *wraparound
   boundary*; `0` means no wrap. This is our oracle for behaviour, not a runtime dependency.

## Day 2 — the walk

`describe`, `resolve_operand`, `substitute_iter_arg`, `fold_constant`, `BoundedWalker` (`MAX_HOPS = 32`).

The required behaviour is the heart of the project:

- An operand that is an `scf.for` `iter_arg` is resolved by following the recurrence: the value defined before
  the loop plus the per-iteration advance from the body, substituted from `scf.yield`.
- The advance **must be decidable**. If it is not, the answer is
  `Unstructured("non-decidable increment")`. **Never return `Ok` with an unknown increment** — that is the
  silent-wrong-answer path this project is forbidden from having.
- 40-deep chain → `BudgetExhausted(hops=32, limit=32)`. A budget is a *result*, never an exception.

## Day 3 — idioms and both true negatives

`detect_mac`, `detect_epilogue`, `detect_all`, `annotate` (annotate-don't-rewrite: the module is not mutated).

The true negatives are worth as much as the true positive, and there are two:
- T0 vector-add: no `tt.dot` at all → zero MAC matches.
- T3 modulo: the operand is unstructured → zero MAC matches.

Also: two dots in one module → **two** matches, not the first one you find. And capture `input_precision` from
`tt.dot` (`tf32` is a literal in our corpus) — D's emulator needs it.

## Day 4 — coverage, reported as a boolean

`report/coverage.py`. Three rules, all of them corrections to a metric that was wrong before:

1. **`fully_lowered` is a boolean and it comes first.** True iff every value-producing operation is annotated or
   provably elided. A kernel with one unlowered op early in a dataflow chain can read 90% annotated and be
   useless.
2. **The annotated fraction is labelled "upper bound"** in the data and in the rendered text, so it cannot be
   quoted as coverage.
3. **Per tier, per ISA, never merged.** Plus the unsupported inventory with the originating `loc` name, so a
   reader can find the operation in the original Python.

Then the limitations document, generated from the non-goals, each cross-referenced to the source work that
solves it at full generality — or an explicit `no known implementation`.

## Definition of done

- [ ] `AccessDescriptor` committed day 1; consumed by D and B without a field rename
- [ ] T0/T1/T2 operands `Ok` with **exact** field values — not "resolved: yes/no"
- [ ] T3 `Unstructured`, never `Ok`
- [ ] Loop-recurrence test exists (`tests/unit/test_loop_recurrence.py`) and passes on the real T1 fixture
- [ ] Symbolic increment → `Unstructured`; 40-deep chain → `BudgetExhausted`; no `RecursionError`
- [ ] MAC exactly once on T1/T2; **zero** times on T0 and T3
- [ ] Two dots → two matches; `input_precision` captured and propagated
- [ ] Epilogue classified add/relu, and it is emitted **after** the loop
- [ ] Conformance report against `tts.make_tptr` semantics, discrepancies listed
- [ ] Coverage: boolean first, fraction labelled, per tier and per ISA, unsupported inventory with `loc` names
- [ ] The gameable-case fixture reports `fully_lowered = false` with a high fraction

## Gotchas

- **The loop-recurrence case is the whole fight.** The previous revision's operation list —
  `tt.addptr ← tt.splat/tt.broadcast ← arith.muli/addi over tt.make_range/get_program_id` — omitted `scf.for`
  iter_args, `scf.yield` and `tt.expand_dims`, so the matcher's precondition failed on the one kernel it existed
  to demonstrate. `tt.expand_dims` alone appears 6+ times in Tier 1.
- **The descriptor is not `{base, stride, shape}`.** It carries `base, sizes, strides, offsets, shape, order,
  dtype, loop_carried, increment, provenance`. `offsets` and `order` were missing from the earlier design, and
  `shape` there means wraparound boundary, not tile shape.
- **`Unstructured` is never a failure of the project.** For Tier 3 it is the correct, intended answer — it is
  what the negative control exists to produce.
- **Fail closed everywhere**, in the walk as much as in the predicates.
- **The percentage is not the metric.** Say it out loud in the write-up.

## If you are blocked

Day 1: nothing (hand-built `Module`s). Day 2: B's `build_def_use`/`walk_region` — fields frozen day 1, so write
against them and switch to the real graph when it lands. Day 3: nothing. Day 4: D's fallback records and B's
selection data for the coverage table.
