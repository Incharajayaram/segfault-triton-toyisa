# Edge Cases: Triton-IR-to-Declarative-Toy-ISA

100 cases, each with the required behaviour and how it is verified. **Verification** is either a test path or
a disposition (`LIMITATIONS` = recorded in the limitations document with a reason; `UNSUPPORTED` = must produce
a marker, never a wrong answer). No case is left undecided: SC-009 asserts that every row here has one of
these two, and the coverage test reads this file.

Rules that apply to every row: **no crash, no silent wrong output, no unbounded recursion** (Constitution
Principle II, FR-001, FR-005).

---

## A. `ttir` syntax and parsing (EC-001 … EC-025)

| ID | Case | Required behaviour | Verification |
|---|---|---|---|
| EC-001 | Empty input | valid `Module`, zero operations | `tests/unit/test_parser_edge.py` |
| EC-002 | Comment-only input (`// …`) | valid empty `Module` | same |
| EC-003 | Module with `module {}` and no function | valid, zero functions | same |
| EC-004 | Multi-result op `%acc_25:3 = scf.for …` | **all three** results bound, each with its own `SsaValue` | `tests/unit/test_parser_multi_result.py` |
| EC-005 | Multi-result op used with a different arity | `PARSE_UNSUPPORTED` naming the line and the arity | same |
| EC-006 | Out-of-line `#loc` table entries | table retained; operations resolve their `loc` by reference | `tests/unit/test_parser_loc.py` |
| EC-007 | Inline `loc("a_ptrs")` without a table | `Operation.loc.name == "a_ptrs"` | same |
| EC-008 | `loc("file.mlir":12:5)` (path form) | parsed; `name` kept as the path, line/col recorded | same |
| EC-009 | Loc name containing an escaped quote | parsed without terminating the string early | same |
| EC-010 | Loc name containing an underscore and digits | preserved verbatim (`loc("a_ptrs_34")`) | same |
| EC-011 | Nested attribute dicts (`{a = {b = 1}}`) | parsed; both levels retained | `tests/unit/test_parser_attrs.py` |
| EC-012 | Result type attributes (`-> tensor<64x64xf32>`) | `TypeExpr.shape == [64,64]`, `dtype == "f32"` | same |
| EC-013 | `!tt.ptr<f32, 1>` with an address space | `TypeExpr.ptr_space == 1` | same |
| EC-014 | `tensor<64x64x!tt.ptr<f32>>` (nested pointer element) | both the outer shape and the element pointer parsed | same |
| EC-015 | `scf.if` with results and two regions | nested regions retained; `walk_region` descends into both | `tests/unit/test_parser_regions.py` |
| EC-016 | Multi-block region (CFG inside `scf.if`) | both blocks parsed; block arguments bound | same |
| EC-017 | `scf.yield` with three operands | all three operands recorded on the terminator | same |
| EC-018 | Missing trailing newline / no trailing semicolon | parsed identically to the canonical form | `tests/unit/test_parser_edge.py` |
| EC-019 | CRLF line endings | parsed identically to LF; offsets reported in the correct convention | same |
| EC-020 | UTF-8 BOM at file start | tolerated, not treated as a token | same |
| EC-021 | Whole module minified onto one line | parsed; diagnostics still report a column | same |
| EC-022 | `arith.constant` with a `dense<…>` attribute | value decoded, or retained as opaque with `UNSUPPORTED` on use — never mis-decoded | `tests/unit/test_parser_attrs.py` |
| EC-023 | `i64`, `index`, and mixed-width integer types | types parsed; unsupported widths flagged at selection, not at parse | same |
| EC-024 | Unknown dialect operation (`foo.bar`) | retained verbatim; becomes `UNSUPPORTED` at lowering, not a parse error | `tests/unit/test_parser_unknown_op.py` |
| EC-025 | Truncated module (unclosed brace) | `PARSE_UNSUPPORTED` with line and column; no partial module returned | `tests/unit/test_parser_negative.py` |
| EC-026 | Duplicate SSA name for two definitions | `PARSE_UNSUPPORTED` (invalid IR); never last-def-wins | same |
| EC-027 | 200-deep nested regions | no `RecursionError`; iterative descent or an explicit depth limit reported | same |

*Group A contains 27 rows and group G contains 8, not the 25/10 first drafted: EC-026 (duplicate SSA name) and
EC-027 (200-deep nesting) were added after the parser contract was written, and two anticipated seam cases
proved to be the same case as EC-095. The distribution is stated here as it is, not as it was planned, so the
count stays auditable: 27 + 20 + 10 + 15 + 10 + 10 + 8 = 100.*

---

## B. Descriptor recognition (EC-028 … EC-047)

| ID | Case | Required behaviour | Verification |
|---|---|---|---|
| EC-028 | In-loop `tt.load` whose pointer is an `scf.for` iter_arg, constant advance | `Ok`, `loop_carried=True`, `increment == 32` | `tests/contract/test_descriptor.py` |
| EC-029 | Same, advance depends on a symbolic stride | `Unstructured("non-decidable increment")` | same |
| EC-030 | Pointer advanced by the loop's *upper bound* rather than a constant | `Unstructured` (not a constant, not a clean symbol) | same |
| EC-031 | Doubly nested `scf.for` (2-D tiling) | descriptor for the inner operand; outer recurrence folded or `Unstructured` — never silently ignored | same |
| EC-032 | Masked load (`tt.load %p, %mask, %other`) | descriptor for `%p`; the mask recorded as an operands note, not dropped | `tests/contract/test_descriptor.py` |
| EC-033 | Load with `cacheModifier` / `evictionPolicy` attributes | attributes retained on the instruction; descriptor unaffected | same |
| EC-034 | Store rather than load (`tt.store`) | symmetric descriptor; no special-casing needed | same |
| EC-035 | Gather (non-affine index) | `Unstructured("non-affine index")` | same |
| EC-036 | Scatter | `Unstructured`, same rule as EC-035 | same |
| EC-037 | Tier-3 modulo wraparound operand | `Unstructured("modulo wraparound")`; **must not** be `Ok` | `tests/integration/test_tier3.py` |
| EC-038 | `tt.expand_dims` inside the index chain | traversed transparently; shape rank adjusted | `tests/contract/test_descriptor.py` |
| EC-039 | `tt.splat` of a scalar into the index | folded as a constant dimension | same |
| EC-040 | `tt.broadcast` of a tensor index | dimensions resolved; no assumption that broadcast is redundant | same |
| EC-041 | `tt.make_range` / `tt.get_program_id` at a chain root | resolved to the program-id/range symbol; `order` recorded | same |
| EC-042 | 40-deep `tt.addptr` chain | `BudgetExhausted(hops=32, limit=32)` | `tests/unit/test_walk_budget.py` |
| EC-043 | Operand used twice in the same chain (diamond, not a cycle) | resolved once, memoised; no infinite descent | same |
| EC-044 | Cyclic operand reference (invalid SSA) | parse already rejects it (EC-026); if it reaches the walk, `Unstructured("cycle")` | same |
| EC-045 | Pointer loaded from memory (pointer-to-pointer) | `Unstructured("pointer indirection")` | same |
| EC-046 | Pointer selected by `arith.select` | `Unstructured` (both branches not modelled) — never the fall-through branch | same |
| EC-047 | Pointer computed inside `scf.if` and used after | resolve or `Unstructured`; never assume the branch was taken | same |

---

## C. Idiom detection (EC-048 … EC-057)

| ID | Case | Required behaviour | Verification |
|---|---|---|---|
| EC-048 | Tier-1 structure (one loop, one dot, one accumulator) | exactly one `MatchResult` | `tests/contract/test_idioms.py` |
| EC-049 | Tier-0 (no `tt.dot`) | zero matches — true negative | `tests/integration/test_true_negative.py` |
| EC-050 | Tier-3 (modulo, no clean dot) | zero matches — second true negative | same |
| EC-051 | Two independent dots in one module | **two** `MatchResult`s, not the first | `tests/contract/test_idioms.py` |
| EC-052 | `tt.dot` with default precision (no `inputPrecision`) | matched; `input_precision == "ieee"` | same |
| EC-053 | `tt.dot` with `inputPrecision = tf32` | matched; `input_precision == "tf32"` propagated to the emulator policy | same |
| EC-054 | Fused double accumulator (two `iter_args` accumulators, one dot) | matched with one accumulator binding; the second flagged for review | same |
| EC-055 | `tt.dot` outside any loop (single-block) | matched as a non-reduction dot; **not** reported as a MAC reduction | same |
| EC-056 | Post-loop `relu` | epilogue matched, classified `relu` | same |
| EC-057 | Post-loop `add` | epilogue matched, classified `add` | same |

---

## D. ISA schema and selection (EC-058 … EC-072)

| ID | Case | Required behaviour | Verification |
|---|---|---|---|
| EC-058 | Both DMA variants admissible | cheaper (`DMA2D`) chosen; rejection of `DMA1D` recorded with its reason | `tests/contract/test_selector.py` |
| EC-059 | `stride[0] % 4 != 0` | `DMA2D` rejected by predicate → `DMA1D` chosen | same |
| EC-060 | Tile `m` not a multiple of 16 | `MAC16` rejected by `m % 16 == 0` → `MAC8` chosen | same |
| EC-061 | Two candidates with equal cost | deterministic tie-break by schema order; both costs recorded | same |
| EC-062 | Predicate evaluates to `unknown` | treated as inadmissible, and the reason is visible | same |
| EC-063 | No admissible instruction | `no_admissible_lowering = True` → `UNSUPPORTED`; no default emitted | same |
| EC-064 | Cost expression with a zero tile dimension | schema error at load (`cost_of` would divide by zero), not a runtime exception | `tests/contract/test_schema.py` |
| EC-065 | Operand dtype absent from the data model | `UNSUPPORTED` naming the dtype | same |
| EC-066 | Schema declaring a single DMA instruction | `SchemaViolation("instructions.dma.count must be >= 2")` | same |
| EC-067 | Compute instruction missing `accumulate` | `SchemaViolation` naming the instruction | same |
| EC-068 | Program's `schema_version` differs from the loaded schema | refuse with a clear error; never reinterpret | `tests/unit/test_serialize.py` |
| EC-069 | Duplicate instruction names in a schema | `SchemaViolation` | `tests/contract/test_schema.py` |
| EC-070 | Constraint references an undefined term | `SchemaViolation` naming the term | same |
| EC-071 | ISA-2 scratchpad requires an explicit bank move | schema expresses `MOVE`; the pipeline lowers to it or reports `UNSUPPORTED` — no implicit copy | `tests/integration/test_isa2.py` |
| EC-072 | Greedy choice is globally suboptimal (cheap DMA forces an expensive MAC) | emitted cost ≥ oracle minimum, `gap > 0`, and the gap is reported rather than hidden | `tests/contract/test_selector.py` |

---

## E. Emission and ordering (EC-073 … EC-082)

| ID | Case | Required behaviour | Verification |
|---|---|---|---|
| EC-073 | Two loads and a dot inside one loop | all three emitted inside the `Loop`; order preserved (T-4 regression) | `tests/contract/test_assembler.py` |
| EC-074 | Accumulator threaded by `scf.yield` | emitted as `Loop.iter_args`, **not** as an instruction | same |
| EC-075 | `scf.yield` itself | maps to iter-arg re-threading; never emitted as a standalone instruction | same |
| EC-076 | Epilogue after the loop | emitted after `LOOPEND`, never inside | same |
| EC-077 | `UNSUPPORTED` marker originating inside a loop | serialised inside that loop body, with its `loc` name | same |
| EC-078 | Two loops, only one containing a matched dot | the other's contents marked `UNSUPPORTED`; both loops present in the program | same |
| EC-079 | Operation after the loop consuming the loop result | ordering derived from def-use across the region boundary; no hoisting | same |
| EC-080 | Store to `global` from inside the loop | emitted inside the loop; storage space recorded on the operand | same |
| EC-081 | Round-trip of a program containing an `UNSUPPORTED` marker | `deserialize(serialize(p)) == p` | `tests/unit/test_serialize.py` |
| EC-082 | Serialised header's `total_cost` vs recomputed cost | equal; header present so a consumer needs no schema | same |

---

## F. Emulator and precision (EC-083 … EC-092)

| ID | Case | Required behaviour | Verification |
|---|---|---|---|
| EC-083 | tf32 dot | multiply inputs truncated to tf32 mantissa width; independent expected value computed for the test | `tests/contract/test_emulator.py` |
| EC-084 | Default-precision dot | full fp32 arithmetic; tighter tolerance | same |
| EC-085 | Integer path | exact equality (`tolerance == 0`) | `tests/integration/test_numerics.py` |
| EC-086 | `k_blocked(4)` vs `k_major_sequential` accumulation | results differ; each agrees with a reference using the same order; the difference is bounded by the derivation | `tests/unit/test_precision.py` |
| EC-087 | Program containing an `UNSUPPORTED` marker | `ProgramNotExecutable(marker)`; seam routes to eager fallback | `tests/contract/test_emulator.py` |
| EC-088 | Input shape disagrees with the descriptor | `ShapeMismatch` naming operand, expected and got | same |
| EC-089 | Pointer outside emulated storage | error naming pointer and size; no zero-fill | same |
| EC-090 | NaN/Inf in the input | no special-casing; compared under the same tolerance rule | `tests/integration/test_numerics.py` |
| EC-091 | Long reduction (k = 4096) precision drift | drift within the derived tolerance, or the derivation is corrected and the change recorded | same |
| EC-092 | Three runs of the same program | byte-identical outputs (determinism) | `tests/unit/test_determinism.py` |

---

## G. PyTorch seam (EC-093 … EC-100)

| ID | Case | Required behaviour | Verification |
|---|---|---|---|
| EC-093 | 64×64 matmul through `torch.compile` | executes on the toy device; matches eager within tolerance (SC-001) | `tests/integration/test_torch_seam.py` |
| EC-094 | `torch.tensor(...).to("toyisa")` | succeeds; device visible to PyTorch | same |
| EC-095 | `device_count` / `is_available` / `current_device` | consistent, non-crashing values | `tests/contract/test_device_interface.py` |
| EC-096 | Graph break inside the compiled function | handled by PyTorch; the break is recorded in the coverage report, not hidden | `tests/integration/test_torch_seam.py` |
| EC-097 | Operation outside the corpus | eager fallback with a **correct** result plus a `FallbackRecord`; never a wrong number | same |
| EC-098 | Backend invoked twice on the same graph | deterministic result; no state leakage between invocations | same |
| EC-099 | `Stream` / `Event` stubs | raise `NotImplementedError` **with a reason**, and the reason appears in the limitations document — silent no-op stubs are a violation | `tests/contract/test_device_interface.py` |
| EC-100 | Upstream PyTorch adds a `DeviceInterface` slot | the slot-inventory contract test fails, forcing an explicit decision instead of a silent default | same |

---

## Dispositions

- Every `UNSUPPORTED` / `PARSE_UNSUPPORTED` row above must be exercised by the fuzz pass (SC-008), which
  asserts the marker exists and no exception was raised.
- `LIMITATIONS`-style rows (EC-029, EC-031, EC-044/045/046/047, EC-071, EC-099) each produce an entry in the
  generated limitations document with the source work that solves it at full generality (FR-031).
- The count is fixed at 100 with two substitutions in group A (EC-026, EC-027). Substitutions are recorded
  here rather than by renumbering, so that a reference to an ID stays valid across revisions.
