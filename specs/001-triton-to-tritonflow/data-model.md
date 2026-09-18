# Data Model: Triton-IR-to-Declarative-Toy-ISA

**Phase 1 output** for `001-triton-to-tritonflow`. Every entity below is a real data structure with named
fields. Where a field exists because of a specific corrected defect, the defect is named.

---

## 1. ISA description document (the contract with the generator)

Format: YAML, one file per ISA, loaded by `isa/schema.py:load_schema()`. The document is the *only*
ISA-specific input to the pipeline; everything else is either ISA-agnostic or lives in one rule module per
ISA (FR-013…FR-016).

### 1.1 Complete ISA-1 document (`isa/schemas/tritonflow1.yaml`)

```yaml
schema_version: 1
name: tritonflow1
description: >
  Flat-memory tile ISA. 1-D contiguous DMA, 2-D tiled DMA, square MAC tiles,
  post-loop elementwise epilogue. No scratchpad/accumulator distinction: bank
  and tiling constraints are the complexity TensorLift's RTL extraction exists
  to recover; this project starts downstream of that problem.

data_model:
  memory_spaces:
    - name: global
      kind: flat                 # 'flat' | 'scratchpad' | 'accumulator'
      dtype: f32
      alignment_words: 4
  tile_dims: [BM, BN, BK]        # names only; values come from the IR
  max_dims: 4

instructions:
  - name: DMA1D
    kind: memory
    addressing: {form: contiguous_1d, fields: [base, length]}
    time: {expr: "words / 1.0", unit: word}
    cost: {expr: "1.00 * words", unit: word}
    constraint: "in_bounds(base, length)"
    semantics: "dst[0:length] = src[0:length]"

  - name: DMA2D
    kind: memory
    addressing: {form: tiled_2d, fields: [base, sizes, strides, offsets]}
    time: {expr: "words / 1.6", unit: word}
    cost: {expr: "0.60 * words", unit: word}
    # Write-combining only pays off when rows are a whole number of cache lines
    # and the row stride is unit: this is why the variant exists at all.
    constraint: "all_of(shape[0] % 4 == 0, stride[0] % 4 == 0, in_bounds_all())"
    semantics: "dst[i, 0:sizes[1]] = src[offsets[0]+i*strides[0], offsets[1]:...]"

  - name: MAC8
    kind: compute
    tile: {m: 8, n: 8}
    addressing: {form: register}
    computation_attrs: [tile_shape, dtype, accumulate]
    accumulate: {precision: f32, reduce_dim: k, order: k_major_sequential}
    time: {expr: "1.0 * m * n * k / (8*8)", unit: cycle}
    cost: {expr: "1.00 * m * n * k / (8*8)", unit: mac}
    constraint: "all_of(aligned(a_base, 4), aligned(b_base, 4))"
    semantics: "acc[m,n] += sum_k a[m,k] * b[k,n]"

  - name: MAC16
    kind: compute
    tile: {m: 16, n: 16}
    addressing: {form: register}
    computation_attrs: [tile_shape, dtype, accumulate]
    accumulate: {precision: f32, reduce_dim: k, order: k_major_sequential}
    time: {expr: "1.0 * m * n * k / (16*16)", unit: cycle}
    cost: {expr: "0.70 * m * n * k / (16*16)", unit: mac}
    constraint: "all_of(stride[1] == 1, aligned(a_base, 4), m % 16 == 0, n % 16 == 0)"
    semantics: "acc[m,n] += sum_k a[m,k] * b[k,n]"

  - name: EPI
    kind: compute
    op: [add, relu]
    addressing: {form: elementwise, fields: [base, length]}
    cost: {expr: "0.50 * elements", unit: element}
    constraint: "all_of(acc_dtype == op_dtype, in_bounds(base, length))"
    semantics: "dst[i] = op(src[i])"
```

### 1.2 Constraint predicate language (decidable, fail-closed)

Terms: `shape[i]`, `stride[i]`, `offset[i]`, `base`, `length`, `m`, `n`, `k`, `dtype_bits`,
`acc_dtype`, `op_dtype`.
Predicates: `X == k`, `X % k == 0`, `aligned(X, k)`, `in_bounds(base, length)`, `in_bounds_all()`,
`all_of(p1, p2, …)`, `any_of(p1, p2, …)`.

- **Fail-closed**: any predicate that cannot be decided from the descriptor returns **inadmissible**, not
  admissible. An unknown is never silently treated as a pass (FR-017).
- Deliberately *not* Turing-complete. ACT's addressing phase is integer constraint programming (§6); a full
  CP solver does not fit the budget, and a decidable subset is enough to make selection real.

### 1.3 ISA-2 document (`isa/schemas/tritonflow2.yaml`) — what differs

Same grammar, different contents, along exactly the four axes of §`isa2` in the methodology:

| Axis | ISA-1 | ISA-2 |
|---|---|---|
| Memory model | flat `global` only | `scratchpad` + `accumulator` banks; explicit `move` between them |
| Compute primitive | square `MAC8` / `MAC16` tiles | `OPU` outer-product unit (`outer u×v→u×v`) + `CONV` 1-D window reduce |
| Addressing | 1-D pointer walk, 2-D tile | 2-D strided DMA with an explicit `{row_off, col_off}` pair; no unit-stride case |
| Epilogue | none beyond elementwise add/relu | `CLAMP` saturation (TensorLift Table 2 pass B5, `detect-clamp`) |
| Accumulator order | `k_major_sequential` | `k_blocked(4)` — *different*, on purpose |

**Rule**: ISA-2 must be authored against this table alone, without consulting the ISA-1 rule module, and the
transfer report must list every line that had to change outside `isa/schemas/` and `isa/rules/tritonflow2.py`.
That list is the falsification instrument (FR-030, US4).

---

## 2. Parsed IR model (`ttir/ssa.py`)

| Entity | Fields | Notes |
|---|---|---|
| `Module` | `body: Region`, `loc_table: dict[str, Loc]`, `source_path`, `triton_version` | the `#loc` table is kept, not discarded |
| `Function` | `name`, `args: list[SsaValue]`, `result_types`, `body: Region`, `loc` | |
| `Operation` | `name: str`, `operands: list[SsaValue]`, `results: list[SsaValue]`, `attributes: dict[str, Attr]`, `regions: list[Region]`, `loc`, `line: int` | `results` is a **list** — a multi-result op (`%acc_25:3`) binds all three, not the first (FR-002) |
| `Region` | `blocks: list[Block]`, `parent: Operation` | never flattened (FR-019) |
| `Block` | `args: list[SsaValue]`, `operations: list[Operation]`, `terminator` | `scf.yield` is the terminator of a loop-body block |
| `SsaValue` | `name` (e.g. `%acc_37`), `type: TypeExpr`, `def_op: Operation`, `loc` | |
| `TypeExpr` | `raw: str`, `kind` (`tensor`/`ptr`/`scalar`/…), `shape`, `dtype`, `ptr_space` | `tensor<64x32x!tt.ptr<f32>>` parsed, not string-matched |
| `Loc` | `name: str`, `line: int`, `col: int` | 169 occurrences in the Tier-1 fixture; names are the original Python identifiers (`loc("a_ptrs")`) |
| `ParseDiagnostic` | `kind = PARSE_UNSUPPORTED`, `line`, `col`, `expected`, `found` | the failure route that v1 did not have |

**Invariants.** Every `SsaValue` has exactly one defining `Operation`; every operand reference resolves or is a
function argument; `def_op` for a block argument is `None`. A violation is a parser bug, not a user error.

---

## 3. Recognition model (`recognize/descriptor.py`)

`AccessDescriptor` — the canonical form of a memory operand. Fields are the ones a conformant structured
descriptor needs, not v1's `{base, stride, shape}` (FR-007).

| Field | Type | Meaning |
|---|---|---|
| `base` | `str` | SSA value of the base pointer |
| `sizes` | `list[int \| SymExpr]` | per-dimension extent |
| `strides` | `list[int \| SymExpr]` | per-dimension stride; `stride[-1] == 1` implied for contiguous |
| `offsets` | `list[int \| SymExpr]` | per-dimension start offset (absent from v1's descriptor) |
| `shape` | `list[int \| SymExpr]` | **wraparound boundary**, not tile shape — `0` means no wrap |
| `order` | `list[int]` | dimension order of the tile |
| `dtype` | `str` | element dtype |
| `loop_carried` | `bool` | true when the descriptor's base is an `scf.for` iter_arg (the wrong-answer guard) |
| `increment` | `int \| SymExpr \| None` | the value added per iteration; **must be decidable** to keep the descriptor structured |
| `provenance` | `list[str]` | the op chain walked, for diagnostics |

Result union:

```
DescriptorResult = Ok(AccessDescriptor)
                 | Unstructured(reason: str, ops: list[str])
                 | BudgetExhausted(hops: int, limit: int)
```

`MAX_HOPS` is a named constant. Exhausting it is a *result* (`BudgetExhausted`), never an exception (FR-008).

**Why `loop_carried` and `increment` exist.** Tier 1's operands are `scf.for` iter_args advanced by
`arith.muli %sak, %c32_i32` — a `muli` of two values, one of them the loop variable. The walk must
substitute the constant advance and refuse to pretend when the increment is not decidable. This is the exact
case v1's operation list omitted.

---

## 4. Idiom model (`idioms/patterns.py`)

| Entity | Fields |
|---|---|
| `ReductionPattern` | `id: "mac"`, `required: {one scf.for, iter_args with an accumulator operand, exactly one tt.dot consuming it, one scf.yield re-threading it}`, `optional: [pre-load splat/broadcast chain, post-loop epilogue]` |
| `EpiloguePattern` | `id: "epilogue"`, `required: {ops applied to the accumulator after the loop}`, `classify: add \| relu \| other` |
| `MatchResult` | `pattern_id`, `ops: list[Operation]`, `bindings: dict[str, SsaValue]` (e.g. `{"a": %a, "b": %b, "acc": %acc_36, "tile": (64,64,32)}`), `tile_shape`, `dtype`, `input_precision` (`tf32` \| `ieee`), `multiplicity: int` |

**Rules.** All matches are reported, not the first (`multiplicity`, one `MatchResult` per occurrence). Zero
matches is a valid result and is asserted by two true-negative tests (Tier 0 and Tier 3).
`input_precision` is captured here because it is read from `tt.dot`'s attributes and is required downstream by
the emulator (T-6).

---

## 5. Emission model (`emit/ir.py`)

```
Instr   = name, operands: dict[str, Operand], loop: LoopId | None, cost: float, source_ops: list[Operation]
Loop    = id, induction_var, lower, upper, step, iter_args: list[SsaValue], body: list[Instr]
Program = isa_name, kernel_name, loops: list[Loop], instrs: list[Instr], epilogue: list[Instr],
          total_cost: float, unsupported: list[UnsupportedMarker]
Operand = SsaRef(name) | Imm(int | float) | MemRef(space, base, access: AccessDescriptor)
UnsupportedMarker = op_name, loc_name, reason, kind = UNSUPPORTED | PARSE_UNSUPPORTED
```

**Invariant (FR-019).** An `Instr` whose `source_ops` lie inside a region is emitted inside that region's
`Loop`. `scf.yield`-threaded values map to `Loop.iter_args`. Flat topological order is not used and not
produced (T-4, D10).

**Serialised form (FR-020)** — stable, round-trippable, human-readable:

```
; tritonflow1  kernel=t1_matmul  cost=18432.0
LOOP k: 0..32 step 1 iter(%a_ptrs, %b_ptrs, %acc)
  DMA1D  dst=smem_a  src=a_ptrs  len=2048
  DMA1D  dst=smem_b  src=b_ptrs  len=2048
  MAC8   acc=acc     a=smem_a    b=smem_b  k=32
  ADV    reg=a_ptrs  by=32
LOOPEND
EPI relu  dst=c_ptrs  src=acc
```

---

## 6. Selection model (`isa/select.py`)

| Entity | Fields |
|---|---|
| `Candidate` | `instruction: Instruction`, `admissible: bool`, `rejected_by: str \| None` (the predicate that failed), `cost: float \| None` |
| `SelectionReport` | `chosen: Instruction`, `chosen_cost`, `candidates: list[Candidate]`, `oracle_min_cost: float`, `oracle_chosen: Instruction \| None`, `gap: float`, `no_admissible_lowering: bool` |

**Rules.** Enumeration is exhaustive over the schema for each operand (the schema is small by design, so the
oracle *is* the enumeration). A rejection is recorded with the failing predicate, never as a silent skip
(FR-018). If nothing is admissible, `no_admissible_lowering = True` and the emitter produces `UNSUPPORTED` —
it never falls back to a default instruction (FR-017).

---

## 7. Device and seam model (`torch_backend/`)

| Entity | Fields / members |
|---|---|
| `ToyDevice` | `name = "tritonflow"`, `storage: dict[DevicePtr, np.ndarray]`, `allocator`, `streams`, `events`, `rng_state` |
| `TritonFlowInterface(DeviceInterface)` | the ~14 device-slot methods: `current_device`, `set_device`, `device_count`, `is_available`, `stream`, `current_stream`, `set_stream`, `synchronize`, `get_device_properties`, … plus `Event`, `Stream`, `Worker` nested classes; unimplemented slots explicitly delegate or raise `NotImplementedError` with a reason |
| `Emulator` | `emulate(program: Program, inputs: dict) -> dict`; `apply(instr: Instr)`; `precision_policy: PrecisionPolicy` |
| `PrecisionPolicy` | `input_precision: "tf32" \| "ieee"`, `accumulation_order`, `tolerance_for(dtype) -> float`, `derivation: str` |
| `FallbackRecord` | `op_name`, `reason`, `kind`, `loc_name` — every eager fallback, surfaced in the coverage report (FR-025) |

---

## 8. Report model (`report/`)

### CoverageReport (per tier, per ISA — FR-027, FR-028)

```json
{
  "isa": "tritonflow1", "tier": "t1_matmul", "kernel": "matmul",
  "fully_lowered": true,                    // boolean first, always
  "largest_lowered_subgraph": 0.94,
  "annotated_node_fraction": 0.97,          // labelled: upper bound
  "unsupported": [],
  "fallbacks": [],
  "generation_latency_ms": 41.2
}
```

### SelectionReport (per program — FR-029)

```json
{ "kernel": "matmul", "chosen_cost": 18432.0, "oracle_min_cost": 18432.0, "gap": 0.0,
  "choices": [ {"operand": "a_ptrs", "chosen": "DMA2D",
                "rejected": [{"name": "DMA1D", "by": "cost(1.00*words) > 0.60*words"}]} ] }
```

### TransferReport (per stage, per ISA — FR-030)

```json
{ "isa": "tritonflow2",
  "stages": [ {"stage": "parser", "transferred": true,  "edit": null},
              {"stage": "recogniser", "transferred": false,
               "edit": "isa/rules/tritonflow2.py:88 (stride-1 assumption)"} ],
  "transfer_rate": 0.86,
  "new_rules": 7, "new_rule_cost": 0.39,
  "edits_outside_schema_and_rules": ["recognize/walk.py:112"] }
```

`edits_outside_schema_and_rules` is the falsification field. It is expected to be non-empty in a first
attempt; what makes the experiment honest is that the field exists and is populated rather than the rate being
high.

---

## 9. Cross-entity invariants

1. Every `SsaValue` has exactly one definition; every operand resolves (parser).
2. Every in-loop memory operand is either `Ok(descriptor)` with a decidable `increment`, or explicitly
   `Unstructured`/`BudgetExhausted` — never `Ok` with an unknown increment.
3. Every `Instruction` in an emitted `Program` satisfied its own `constraint` against the operand it was
   chosen for (checked in the emitter, not assumed from the selector).
4. Every operation in the input module is either annotated in the emitted program or present in
   `unsupported`/`fallbacks`. The two sets must cover the module exactly — this is the machine-checkable form
   of Principle II.
5. Reports are generated from the data structures above, never assembled by hand in prose (Constitution,
   workflow rule 5).
