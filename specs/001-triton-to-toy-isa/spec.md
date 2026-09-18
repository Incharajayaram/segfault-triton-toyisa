# Feature Specification: Triton-IR-to-Declarative-Toy-ISA backend generator

**Feature Branch**: `001-triton-to-toy-isa`

**Created**: 2026-09-14

**Status**: Draft

**Input**: Methodology `methodology-v2.tex` + audit `AUDIT.md`; user description: "take a declarative
description of a tensor ISA and automatically emit a Triton-IR-consuming backend for it, registered with
PyTorch's `torch.compile`". Deliverable of record: a working vertical slice, with its transferability
*measured* rather than asserted.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - A real PyTorch operation runs on a generated device (Priority: P1)

A user writes ordinary PyTorch (`torch.compile`d matmul, then add/relu). The registered backend consumes the
Triton IR that `torch.compile` produces, lowers it through the pipeline, and executes it on the toy-ISA device
emulator. The result matches eager PyTorch within the declared tolerance. No bespoke harness is involved: the
entry points are `register_backend` and a `DeviceInterface` subclass.

**Why this priority**: It is the only story whose output a framework can actually invoke. It converts "emits
toy assembly" into "gives `torch.compile` a new device" — the claim worth making. It is also ~10 lines of
registration against a documented API with an in-tree template (`torch_openreg/compiler.py`).

**Independent Test**: A test that registers the backend, compiles a 64x64 matmul through
`torch.compile`, executes it, and compares against eager. Runs without touching any other story.

**Acceptance Scenarios**:

1. **Given** the backend registered and a toy-ISA device present, **When** `torch.compile(fn)` runs `fn` = a
   64x64 matmul, **Then** it executes on the toy device and matches within tolerance.
2. **Given** the same registration, **When** an operation the pipeline cannot lower is compiled, **Then** it
   runs through PyTorch's eager fallback and the coverage report lists it as `UNSUPPORTED` — never a wrong
   number.
3. **Given** `DeviceInterface` is registered, **When** `torch.tensor(...).to("tritonflow")` is called, **Then**
   the device is visible to PyTorch (`device_count`, `is_available`, `current_device` behave).

---

### User Story 2 - Structure is recognised, not reconstructed (Priority: P2)

Given a frozen `ttir` module, the pipeline recovers the def-use graph, resolves each memory operand to a
canonical access descriptor or `UNSUPPORTED`, and detects the reduction (MAC) and epilogue idioms. The
existing loop structure in the IR is how the reduction is found: operands carried through `scf.for`
`iter_args` and re-threaded by `scf.yield`.

**Why this priority**: This is the technical core, and the part that must work for any of the rest to mean
anything. It carries the project's largest single risk (the hand-rolled parser) and its most-corrected
specification.

**Independent Test**: Per-tier assertions on the frozen corpus: Tier 0/1/2 resolve to the expected descriptors
with exact field values; Tier 3 resolves to unstructured; the MAC pattern is found exactly once on Tiers 1–2
and zero times on Tiers 0 and 3.

**Acceptance Scenarios**:

1. **Given** the Tier-1 fixture, **When** recognition runs, **Then** both `tt.load` operands resolve to
   descriptors whose base/increment are recovered from the loop-carried values.
2. **Given** the Tier-0 fixture (vector add), **When** idiom detection runs, **Then** no MAC is reported
   (true negative).
3. **Given** the Tier-3 fixture (modulo wraparound), **When** recognition runs, **Then** the operand is
   reported unstructured and an `UNSUPPORTED` marker is produced with the originating `loc` name — not a
   crash.

---

### User Story 3 - Selection is a decision, not a template fill (Priority: P3)

The schema declares multiple instructions per kind, each with an admissibility predicate and a cost. For a
given operand the generator enumerates the admissible lowerings, rejects the inadmissible by predicate, and
emits the minimum-cost one. Selection quality is reported against an exhaustive oracle and against the
hand-written reference program.

**Why this priority**: Without alternatives, the generator has nothing to decide and the word "generate" is
unearned. This is cheap to add once recognition works and it converts the artifact from a template into a
decision procedure.

**Independent Test**: On a case where only `DMA1D` is admissible (unaligned stride), the emitted program uses
`DMA1D`; on a case where both are admissible, it uses the cheaper one; the oracle comparison is computed by
exhaustive enumeration and matches.

**Acceptance Scenarios**:

1. **Given** an operand whose stride is not divisible by 4, **When** selection runs, **Then** `DMA2D` is
   rejected by its constraint and `DMA1D` is emitted.
2. **Given** an operand where both DMA variants are admissible, **When** selection runs, **Then** the lower
   cost variant is emitted and the choice is recorded in the selection report.
3. **Given** the emitted program, **When** cost is computed, **Then** it is ≥ the oracle minimum, and the gap
   is reported.

---

### User Story 4 - The method transfers to an independently authored ISA (Priority: P4)

A second ISA description, differing in memory model, compute primitive, addressing and epilogue, is run through
the *unedited* pipeline. The transfer rate per stage and the new-rule cost are measured and reported, together
with an explicit list of every edit forced outside the schema and rule modules.

**Why this priority**: It is the only falsifiable test of the project's central word. It is half a day, and a
negative result is a reportable result. Without it, generality is asserted.

**Independent Test**: Run the pipeline on ISA-2's schema and corpus; assert that the report contains a
per-stage transfer row, a new-rule count, and that no edit outside `isa/schemas/` and one rule module was
needed.

**Acceptance Scenarios**:

1. **Given** the ISA-2 schema, **When** the unedited pipeline runs on an ISA-2 kernel, **Then** it either
   lowers it or emits `UNSUPPORTED` — never a parse crash.
2. **Given** the run, **When** the transfer report is generated, **Then** it names every stage that required
   an edit and the location of that edit.
3. **Given** a stage that did not transfer, **When** the limitations document is generated, **Then** the
   stage appears there with the reason.

---

### User Story 5 - The report is the artifact (Priority: P5)

Coverage is reported per tier and per ISA, as a boolean ("fully lowered") before any percentage, plus the
largest lowered subgraph, the `UNSUPPORTED`/`PARSE_UNSUPPORTED` inventory with originating variable names, and
the selection-quality and transfer tables. Prose is generated from the table.

**Why this priority**: It is what a reader actually reads, and the gameable-percentage failure of v1 is fixed
here. Cheap, but it must not be an afterthought.

**Independent Test**: Report generation over the corpus produces every required column; a kernel with one
early unlowered operation reports `fully lowered = false` even though its annotated-node percentage is high.

**Acceptance Scenarios**:

1. **Given** Tier 1, **When** the report is generated, **Then** it contains `fully lowered`, largest lowered
   subgraph, annotated fraction (labelled as an upper bound), and the unsupported inventory.
2. **Given** a kernel with an unlowerable operation at the head of a chain, **When** coverage is computed,
   **Then** `fully lowered` is false.

---

### Edge Cases

See `edge-cases.md` for the full register of 100 cases (EC-001 … EC-100), grouped as: ttir syntax and
parsing (27), descriptor recognition (20), idiom detection (10), ISA schema and selection (15), emission and
ordering (10), emulator and precision (10), PyTorch seam (8). Each case is either covered by a test or given
a documented disposition; none is silently ignored. Representative cases:

- Multi-result operations (`%acc_25:3 = scf.for ...`) — the parser must bind all results, not the first.
- `#loc` table indirection and `loc("...")` names containing underscores and quotes.
- A `tt.dot` with `inputPrecision = tf32` (mandatory in the corpus) and one with default precision.
- A reduction whose increment is loop-carried and *not* constant (out of scope — must degrade to
  `UNSUPPORTED`, not mislower).
- An operand that is both partially structured and partially gathered (Tier-3-like) — unstructured, no crash.
- Two candidate MAC patterns in one module (must report both, not the first).
- An ISA schema with an instruction whose constraint is unsatisfiable for every corpus operand (the selector
  must report "no admissible lowering", not silently pick one).
- An empty kernel, a kernel with no memory access, and a kernel whose only op is `tt.dot` on constants.

## Requirements *(mandatory)*

### Functional Requirements

**Input handling and safety**

- **FR-001**: The system MUST consume `ttir` text from a file or a string and produce either a module object
  or a `PARSE_UNSUPPORTED` diagnostic naming the line and column. It MUST NOT raise an unhandled exception on
  any input.
- **FR-002**: The parser MUST bind every result of a multi-result operation and MUST preserve `loc` names when
  present.
- **FR-003**: Dynamic LLVM/MLIR & JIT Integration Permitted. The restriction against building/using LLVM/MLIR has been removed. The system supports direct dynamic JIT compilation, native MLIR dialect passes, and LLVM/Triton toolchain integration for on-the-fly kernel extraction and lowering alongside Python traversal.
- **FR-004**: Every pipeline stage MUST be deterministic: identical input bytes produce identical output
  bytes, across processes and hash seeds.
- **FR-005**: Any operation or operand that cannot be lowered MUST become an explicit `UNSUPPORTED` marker
  carrying the originating `loc` name when available. No operation may be dropped or approximated silently.

**Recognition**

- **FR-006**: The system MUST resolve in-loop `tt.load` operands carried through `scf.for` `iter_args` and
  re-threaded by `scf.yield` to canonical access descriptors.
- **FR-007**: The access descriptor MUST carry base, sizes, strides, offsets, shape and order — the fields a
  conformant structured descriptor needs — not a three-field `{base, stride, shape}`.
- **FR-008**: The recogniser MUST be a bounded walk with an explicit hop budget, and MUST report the budget
  exhausted as `UNSUPPORTED` rather than continuing.
- **FR-009**: The recogniser MUST be conformance-tested against `tts.make_tptr` semantics on the same
  fixtures, as an oracle for behaviour and not as a runtime dependency.
- **FR-010**: The system MUST detect reductions matching the `tt.dot`-in-`scf.for` pattern and MUST report
  zero matches on kernels with no `tt.dot` (true negative).
- **FR-011**: The system MUST detect the post-loop elementwise epilogue and MAY classify it as add or relu.
- **FR-012**: Canonicalisation MUST be idempotent and MUST NOT claim a reduction magnitude; it exists for
  vocabulary closure only.

**ISA description and selection**

- **FR-013**: The ISA description MUST be a declarative document (YAML) supporting at least: a data model, a
  set of instruction declarations, and a schema version.
- **FR-014**: Each instruction MUST declare kind, addressing form, an admissibility predicate over tile shape
  / stride / alignment, and a cost.
- **FR-015**: The schema MUST declare at least two instructions of each of the DMA and MAC kinds and one
  elementwise instruction.
- **FR-016**: Each compute instruction MUST declare accumulator precision, reduce dimension and accumulation
  order.
- **FR-017**: The selector MUST enumerate admissible lowerings, reject inadmissible ones by predicate, and
  emit the minimum-cost admissible lowering; when none is admissible it MUST emit `UNSUPPORTED`, not a
  default.
- **FR-018**: The selector MUST record the chosen lowering, the rejected alternatives and the reason for each
  rejection.

**Emission and execution**

- **FR-019**: Emission MUST preserve region structure: operations that lie inside a loop MUST be emitted
  inside that loop, and loop-carried values MUST remain loop-carried.
- **FR-020**: The emitted instruction stream MUST be serialisable to a stable text form and re-readable
  (round-trip).
- **FR-021**: The device emulator MUST execute the emitted instruction stream, not a shortcut around it, and
  MUST implement the declared accumulator precision and accumulation order.
- **FR-022**: The emulator MUST expose a deterministic float tolerance derived per dtype, with the derivation
  recorded in the report; integer paths MUST match exactly.

**PyTorch seam**

- **FR-023**: The system MUST register a `torch.compile` backend via `torch._dynamo.backends.registry.register_backend`.
- **FR-024**: The system MUST register an implementation of `torch._dynamo.device_interface.DeviceInterface`
  for a device name of its own, with the method slots it needs implemented and the remainder explicitly
  delegated or stubbed.
- **FR-025**: Operations the pipeline cannot lower MUST fall back to PyTorch eager execution, and the fallback
  MUST be visible in the coverage report.
- **FR-026**: The seam MUST be proven end to end on day 1 with a hard-coded lowering before the general
  pipeline is wired to it.

**Reporting**

- **FR-027**: Coverage MUST be reported per tier and per ISA as `fully lowered` (boolean) before any
  percentage.
- **FR-028**: The report MUST include the largest lowered subgraph, the annotated-node fraction labelled as an
  upper bound, and the unsupported inventory with originating `loc` names.
- **FR-029**: The report MUST include selection quality against an exhaustive oracle and against the
  hand-written reference program.
- **FR-030**: The report MUST include a per-stage cross-ISA transfer table with the edit location for every
  stage that did not transfer.
- **FR-031**: The limitations document MUST be generated from the explicit non-goals and MUST cross-reference,
  per limitation, which source work solves it at full generality.

**Process**

- **FR-032**: Every quantitative claim in project documents MUST carry its evidence (command, file and line,
  or fixture) — see Constitution Principle I.
- **FR-033**: The corpus MUST include a negative-control kernel (Tier 3, modulo wraparound) that must NOT
  satisfy the reduction pattern, and a Tier 0 kernel that must NOT satisfy it either.
- **FR-034**: Fixtures MUST be frozen against a pinned Triton version and MUST NOT be regenerated as a side
  effect of running tests.
- **FR-035**: The project MUST state, in writing, that it is a Python tool operating on `ttir` text and does
  not plug into Triton's compiler pipeline.

### Key Entities

- **Module**: a parsed `ttir` unit; contains functions; carries the `#loc` table.
- **Operation**: name, operands, results, attributes, parent region, optional `loc`. Multi-result aware.
- **Region / Block**: nesting structure. Regions are not flattened; loop bodies stay loop bodies.
- **SsaValue**: a definition with its type string and its defining operation.
- **DefUseGraph**: SSA values as nodes, operand edges; region-aware traversal order.
- **AccessDescriptor**: `{base, sizes, strides, offsets, shape, order, dtype}` — canonical form of a memory
  operand; or `Unstructured(reason)` or `BudgetExhausted(hops)`.
- **MatchResult**: which idiom matched, on which operations, with which operand bindings; or no-match.
- **Instruction** (schema side): name, kind, addressing form, constraint predicate, cost, accumulator
  semantics.
- **IsaSchema**: versioned set of instructions plus the data model.
- **Instr / Program**: the emitted stream — opcode, operands, loop nesting, cost.
- **SelectionReport**: chosen lowering, rejected alternatives, rejection reasons, total cost, oracle cost.
- **CoverageReport**: per tier and per ISA: `fully_lowered`, largest lowered subgraph, annotated fraction,
  unsupported inventory.
- **TransferReport**: per stage: transferred-or-edited, edit location; plus transfer rate and new-rule cost.
- **ToyDevice**: the emulated device behind `DeviceInterface` — storage, streams, events, RNG.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A 64x64 `torch.compile`d matmul executes on the registered toy device and matches eager PyTorch
  within the recorded per-dtype tolerance. Verified by an integration test that runs on every commit.
- **SC-002**: Tiers 0–2 report `fully lowered = true`. Tier 3 reports `fully lowered = false`, at least one
  `UNSUPPORTED` marker, and zero crashes/zero unhandled exceptions.
- **SC-003**: The reduction idiom is detected exactly once on Tiers 1 and 2, and zero times on Tiers 0 and 3
  (two independent true-negative tests).
- **SC-004**: Loop-carried operands resolve to descriptors whose fields equal the values recovered by hand from
  the fixture, for every in-loop operand of Tiers 1–2 (100% of operands, not a fraction).
- **SC-005**: On every case where both DMA variants are admissible, the emitted lowering's cost is ≥ the
  exhaustive oracle minimum, and the gap is reported. Emitted cost equals the oracle minimum wherever the
  greedy rule is optimal.
- **SC-006**: ISA-2 runs through the unedited pipeline; the transfer report is produced with a per-stage row
  for every stage and a named location for every forced edit. The numbers are whatever they are — the
  criterion is that they are produced.
- **SC-007**: Generation latency is under 2 s per kernel, recorded per kernel as a one-time per-shape cost.
- **SC-008**: No input in the corpus or in the fuzz set produces a crash: every failure appears as
  `UNSUPPORTED` or `PARSE_UNSUPPORTED`. Verified by a fuzz pass over malformed and adversarial `ttir`.
- **SC-009**: 100% of the cases in `edge-cases.md` have either an automated test or a written disposition in
  the limitations document; none is unaddressed.

## Assumptions

- **Pinned environment**: Triton 3.7.1, PyTorch 2.12.1, NumPy, Python 3.11+. Fixture generation happens once
  against this pin and the version hash is recorded.
- **GPU-free**: the extraction gate uses `triton.compile(..., target=GPUTarget("cuda", 80, 32))` with no
  driver present. Verified working.
- **Self-referential risk accepted and measured**: the ISA and the generator share an author. This is not
  eliminated; it is measured by US4.
- **`OpenReg` is the seam template**: an in-tree, official, stub-level reference for device registration and a
  `torch.compile` backend. Copied in structure, not in content.
- **Eager fallback exists**: the compiled path raises the floor but does not remove the operator library. The
  project attacks the ceiling, not the floor.
- **Scope**: four corpus kernels, two ISA descriptions, one device. Not a competitor to ACT or Triton-MTIA.
- **Budget**: approximately 6.5 working days, not 5; the published cut order applies under a hard cap.
- **Open item (US4 / schedule block 7)**: the ISA-2 corpus — [NEEDS CLARIFICATION: re-express the existing
  four kernels in the second ISA, or author two new kernels for it? Default: re-express all four, plus one new
  kernel if budget allows. Decided at block 7 and recorded in `research.md`.]
