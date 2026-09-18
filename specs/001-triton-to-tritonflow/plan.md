# Implementation Plan: Triton-IR-to-Declarative-Toy-ISA

**Branch**: `001-triton-to-tritonflow` | **Date**: 2026-09-14 | **Spec**: `specs/001-triton-to-tritonflow/spec.md`
**Methodology of record**: `methodology-v2.tex` | **Audit**: `AUDIT.md` | **Research**: `research.md`

## Summary

Consume `ttir` text produced by `torch.compile`, recover structured memory accesses and the reduction
idiom, select the minimum-cost admissible lowering from a declarative ISA description, emit an instruction
stream, and execute it through a device emulator behind a registered `torch.compile` backend. Deliverable of
record: a registered device with a coverage report and a measured cross-ISA transfer report, in ~6.5 days.

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: `pyyaml` (schema), `numpy` (emulator). Pipeline runtime has **no** Triton/PyTorch
dependency: it consumes frozen fixtures.
**Test-only dependencies**: `pytest`, `triton==3.7.1` (fixture generation), `torch==2.12.1` (seam + eager
reference).
**Storage**: files — `.ttir` fixtures, `.yaml` schemas, `.txt` programs, `.json` reports.
**Testing**: `pytest`, with fixtures committed and never regenerated as a side effect (FR-034).
**Target Platform**: CPU-only Linux. GPU-free by construction: the extraction gate uses
`triton.compile(..., target=GPUTarget("cuda", 80, 32))`.
**Project Type**: single project — a library plus a CLI.
**Performance Goals**: generation latency < 2 s per kernel, reported as a one-time per-kernel-shape cost
(SC-007). No runtime performance goal: this is not a fast backend.
**Constraints**: no LLVM/MLIR build; deterministic output bytes; no silent miscompile; ~6.5 days.
**Scale/Scope**: 4 corpus kernels × 2 ISA descriptions, 5 ISA-1 instructions, 1 device.

## Constitution Check

*GATE: must pass before Phase 0. Re-checked after Phase 1 design. Items marked REVIEW are where the review
changed a requirement.*

| Principle | Gate | Status |
|---|---|---|
| I. Evidence or It Does Not Ship | Every FR/SC in `spec.md` cites a command, file, or fixture; every report field is reproducible | **PASS** — evidence column present throughout; `research.md` records the verification for each truth |
| II. Never Silently Miscompile | `UNSUPPORTED`/`PARSE_UNSUPPORTED` are in the data model, the contracts and the tests; the fuzz pass asserts "marker, never crash" | **PASS** — FR-001, FR-005, SC-008, contract invariants 2/4 |
| III. Declarative Means Falsifiable | Second ISA is a scheduled block with a measured transfer rate and a forced-edit list | **PASS** — US4, FR-030, SC-006, block 7 |
| IV. The Seam Is the Deliverable | `register_backend` + `DeviceInterface` are the artifact of record; the seam is proven on day 1 | **PASS** — FR-023…026, block 1b |
| V. Bounded Scope, Stated Bound | Limitations document generated from non-goals; coverage reported per tier and per ISA | **PASS** — FR-031, FR-027, `edge-cases.md` dispositions |

**Complexity tracking** (see table at the end): three justified violations, all recorded.

---

## Project Structure

### Documentation (this feature)

```text
specs/001-triton-to-tritonflow/
├── spec.md              # requirements and user stories
├── plan.md              # this file
├── research.md          # Phase 0: 10 grilled questions, truths, rejected alternatives
├── data-model.md        # Phase 1: entities, ISA schema, program format, reports
├── edge-cases.md        # the 100 cases with required behaviour and verification
├── quickstart.md        # Phase 1: run it end to end in five commands
├── contracts/           # Phase 1: one file per interface
│   ├── raw-module.md    # the parser/IR seam (Track A <-> Track B)
│   ├── ttir-parser.md
│   ├── access-descriptor.md
│   ├── isa-schema.md
│   ├── selector.md
│   ├── assembler.md
│   ├── emulator.md
│   ├── torch-seam.md
│   └── coverage-report.md
├── checklists/
│   └── requirements.md  # reviewer-owned requirements-quality review
└── tasks.md             # Phase 2 output
```

### Source Code (repository root)

```text
src/tritonflow/
├── __init__.py
├── cli.py
├── ttir/
│   ├── __init__.py
│   ├── lexer.py        # Track A
│   ├── parser.py       # Track A: parse_raw -> RawModule (text-level only)
│   ├── ssa.py          # Track B: the IR data structures
│   ├── types.py        # Track B: type text -> TypeExpr
│   ├── to_ir.py        # Track B: build_ir(RawModule) -> Module
│   └── graph.py        # Track B: def-use, region-aware traversal
├── canon/
│   ├── __init__.py
│   └── canonicalize.py
├── recognize/
│   ├── __init__.py
│   ├── descriptor.py
│   ├── op_shapes.py
│   └── walk.py
├── idioms/
│   ├── __init__.py
│   ├── patterns.py
│   └── detect.py
├── isa/
│   ├── __init__.py
│   ├── schema.py
│   ├── select.py
│   ├── rules/
│   │   ├── __init__.py
│   │   ├── tritonflow1.py
│   │   └── tritonflow2.py
│   └── schemas/
│       ├── tritonflow1.yaml
│       └── tritonflow2.yaml
├── emit/
│   ├── __init__.py
│   ├── ir.py
│   ├── assemble.py
│   └── disasm.py
├── emu/
│   ├── __init__.py
│   ├── exec.py
│   └── precision.py
├── torch_backend/
│   ├── __init__.py
│   ├── compiler.py
│   ├── device_interface.py
│   └── device.py
├── report/
│   ├── __init__.py
│   ├── coverage.py
│   └── transfer.py
└── harness/
    ├── __init__.py
    └── extract_fixtures.py     # dev-only; the only thing that needs Triton

fixtures/
├── raw/                  # as dumped, unnormalised, absolute source paths and all
├── t0_vecadd.ttir        # normalized: loc("<abs path>") -> loc("LOCFILE")
├── t1_matmul.ttir
├── t2_matmul_relu.ttir
├── t3_modulo.ttir
├── fuzz.ttir
├── ttgir_snapshot.txt
├── GOLDEN.json           # schema + provenance + observations + intent  (the canon)
└── VERSIONS.txt          # triton version + commit hash used to generate them

tests/
├── conftest.py      # canon fixtures, directory-derived markers, the two input classes
├── unit/            # per-module, on fixtures (parser edge cases, predicates, precision, serialisation)
├── contract/        # one file per contract in specs/.../contracts/  (test_fixtures.py is the canon)
├── integration/     # per tier, the seam, fuzzing, the second ISA
├── e2e/             # the frozen corpus end to end, coverage and numerical parity
└── data/            # .npz inputs for the differential tests

bench/                # T-L5: publishes, never gates
├── cases.yaml        # 4 tiers x 8 metrics = 32 rows, plus the fixed protocol values
├── adapter.py        # the only place bench touches the pipeline (track D)
├── run.py            # emits every row every run; never omits a row
├── report_table.py   # renders the results table for the report and the CI summary
└── results.json      # generated artifact

tools/
├── fixtures_lib.py        # normalizer + scanner; stdlib only
├── snapshot_fixtures.py   # --write / --golden / --check (CI gate)
└── check_test_map.py      # module<->test and contract<->test mapping; import hygiene

.github/workflows/ci.yml   # every job is a make target
Makefile                   # the single test/bench/CI entry point
pyproject.toml             # project + pytest markers + ruff config
requirements-dev.txt       # pytest, ruff, numpy, pyyaml — no Triton, no torch
```

**Test and CI contract**: normative in `docs/team/testing-ci.md`. Summary: no expectation is
hand-typed (everything comes from `fixtures/GOLDEN.json`); the test layer is fixed by the directory and
the marker is applied automatically; every contract test is parametrised over `handbuilt` and `real`
input classes with a day-3 deadline flag (`STRICT_REAL`); `make ci` is exactly what CI runs; and the
benchmark emits 32 rows every run without ever gating a merge.

**Structure Decision**: single project, library + CLI (`src/tritonflow`), because the artifact is one
pipeline and one seam; a multi-package split would add interfaces that nothing consumes. The `harness/`
subpackage is the only module that imports Triton, and it is excluded from the runtime import graph — the
pipeline imports it nowhere (asserted by a test that imports `tritonflow` with Triton uninstalled).

---

## Components and functions

Every module below lists its public functions with signatures, and the requirements each satisfies. Private
helpers are named where they carry a correctness rule.

### 1. `cli.py` — FR-032, SC-007

| Function | Signature | Behaviour |
|---|---|---|
| `main` | `main(argv: list[str] \| None = None) -> int` | dispatch; exit code 0/1/2 |
| `cmd_extract` | `cmd_extract(args) -> int` | regenerate fixtures (dev-only; refuses without `--force`) |
| `cmd_compile` | `cmd_compile(args) -> int` | `.ttir` + schema → serialised program, plus the selection report |
| `cmd_emulate` | `cmd_emulate(args) -> int` | program + inputs (`.npz`) → outputs, plus the parity result |
| `cmd_report` | `cmd_report(args) -> int` | corpus → coverage markdown/JSON |
| `cmd_transfer` | `cmd_transfer(args) -> int` | both ISAs → transfer markdown/JSON |
| `cmd_serve` | `cmd_serve(args) -> int` | import the seam so the device registers, for interactive debugging |

### 2. `ttir/` — FR-001 … FR-004, FR-019, FR-002

| Function | Signature | Behaviour |
|---|---|---|
| `lexer.tokenize` | `tokenize(text: str) -> Iterator[Token]` | tokens with line/col; never raises |
| `lexer.strip_comments` | `strip_comments(text: str) -> str` | keeps offsets aligned for diagnostics |
| `parser.parse_raw` | `parse_raw(text: str, *, source_path: str = "<string>") -> RawModule` | **Track A.** Total, syntactic only; produces `RawModule` (see `contracts/raw-module.md`) |
| `parser.parse_file_raw` | `parse_file_raw(path) -> RawModule` | as above, handling BOM/CRLF (EC-019, EC-020) |
| `parser.parse_loc_table` | `parse_loc_table(tokens) -> dict[str, RawLoc]` | `#loc` **syntax** (EC-006); binding a reference to the table is Track B's job |
| `parser.parse_attr_dict` | `parse_attr_dict(tokens) -> dict[str, RawAttr]` | nested dicts, values as text (EC-011) |
| `types.parse_type` | `parse_type(raw: str) -> TypeExpr` | **Track B.** shape/dtype/ptr-space, nested pointers (EC-012…014) |
| `to_ir.build_ir` | `build_ir(raw: RawModule, *, raise_on_invalid=False) -> ParseResult` | **Track B.** value numbering, region nesting, `loc` binding, structural-invalidity diagnostics |
| `to_ir.bind_results` | `bind_results(op: RawOp) -> list[SsaValue]` | **binds every result**, not the first (EC-004) |
| `ssa.Module.find_function` | `find_function(name) -> Function \| None` | |
| `ssa.Operation.results` | attribute | list; multi-result aware |
| `graph.build_def_use` | `build_def_use(module: Module) -> DefUseGraph` | one definition per value; operand edges |
| `graph.walk_region` | `walk_region(region) -> Iterator[Operation]` | region-aware depth-first; no flattening |
| `graph.topo_within_region` | `topo_within_region(region) -> list[Operation]` | per-region only — the T-4 guard |
| `graph.iter_loops` | `iter_loops(module) -> Iterator[LoopInfo]` | `LoopInfo{op, iv, lower, upper, step, iter_args, body}` |

**The parser/IR seam.** `parser.py` and `to_ir.py` are separate files owned by separate people (a parsing
specialist and an IR specialist). The boundary is `RawModule`, specified in `contracts/raw-module.md`: text
stays text in the parser (types, attributes, SSA names, no terminator special-casing) so that the IR-side
semantics live in exactly one place. This is what makes the critical path's first link parallel rather than a
1.8-day single-person bottleneck, and it is why the failure route is specified per layer (`layer="syntax"` vs
`layer="ir"` on every `PARSE_UNSUPPORTED` diagnostic).

### 3. `canon/canonicalize.py` — FR-012

| Function | Signature | Behaviour |
|---|---|---|
| `canonicalize` | `canonicalize(module: Module) -> Module` | idempotent; attribute-order and naming normalisation only |
| `canonical_form` | `canonical_form(module: Module) -> str` | stable textual form for determinism tests |
| `assert_no_reduction_claim` | `assert_no_reduction_claim(before, after) -> None` | guard: asserts idempotence, **not** a node-count drop (T-8-adjacent; v1's false premise) |

### 4. `recognize/` — FR-006 … FR-009

| Function | Signature | Behaviour |
|---|---|---|
| `descriptor.describe` | `describe(ptr, access, graph) -> DescriptorResult` | canonical descriptor or structured failure |
| `walk.resolve_operand` | `resolve_operand(value, graph) -> DescriptorResult` | **the loop-recurrence path** (EC-028) |
| `walk.substitute_iter_arg` | `substitute_iter_arg(value, graph) -> tuple[SsaValue, int \| None]` | (base, increment); `None` increments → `Unstructured` (EC-029) |
| `walk.fold_constant` | `fold_constant(value, graph) -> int \| None` | constant folding for the advance |
| `walk.BoundedWalker.visit` | `visit(op) -> None` | increments the hop counter; 33rd hop → `BudgetExhausted` (EC-042) |
| `op_shapes.is_memory_op` | `is_memory_op(name: str) -> bool` | `tt.load`, `tt.store` |
| `op_shapes.is_index_op` | `is_index_op(name: str) -> bool` | `tt.addptr`, `tt.splat`, `tt.broadcast`, `tt.expand_dims`, `arith.muli/addi`, `tt.make_range`, `tt.get_program_id` |
| `descriptor.conformance_check` | `conformance_check(module, oracle) -> ConformanceReport` | oracle comparison against `tts.make_tptr` semantics (FR-009) |
| `descriptor.descriptor_key` | `descriptor_key(d) -> str` | equality/caching, deterministic |
| `MAX_HOPS` | constant `32` | named, reported, never exceeded silently |

### 5. `idioms/` — FR-010, FR-011, FR-018

| Function | Signature | Behaviour |
|---|---|---|
| `detect.detect_mac` | `detect_mac(module, graph) -> list[MatchResult]` | all matches, not the first (EC-051); zero is valid (EC-049, EC-050) |
| `detect.detect_epilogue` | `detect_epilogue(module, graph, mac) -> list[MatchResult]` | add / relu classification (EC-056, EC-057) |
| `detect.detect_all` | `detect_all(module, graph) -> DetectionResult` | MAC + epilogue + unmatched inventory |
| `detect.annotate` | `annotate(module, detections) -> AnnotationSet` | annotate-don't-rewrite; the module is not mutated |
| `patterns.MAC_REQUIRED` | table | one `scf.for`, iter_args with an accumulator, exactly one `tt.dot` consuming it, one `scf.yield` re-threading it |
| `patterns.EPILOGUE_REQUIRED` | table | post-loop ops applied to the accumulator |
| `patterns.MatchResult.bindings` | attribute | `{a, b, acc, tile}` plus `input_precision` (T-6) |

### 6. `isa/` — FR-013 … FR-018

| Function | Signature | Behaviour |
|---|---|---|
| `schema.load_schema` | `load_schema(path) -> IsaSchema` | raises `SchemaError`; never defaults |
| `schema.validate_schema` | `validate_schema(schema) -> list[SchemaViolation]` | ≥2 per kind, cost/constraint/accumulate present (EC-066…070) |
| `schema.evaluate` | `evaluate(predicate, descriptor) -> True \| False \| "unknown"` | **fail-closed** (EC-062) |
| `schema.cost_of` | `cost_of(instr, descriptor, tile) -> float` | zero-dimension guard at load time (EC-064) |
| `select.enumerate_candidates` | `enumerate_candidates(schema, kind, descriptor, tile) -> list[Candidate]` | includes rejected, with the failing predicate |
| `select.select` | `select(schema, kind, descriptor, tile) -> SelectionReport` | min cost among admissible; no default (EC-063) |
| `select.oracle_min` | `oracle_min(schema, kind, descriptor, tile) -> SelectionReport` | exhaustive oracle; the gap source (EC-072) |
| `rules.tritonflow1.ISA_RULES` | `list[Rule]` | the only ISA-1-specific lowering logic |
| `rules.tritonflow2.ISA_RULES` | `list[Rule]` | authored from `data-model.md` §1.3 alone |

### 7. `emit/` — FR-019, FR-020, FR-005

| Function | Signature | Behaviour |
|---|---|---|
| `assemble.assemble` | `assemble(module, graph, annotations, schema) -> Program` | region-preserving (EC-073…080) |
| `assemble.order_regions` | `order_regions(module, graph) -> list[OrderedOp]` | region-aware ordering — **not** a flat topological sort (T-4) |
| `assemble.emit_instr` | `emit_instr(op, binding, schema, report) -> Instr \| UnsupportedMarker` | marker on any failure (FR-005) |
| `assemble.check_constraint` | `check_constraint(instr, operand) -> None` | independent re-validation (data-model invariant 3) |
| `disasm.serialize` | `serialize(program) -> str` | byte-stable header with `isa_name`, `schema_version`, `total_cost` |
| `disasm.deserialize` | `deserialize(text) -> Program` | inverse; refuses a version mismatch (EC-068) |
| `disasm.disassemble` | `disassemble(program) -> str` | human-readable, same information (EC-082) |

### 8. `emu/` — FR-021, FR-022

| Function | Signature | Behaviour |
|---|---|---|
| `exec.emulate` | `emulate(program, inputs, policy) -> dict[str, np.ndarray]` | consumes the emitted stream (EC-083…092) |
| `exec.apply` | `apply(instr, state, policy) -> None` | per-instruction; `UNSUPPORTED` raises `ProgramNotExecutable` (EC-087) |
| `precision.PrecisionPolicy.tolerance_for` | `tolerance_for(dtype) -> float` | exact for integer paths |
| `precision.derive_tolerance` | `derive_tolerance(dtype, k, precision) -> tuple[float, str]` | formula **and** its human-readable derivation |
| `precision.tf32_truncate` | `tf32_truncate(x: np.ndarray) -> np.ndarray` | mantissa truncation before multiply (EC-083) |
| `precision.accumulate` | `accumulate(values, order) -> float` | follows the ISA-declared order (EC-086, T-5) |
| `precision.compare` | `compare(actual, expected, policy) -> ParityResult` | exact for integer/low precision; within tolerance for float |

### 9. `torch_backend/` — FR-023 … FR-026, SC-001

| Function | Signature | Behaviour |
|---|---|---|
| `compiler.tritonflow_backend` | `tritonflow_backend(gm, example_inputs) -> Callable` | `@register_backend` entry point |
| `compiler.extract_ttir` | `extract_ttir(gm, example_inputs) -> str` | captured IR; `PARSE_UNSUPPORTED` handled, not raised (EC-096) |
| `compiler.lower_and_run` | `lower_and_run(ttir_text, inputs) -> list[torch.Tensor]` | pipeline → emulator → tensors |
| `compiler.fallback` | `fallback(gm, example_inputs) -> Callable` | eager path with a `FallbackRecord` (EC-097) |
| `device_interface.TritonFlowInterface` | class | the device slots; unimplemented ones raise with a reason (EC-099) |
| `device_interface.register_interface` | `register_interface() -> None` | `register_interface_for_device("tritonflow")` |
| `device.ToyDevice.allocate` | `allocate(nbytes) -> DevicePtr` | |
| `device.ToyDevice.copy_host_to_device` | `copy_host_to_device(src, dst) -> None` | |
| `device.ToyDevice.synchronize` | `synchronize() -> None` | no-op, documented as such |

### 10. `report/` — FR-027 … FR-031

| Function | Signature | Behaviour |
|---|---|---|
| `coverage.fully_lowered` | `fully_lowered(module, program) -> bool` | computed **before** any fraction (FR-027, SC-002) |
| `coverage.largest_lowered_subgraph` | `largest_lowered_subgraph(module, program) -> float` | connected subgraph fraction |
| `coverage.unsupported_inventory` | `unsupported_inventory(program) -> list[Unsupported]` | op, reason, originating `loc` |
| `coverage.coverage_report` | `coverage_report(runs) -> CoverageReport` | per ISA, per tier; never merged |
| `coverage.render_markdown` | `render_markdown(report) -> str` | prose generated from the table |
| `transfer.PIPELINE_STAGES` | list | exactly the 10 modules above, in order |
| `transfer.transfer_report` | `transfer_report(runs, edits) -> TransferReport` | raises if a stage is missing |
| `transfer.diff_edits` | `diff_edits(base, isa) -> list[Edit]` | git-diff-based: which files changed outside `isa/schemas/` and `isa/rules/` |
| `transfer.render_markdown` | `render_markdown(report) -> str` | |

### 11. `harness/extract_fixtures.py` — FR-034 (dev-only, the only Triton import)

| Function | Signature | Behaviour |
|---|---|---|
| `extract` | `extract(out_dir, *, force: bool = False) -> list[Path]` | writes the four fixtures + `ffuzz` + `VERSIONS.txt` |
| `TIER_KERNELS` | table | the four kernel sources, verbatim, so fixtures are reproducible |
| `snapshot_ttgir` | `snapshot_ttgir(out_dir) -> Path` | second IR level snapshot, for the record |

---

## Build order ("ways to build")

Three orders were considered; two were rejected on evidence.

| Order | Description | Verdict |
|---|---|---|
| A. **Schema-first** | write the ISA DSL, then the generator, then wire PyTorch last | **rejected** — this is v1's shape, and it defers the only integration risk to the end, where it cannot be absorbed |
| B. **Parser-first** | build the whole front end, then idioms, then emission, then the seam | **rejected** — the parser is the largest risk but it is *not* the largest uncertainty; a working seam changes what "done" means and can be proven in an hour |
| C. **Seam-first, then front end** (chosen) | day-1 smoke test with a hard-coded lowering → fixtures → parser → recognition → idioms → selection → emission → reporting → second ISA | the integration risk is retired on day 1; from then on every block increases the *fraction* of real work behind an already-working interface |

**Block mapping** (from the methodology's schedule, kept in sync; the table below is the normative copy and
`methodology-v2.tex` §`sec:schedule` must not disagree with it):

| Block | Deliverable | Maps to |
|---|---|---|
| 0 (2–3 h) | methodology corrections | this document set |
| 1 (0.5 d) | extraction gate, fixtures, `VERSIONS.txt` | `harness/extract_fixtures.py`, `fixtures/` |
| 1b (0.25 d) | **seam smoke test**: hard-coded 64×64 matmul runs on the registered device | `torch_backend/*`, `emu/exec.py` (hard-coded path) |
| 2 (1.5 d) | parser → def-use graph; recogniser incl. iter_args, `scf.yield`, `expand_dims`; Tier 0 resolves, Tier 3 does not | `ttir/*`, `recognize/*` |
| 3 (1 d) | MAC + epilogue detectors, annotation, true-negative test, loop-recurrence test | `idioms/*` |
| 4 (1 d) | assembly with variant selection and costs; emulator; `PARSE_UNSUPPORTED` path | `isa/*`, `emit/*`, `emu/*` |
| 5 (0.75 d) | wire the real pipeline behind the seam; run a real torch op | `torch_backend/compiler.py` |
| 6 (0.5 d) | coverage, selection-quality and limitations reports | `report/*` |
| 7 (0.5 d) | second ISA + transfer measurement | `isa/schemas/tritonflow2.yaml`, `isa/rules/tritonflow2.py`, `report/transfer.py` |

**Cut order** (published, per Constitution workflow rule 6): fuzz kernel → canonicalisation entirely →
block 7 → Tier-2 epilogue → Tier-3 fixture. **Never cut:** Tier-3 negative control, coverage table,
true-negative test, the seam.

---

## Complexity Tracking

| Violation | Why needed | Simpler alternative rejected because |
|---|---|---|
| Hand-rolled lexer/parser instead of a parser generator (e.g. `lark`) | The ttir text is semi-structured MLIR; the failure route must be total and must report line/column, and the parser must run with no Triton installed | A generator adds a grammar to maintain and still needs hand-written error recovery for `PARSE_UNSUPPORTED`; the grammar is small (≈18 op names in Tier 1) |
| Per-ISA rule module (`isa/rules/tritonflowN.py`) | Some lowering logic is genuinely ISA-specific; hiding it in `if isa == …` branches would make the transfer experiment meaningless | A single shared rule table would force ISA-2's differences into conditionals, which is exactly the "ISA-1 baked in" failure the experiment exists to detect |
| A second semantics implementation (the emulator) alongside the reference | Principle II requires executing the emitted stream, not re-deriving the computation; the differential test then compares against eager PyTorch rather than against another NumPy routine | Reusing a NumPy reference as "the emulator" would validate a shortcut, and the emitted program would never be executed — the validation would prove nothing about the artifact |

## Phase 1 → Phase 2 handoff

`data-model.md` fixes the entities; `contracts/` fixes every interface and its failure modes;
`edge-cases.md` fixes the behaviour of the 100 boundary cases; `tasks.md` sequences the work.
No implementation may begin on a module whose contract is not yet written (Constitution workflow rule 1).
