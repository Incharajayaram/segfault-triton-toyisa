---
description: "Task list for 001-triton-to-toy-isa"
---

# Tasks: Triton-IR-to-Declarative-Toy-ISA

**Input**: Design documents from `specs/001-triton-to-toy-isa/`
**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/`, `edge-cases.md`

**Tests**: Required. This spec requests tests explicitly (SC-002, SC-003, SC-008, SC-009), so test tasks are
included and are written **before** the implementation they cover.

**Organization**: by user story, so each story is independently completable and testable.

**Format**: `[ID] [P?] [Story] Description` — `[P]` = parallelisable (different files, no dependency).

## Path Conventions

Single project. Source: `src/triton_tritonflow/`. Tests: `tests/`. Specs: `specs/001-triton-to-toy-isa/`.

---

## Phase 0: Pre-build infrastructure — DONE

Landed before day 1, so the four tracks start against one set of numbers. These are outside the original
80-task count; the normative spec is `docs/team/testing-ci.md`.

- [x] T-P0-1 `tools/fixtures_lib.py` — normalizer (absolute `loc` paths -> `LOCFILE`) and stdlib scanner
- [x] T-P0-2 `tools/snapshot_fixtures.py` — `--write` / `--golden` / `--check`, with the two-block canon (`observations` machine-derived, `intent` reviewed) and a provenance guard that refuses to re-derive across a Triton upgrade
- [x] T-P0-3 `fixtures/raw/*.ttir` + normalized `fixtures/*.ttir` + `fixtures/GOLDEN.json` + `VERSIONS.txt` — 4 fixtures, 182 operations, triton 3.7.1
- [x] T-P0-4 `pyproject.toml`, `Makefile`, `requirements-dev.txt` — one entry point (`make ci`), pytest markers, ruff config; dev deps exclude Triton and torch
- [x] T-P0-5 `tests/conftest.py` — canon fixtures, directory-derived markers, and the `handbuilt`/`real` two-input-class gate with `--strict-real`
- [x] T-P0-6 `tests/contract/test_fixtures.py` — the canon contract test; 30 assertions over 4 fixtures, green from hour 0
- [x] T-P0-7 `tools/check_test_map.py`, `bench/{cases.yaml,adapter.py,run.py,report_table.py}`, `.github/workflows/ci.yml` — mapping laws, the 32-row benchmark (all rows `unavailable` with a reason until the pipeline lands), and the CI jobs

**Checkpoint**: `make golden-check` and `make test-contract` pass; `make test-map` reports 9 pending contract
tests and zero violations; `make bench` writes 32 rows. Nothing downstream is blocked on a missing upstream
stage.

---

## Phase 1: Setup

- [ ] T001 Create the package skeleton (`src/triton_tritonflow/{ttir,canon,recognize,idioms,isa,emit,emu,torch_backend,report,harness}/__init__.py`) with an `__init__.py` that imports nothing heavy
- [ ] T002 [P] Add `pyproject.toml` with runtime deps `pyyaml`, `numpy` and dev deps `pytest`, `triton==3.7.1`, `torch==2.12.1`; pin the versions in `fixtures/VERSIONS.txt` format
- [ ] T003 [P] Add `tests/` skeleton (`unit/`, `contract/`, `integration/`) and `pytest.ini` with `testpaths`
- [ ] T004 [P] Add a runtime-import guard test in `tests/unit/test_no_triton_runtime.py` asserting `import triton_tritonflow` succeeds with Triton absent from `sys.modules`
- [ ] T005 [P] Record the pinned environment in `fixtures/VERSIONS.txt` (triton version, commit hash, torch version, date) — done in T-P0-3; keep it updated on any regeneration

## Phase 2: Foundational (blocking prerequisites)

**Purpose**: the frozen interface and the seam smoke test. No user story may start before this completes.

- [ ] T006 Implement `harness/extract_fixtures.py:extract()` and `TIER_KERNELS` with the four kernels (T0 vector add, T1 matmul, T2 matmul+relu, T3 modulo) and the fuzz kernel
- [ ] T007 Generate `fixtures/t0_vecadd.ttir`, `t1_matmul.ttir`, `t2_matmul_relu.ttir`, `t3_modulo.ttir`, `fuzz.ttir`, `ttgir_snapshot.txt`, `VERSIONS.txt`
- [ ] T008 [P] Add `tests/contract/test_fixtures.py` asserting the recorded structural facts: T0 has 0 `tt.dot` and 0 `scf.for`; T1 has exactly 1 `scf.for`, 1 `iter_args`, 1 `tt.dot`, 1 `scf.yield`; T3 contains `arith.remsi`; T1 contains `inputPrecision = tf32`
- [ ] T009 Implement `emu/precision.py` (`PrecisionPolicy`, `derive_tolerance`, `tf32_truncate`, `accumulate`, `compare`) with unit tests in `tests/unit/test_precision.py`
- [ ] T010 Implement `emit/ir.py` data structures (`Instr`, `Loop`, `Program`, `Operand`, `UnsupportedMarker`)
- [ ] T011 Implement `emu/exec.py:emulate`/`apply` over `emit/ir.py` structures, including `ProgramNotExecutable` (EC-087)
- [ ] T012 Implement `torch_backend/device.py` (`ToyDevice`, `DevicePtr`, allocate/copy/synchronize)
- [ ] T013 Implement `torch_backend/device_interface.py` (`ToyIsaInterface`, `register_interface`) with the slot inventory recorded in `tests/contract/test_device_interface.py`
- [ ] T014 Implement `torch_backend/compiler.py:register`/`tritonflow_backend` with a **hard-coded** 64×64 matmul lowering (the smoke path)
- [ ] T015 Write `tests/integration/test_torch_seam.py::test_smoke_hardcoded_matmul` — the day-1 gate: a real `torch` op compiles and runs through the registered device (SC-001, FR-026)

**Checkpoint**: the seam works with a hard-coded lowering. Every later block fills in real work behind a
working interface.

---

## Phase 3: User Story 1 — A real PyTorch operation runs on a generated device (P1) 🎯 MVP

**Goal**: the hard-coded path is replaced by the real pipeline end to end, with explicit eager fallback.
**Independent Test**: `tests/integration/test_torch_seam.py` — matmul, vector add, matmul+relu, and an
unlowerable op (fallback), all comparing against eager within the derived tolerance.

### Tests first

- [ ] T016 [P] [US1] `tests/integration/test_torch_seam.py::test_vecadd_roundtrip`, `::test_matmul_roundtrip`, `::test_matmul_relu_roundtrip` (expected to fail until T040–T046)
- [ ] T017 [P] [US1] `tests/integration/test_torch_seam.py::test_unsupported_falls_back_with_correct_result` and `::test_fallback_recorded` (EC-097, FR-025)
- [ ] T018 [P] [US1] `tests/integration/test_torch_seam.py::test_determinism_twice` (EC-098)

### Implementation

- [ ] T019 [US1] Implement `compiler.extract_ttir` capturing the `ttir` for a compiled graph, with `PARSE_UNSUPPORTED` handled rather than raised (EC-096)
- [ ] T020 [US1] Implement `compiler.lower_and_run` wiring pipeline → emulator → `torch.Tensor` outputs
- [ ] T021 [US1] Implement `compiler.fallback` returning an eager callable plus a `FallbackRecord`
- [ ] T022 [US1] Implement the fallback records surfacing in `report/coverage.py` (consumed by US5)
- [ ] T023 [US1] `tests/contract/test_device_interface.py`: device visible (`device_count`, `is_available`, `current_device`/`set_device` round-trip), `.to("tritonflow")` (EC-094, EC-095)
- [ ] T024 [US1] Assert unimplemented `Stream`/`Event` slots raise `NotImplementedError` with a reason, and that each reason is listed in the limitations document (EC-099)

**Checkpoint**: US1 is demonstrable on its own: a real `torch.compile` matmul runs on a generated device.

---

## Phase 4: User Story 2 — Structure is recognised, not reconstructed (P2)

**Goal**: `ttir` → def-use graph → descriptors → MAC and epilogue idioms, with both true negatives.
**Independent Test**: per-tier assertions over the frozen corpus.

### Tests first

- [ ] T025 [P] [US2] `tests/unit/test_parser_multi_result.py` — `%acc_25:3` binds 3 results (EC-004, EC-005)
- [ ] T026 [P] [US2] `tests/unit/test_parser_loc.py` — `#loc` table, inline `loc`, path form, escaped quotes, underscore names (EC-006…EC-010)
- [ ] T027 [P] [US2] `tests/unit/test_parser_regions.py` — `scf.if` with results, multi-block CFG, multi-operand `scf.yield` (EC-015…EC-017)
- [ ] T028 [P] [US2] `tests/unit/test_parser_negative.py` — ≥30 malformed inputs, each asserting `PARSE_UNSUPPORTED` and no exception (SC-008). **Split by layer, per `contracts/raw-module.md`**: `test_parser_syntax_negative.py` (lexer/parser: EC-001…EC-004, EC-011…EC-025, EC-027) and `test_parser_semantic_negative.py` (`build_ir`: EC-005, EC-026, type text, `loc` lookup). The `layer` field on the diagnostic names the owning track
- [ ] T029 [P] [US2] `tests/contract/test_descriptor.py` — the required-results table: T0/T1/T2 `Ok` with exact fields, T3 `Unstructured` (EC-028…EC-047)
- [ ] T030 [P] [US2] `tests/unit/test_walk_budget.py` — 40-deep chain → `BudgetExhausted(hops=32)` (EC-042)
- [ ] T031 [P] [US2] `tests/integration/test_true_negative.py` — MAC detected 0 times on T0 and T3 (EC-049, EC-050, SC-003)
- [ ] T032 [P] [US2] `tests/contract/test_idioms.py` — exactly one match on T1/T2; two matches when two dots exist (EC-048, EC-051…EC-057)
- [ ] T033 [P] [US2] `tests/unit/test_loop_recurrence.py` — in-loop `tt.load` via `iter_args` resolves with `increment == 32` (the mandatory regression, EC-028)

### Implementation

- [ ] T034 [US2] `ttir/lexer.py`: `tokenize`, `strip_comments`; `ttir/parser.py`: `parse_raw` → `RawModule` (**text-level only**, per `contracts/raw-module.md`)
- [ ] T035 [US2] `ttir/ssa.py`: `Module`, `Function`, `Operation`, `Region`, `Block`, `SsaValue`, `TypeExpr`, `Loc`, `Attr`; `ttir/types.py`: `parse_type` (EC-012…EC-014)
- [ ] T036 [US2] `ttir/parser.py`: `parse_attr_dict`, `parse_loc_table` (syntax; EC-011, EC-022) — plus the `RawModule` dataclasses committed on day 1 before lunch
- [ ] T037 [US2] `ttir/to_ir.py`: `build_ir` + `bind_results` — SSA resolution, multi-result binding, region nesting, `loc` binding, and the **ir-layer** diagnostics (EC-005, EC-026), plus the total failure route for EC-001…EC-027
- [ ] T038 [US2] `ttir/graph.py`: `build_def_use`, `walk_region`, `topo_within_region`, `iter_loops`
- [ ] T039 [US2] `canon/canonicalize.py`: idempotent canonicalisation with the no-reduction-claim guard (FR-012; evidence in `research.md` R4)
- [ ] T040 [US2] `recognize/op_shapes.py`: the op-shape tables and predicates
- [ ] T041 [US2] `recognize/walk.py`: `BoundedWalker`, `resolve_operand`, `substitute_iter_arg`, `fold_constant`
- [ ] T042 [US2] `recognize/descriptor.py`: `describe`, `descriptor_key`, `DescriptorResult` union
- [ ] T043 [US2] `recognize/descriptor.py`: `conformance_check` against `tts.make_tptr` semantics (FR-009) + `tests/contract/test_make_tptr_conformance.py`
- [ ] T044 [US2] `idioms/patterns.py`: `MAC_REQUIRED`, `EPILOGUE_REQUIRED`, `MatchResult` with `input_precision`
- [ ] T045 [US2] `idioms/detect.py`: `detect_mac`, `detect_epilogue`, `detect_all`
- [ ] T046 [US2] `idioms/detect.py`: `annotate` producing an `AnnotationSet` without mutating the module

**Checkpoint**: parsing, recognition and idiom detection are complete and independently testable; US1's
end-to-end test can now use the real front end (T019–T021 unblock).

---

## Phase 5: User Story 3 — Selection is a decision, not a template fill (P3)

**Goal**: a schema with alternatives, constraints and costs, and a selector that earns the word "generate".
**Independent Test**: `tests/contract/test_selector.py` — the required-results table, plus oracle agreement.

### Tests first

- [ ] T047 [P] [US3] `tests/contract/test_schema.py` — ISA-1 and ISA-2 validate; six broken schemas produce the specific violations (EC-064, EC-066…EC-070)
- [ ] T048 [P] [US3] `tests/unit/test_predicate_language.py` — the §1.2 predicate table, including the `unknown` cases (EC-062)
- [ ] T049 [P] [US3] `tests/contract/test_selector.py` — required results: both admissible → cheaper; stride not divisible → `DMA1D`; `m % 16 != 0` → `MAC8`; nothing admissible → `UNSUPPORTED` (EC-058…EC-063)
- [ ] T050 [P] [US3] `tests/contract/test_selector.py::test_gap_reported_when_greedy_suboptimal` (EC-072, SC-005)
- [ ] T051 [P] [US3] `tests/contract/test_selector.py::test_oracle_agreement_1000_random` — property test

### Implementation

- [ ] T052 [US3] `isa/schemas/tritonflow1.yaml` — the five instructions of `data-model.md` §1.1
- [ ] T053 [US3] `isa/schema.py`: `IsaSchema`, `Instruction`, `load_schema`, `validate_schema`, `SchemaError`
- [ ] T054 [US3] `isa/schema.py`: `evaluate` (fail-closed) and `cost_of` (zero-dimension guard)
- [ ] T055 [US3] `isa/select.py`: `Candidate`, `SelectionReport`, `enumerate_candidates`, `select`
- [ ] T056 [US3] `isa/select.py`: `oracle_min` (exhaustive) and the gap computation
- [ ] T057 [US3] `isa/rules/tritonflow1.py`: `ISA_RULES` for DMA/MAC/EPI
- [ ] T058 [US3] `emit/assemble.py`: `order_regions` (region-aware), `emit_instr`, `check_constraint`
- [ ] T059 [US3] `emit/disasm.py`: `serialize`, `deserialize`, `disassemble` + `tests/unit/test_serialize.py::test_roundtrip`

**Checkpoint**: a program is emitted with a recorded, auditable selection; the word "generate" is earned.

---

## Phase 6: User Story 4 — The method transfers to an independently authored ISA (P4)

**Goal**: ISA-2 authored from `data-model.md` §1.3 alone, run through the unedited pipeline, transfer measured.
**Independent Test**: `tests/integration/test_isa2.py` plus the transfer report contents.

### Tests first

- [ ] T060 [P] [US4] `tests/integration/test_isa2.py` — an ISA-2 kernel lowers or is marked `UNSUPPORTED`; never a parse crash (EC-071)
- [ ] T061 [P] [US4] `tests/contract/test_transfer_report.py` — every stage present, `edits_outside_schema_and_rules` is a required field, a missing stage raises (FR-030)

### Implementation

- [ ] T062 [US4] `isa/schemas/tritonflow2.yaml` — scratchpad + accumulator banks, `OPU`/`CONV`, strided 2-D DMA, `CLAMP`, `k_blocked(4)` order
- [ ] T063 [US4] `isa/rules/tritonflow2.py` — authored without reading `isa/rules/tritonflow1.py` (record the fact in the commit message)
- [ ] T064 [US4] `report/transfer.py`: `PIPELINE_STAGES`, `transfer_report`, `diff_edits` (git-diff based), `render_markdown`
- [ ] T065 [US4] Produce the transfer report for ISA-2 and record every edit outside `isa/schemas/` and `isa/rules/` — including an empty list only with a stage-by-stage justification
- [ ] T066 [US4] Resolve the open item from `research.md`: re-express the four kernels in ISA-2 (default) plus one new kernel if budget allows

**Checkpoint**: the falsifiable claim is measured; the result is recorded whether or not it is flattering.

---

## Phase 7: User Story 5 — The report is the artifact (P5)

- [ ] T067 [P] [US5] `tests/contract/test_reports.py` — boolean-before-fraction ordering; the `upper bound` label present; the gameable fixture reporting `fully_lowered = false` with a high fraction (FR-027, SC-002)
- [ ] T068 [P] [US5] `tests/contract/test_reports.py::test_golden_markdown` — checked-in snapshot so metric changes are visible in review
- [ ] T069 [US5] `report/coverage.py`: `fully_lowered`, `largest_lowered_subgraph`, `unsupported_inventory`, `coverage_report`, `render_markdown`
- [ ] T070 [US5] `report/coverage.py`: selection-quality table (chosen cost, oracle minimum, gap, rejected alternatives)
- [ ] T071 [US5] Generate the limitations document from the explicit non-goals, each cross-referenced to the source work that solves it at full generality (FR-031)
- [ ] T072 [US5] `cli.py`: `main`, `cmd_compile`, `cmd_emulate`, `cmd_report`, `cmd_transfer`, `cmd_extract`, `cmd_serve`

**Checkpoint**: the numbers a reader will quote are generated from data and cannot be assembled by hand.

---

## Phase 8: Polish & cross-cutting

- [ ] T073 [P] Fuzz pass: `tests/integration/test_fuzz.py` drives EC-001…EC-100 inputs, asserting "marker, never crash" (SC-008, SC-009)
- [ ] T074 [P] Determinism suite: parse/compile/serialize 3× in separate processes with different `PYTHONHASHSEED`, byte-compare (EC-092, FR-004)
- [ ] T075 [P] Edge-case traceability test: every EC ID in `edge-cases.md` maps to a test node ID or a `LIMITATIONS` disposition (SC-009)
- [ ] T076 [P] `quickstart.md` validation: run the five commands on a clean checkout
- [ ] T077 [P] Update the evidence appendix: every quantitative claim in `methodology-v2.tex` and in the reports carries its reproduction command (Constitution Principle I, FR-032)
- [ ] T078 [P] Write the write-up from the reports: coverage table, selection table, transfer table, limitations document — no number entered by hand; and state in writing that the artifact is a Python tool over `ttir` text that does not plug into Triton's pipeline (FR-035)
- [ ] T079 Review `checklists/requirements.md` and mark each item only after the reviewer confirms it
- [ ] T080 Apply the published cut order if the budget is exceeded, and record every drop in the limitations document
- [ ] T081 **End of day 3**: flip `STRICT_REAL` to `"1"` in `.github/workflows/ci.yml`. Every `real` row in the suite stops being an `xfail` and becomes a hard failure, and `check_test_map.py --strict` turns on (R5 laws 1–2). If a `real` row still fails after the flip, the missing stage is the blocker — not the flag
- [ ] T082 Confirm every bench row reports `status: "ok"` in `bench/results.json`, or that the row is `unavailable` for a stated reason. A row that is silently absent is the one failure mode the runner cannot see, so check by count: 32 rows expected

---

## Dependencies & Execution Order

### Phase dependencies

- **Setup (1)**: no dependencies.
- **Foundational (2)**: blocks everything. T006–T007 before T008; T009–T011 before T015.
- **US1 (3)**: after Phase 2. Its tests are written first and fail until Phase 4/5 fill in the real lowering.
- **US2 (4)**: after Phase 2. Unblocks US1's real pipeline.
- **US3 (5)**: after US2 (needs descriptors to select for).
- **US4 (6)**: after US3 (needs the pipeline and the selector to be ISA-agnostic in fact).
- **US5 (7)**: after US3 for coverage; the transfer section after US4.
- **Polish (8)**: after all stories.

### Within each story

- Tests are written first and must fail before implementation.
- Data structures before functions; functions before integration; contract tests before the write-up.

### Parallel opportunities

- All of Phase 1 after T001.
- T025–T033 (US2 tests) in parallel; T047–T051 (US3 tests) in parallel; T060–T061 in parallel.
- T073–T078 (Phase 8) are independent files and can run together.
- `isa/schemas/tritonflow2.yaml` (T062) must be authored before `isa/rules/tritonflow2.py` (T063), and deliberately
  without reading `isa/rules/tritonflow1.py` — that ordering is part of the experiment's validity, not a
  convenience.

## Implementation Strategy

**MVP first**: Phases 1–3 deliver a working registered device with a real lowering path for one kernel. Stop
and validate against `quickstart.md` before continuing — that is the earliest point at which the claim
("gives `torch.compile` a new device") is demonstrable.

**Incremental delivery**: each subsequent phase adds value behind the already-working interface: US2 widens
the set of kernels, US3 makes the choice real, US4 tests the word "declarative", US5 makes the numbers
quotable. No phase invalidates an earlier one.

**Under a hard cap**: follow the published cut order; record every drop; never cut the four items listed in
`plan.md`.

---

## Requirement traceability

Every requirement in `spec.md` maps to at least one task above. This table is the reviewer's check for
CHK018; it is generated by reading the task descriptions, and it is re-read whenever a task is added.

| Requirement | Tasks |
|---|---|
| FR-001 total parse with a diagnostic | T019, T028, T034, T037, T073 |
| FR-002 multi-result binding, `loc` preservation | T025, T026, T035, T036, T037 |
| FR-003 dynamic LLVM/MLIR & JIT integration enabled | `src/triton_tritonflow/extract/dynamic_extract.py`, `src/triton_tritonflow/extract/flaggems_bridge.py`, `tests/unit/test_no_triton_runtime.py` (T004), `verify/verify_mlir_bindings.py` |
| FR-004 determinism | T018, T028, T074 |
| FR-005 explicit `UNSUPPORTED`, never dropped | T010, T046, T058, T073 |
| FR-006 loop-carried operands resolve | T029, T033, T041 |
| FR-007 full descriptor field set | T029, T041, T042 |
| FR-008 bounded walk with a reported budget | T029, T030, T040, T041, T042 |
| FR-009 conformance against `tts.make_tptr` semantics | T043 |
| FR-010 reduction detection + true negative | T031, T032, T044, T045 |
| FR-011 epilogue detection and classification | T032, T044, T045, T046 |
| FR-012 idempotent canonicalisation, no reduction claim | T039 |
| FR-013 declarative schema document | T052, T053, T062 |
| FR-014 constraint + cost per instruction | T047, T048, T052, T054 |
| FR-015 alternatives per kind | T047, T052 |
| FR-016 accumulator precision, reduce dim, order | T047, T052, T054, T062 |
| FR-017 minimum-cost admissible, no default | T049, T051, T055, T057 |
| FR-018 selection recorded with rejection reasons | T050, T056 |
| FR-019 region structure preserved | T027, T035, T038, T058 |
| FR-020 stable serialisation and round-trip | T059 |
| FR-021 emulator executes the emitted stream | T011, T020 |
| FR-022 derived, recorded per-dtype tolerance | T009 |
| FR-023 `register_backend` | T014, T015 |
| FR-024 `DeviceInterface` implementation | T012, T013, T023 |
| FR-025 explicit eager fallback, visible in the report | T017, T019, T021, T022 |
| FR-026 seam proven on day 1 with a hard-coded lowering | T014, T015 |
| FR-027 `fully lowered` before any fraction, per ISA and tier | T067, T068, T069, T072, T078 |
| FR-028 unsupported inventory with `loc` names | T022, T069 |
| FR-029 selection quality vs oracle and reference | T068, T070, T078 |
| FR-030 cross-ISA transfer table with edit locations | T060, T061, T063, T064, T065, T066, T072, T078 |
| FR-031 generated limitations document with cross-references | T024, T071, T075, T080 |
| FR-032 evidence for every quantitative claim | T076, T077, T079 |
| FR-033 negative-control corpus | T008, T031 |
| FR-034 frozen fixtures against a pinned version | T002, T004, T005, T006, T007 |
| FR-035 stated as a Python tool, not a pass | T078 |
| SC-001 real op on the generated device | T015, T016, T023 |
| SC-002 per-tier coverage outcomes | T067, T069, T078 |
| SC-003 both true negatives | T031, T032 |
| SC-004 100% operand resolution on Tiers 1–2 | T029, T033 |
| SC-005 emitted cost vs oracle, gap reported | T050, T056 |
| SC-006 transfer measured on the second ISA | T060, T061, T065, T066 |
| SC-007 generation latency recorded | T069, T078 |
| SC-008 no crash on any input; marker instead | T028, T073 |
| SC-009 every EC ID tested or dispositioned | T075, T080 |
