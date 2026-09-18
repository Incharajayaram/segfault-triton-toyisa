# Ownership, interfaces and blocking (revised: parser and IR are two people)

The team has a parsing specialist and an IR specialist. That is the best possible starting position, because
the critical path's first link — the front end — was the one single-person bottleneck. It becomes two halves
the moment there is a frozen seam between them: **`RawModule`**
(`specs/001-triton-to-tritonflow/contracts/raw-module.md`, written for exactly this reason).

## The four owners

| Owner | Track | Owns | Contracts | Load |
|---|---|---|---|---|
| **A** — parsing specialist | Parser (text → `RawModule`) | `ttir/lexer.py`, `ttir/parser.py`, `harness/extract_fixtures.py`, `fixtures/` | `contracts/raw-module.md` (producer) | ~2.5 d **then floats** |
| **B** — IR specialist | IR model + emitter (`RawModule` → IR → program) | `ttir/ssa.py`, `ttir/types.py`, `ttir/to_ir.py`, `ttir/graph.py`, `canon/`, `emit/`, `cli.py`, `report/transfer.py` | `contracts/raw-module.md` (consumer), `contracts/assembler.md` | ~3.4 d |
| **C** — recognition | Recognition + coverage | `recognize/`, `idioms/`, `report/coverage.py` | `contracts/access-descriptor.md`, `contracts/coverage-report.md` | ~3.0 d |
| **D** — ISA + runtime | ISA, seam, emulator, integration | `isa/`, `torch_backend/`, `emu/`, `tests/integration/` | `contracts/isa-schema.md`, `selector.md`, `torch-seam.md`, `emulator.md` | ~4.0 d |

A is the **float** after mid-day 2: the parser is 1.2 days, so A reinforces whichever of C or D is behind —
never B's `to_ir.py` (B's skill, B's file) and never `isa/`. Two people in one file is the failure mode this
split exists to avoid.

## Module → owner

| Path | Owner | Contract (definition of done) |
|---|---|---|
| `ttir/lexer.py`, `ttir/parser.py` | **A** | `contracts/raw-module.md` |
| `harness/extract_fixtures.py`, `fixtures/*` | **A** | `raw-module.md` (fixtures section) |
| `tests/unit/test_parser_syntax_negative.py` | **A** | EC-001…EC-004, EC-011…EC-025, EC-027 |
| `ttir/ssa.py`, `ttir/types.py`, `ttir/to_ir.py` | **B** | `data-model.md` §2, `raw-module.md` |
| `ttir/graph.py`, `canon/canonicalize.py` | **B** | `contracts/ttir-parser.md` (graph), FR-012 |
| `tests/unit/test_parser_semantic_negative.py` | **B** | EC-005, EC-026, type/loc errors |
| `emit/ir.py`, `emit/assemble.py`, `emit/disasm.py` | **B** | `contracts/assembler.md` |
| `cli.py` (compile/report/transfer) | **B** | `plan.md` §1 |
| `report/transfer.py` | **B** | `contracts/coverage-report.md` §6 |
| `recognize/*` | **C** | `contracts/access-descriptor.md` |
| `idioms/*` | **C** | `contracts/access-descriptor.md` |
| `report/coverage.py` | **C** | `contracts/coverage-report.md` |
| `isa/schema.py`, `isa/select.py`, `isa/rules/`, `isa/schemas/` | **D** | `contracts/isa-schema.md`, `selector.md` |
| `torch_backend/*` | **D** | `contracts/torch-seam.md` |
| `emu/*` | **D** | `contracts/emulator.md` |
| `tests/integration/*` | **D** | `contracts/torch-seam.md` |

**Writing tests for someone else's module is encouraged. Editing someone else's module is not.** A's float
role is *adding tests and debugging*, not adding code to C's or D's files.

## The interface freeze list (before lunch, day 1)

Now six items, and #1 is the new one. Each is **committed code on day 1 with fields only, no logic** — an
agreement in chat is not a freeze.

| # | Frozen | Owner | Consumed by |
|---|---|---|---|
| **1** | **`RawModule`, `RawOp`, `RawRegion`, `RawBlock`, `RawLoc` — the parser/IR seam. Text stays text.** | **A** | **B** |
| 2 | `ttir/ssa.py`: `Module, Function, Operation, Region, Block, SsaValue, TypeExpr, Loc, Attr`; `Operation.results` is a **list** | B | C, D |
| 3 | `recognize/descriptor.py`: `AccessDescriptor` fields — `base, sizes, strides, offsets, shape, order, dtype, loop_carried, increment, provenance`; `DescriptorResult = Ok \| Unstructured \| BudgetExhausted` | C | B, D |
| 4 | `emit/ir.py`: `Instr`, `Loop` (`iter_args`), `Program` (`schema_version`, `total_cost`, `unsupported`), `Operand`, `UnsupportedMarker` | B | D |
| 5 | Program text format: header `; <isa> schema_version=N kernel=<k> cost=<c>`, then `LOOP …/LOOPEND`, then epilogue | B | D |
| 6 | `isa/schema.py`: `IsaSchema`, `Instruction` fields, predicate term vocabulary (`shape[i]`, `stride[i]`, `offset[i]`, `aligned`, `in_bounds`, `all_of`, `any_of`) | D | C |

## Dependency matrix (who unblocks whom)

| Consumer | Needs | From | Available | Can it start earlier on a hand-built input? |
|---|---|---|---|---|
| A | a Triton env | — | hour 0 | n/a |
| B (`build_ir`) | `RawModule` | A | day 1 lunch (fields), day 2 (real output) | **yes** — hand-built `RawModule` literals |
| B (graph, emit) | `AccessDescriptor`, `AnnotationSet` | C | day 1 lunch (fields) | **yes** — literal descriptors |
| C (descriptor) | `Module` with regions/SSA | B | day 1 lunch (fields) | **yes** — hand-built `Module` |
| C (idioms) | real descriptors | C itself | day 2 | n/a |
| D (selector) | nothing | — | hour 0 | n/a |
| D (emulator) | `Program` | B | day 1 lunch (fields) | **yes** — literal `Program` |
| D (integration) | the whole pipeline | A, B, C | day 3 | **no — this is the one true convergence point** |
| B (transfer report) | D's forced-edit list | D | day 4 | partial — P/D records edits as they happen |

**Critical path**: `A (fixtures) → A (parser happy path) → B (to_ir + graph) → C (recognizer → idioms) →
B (assemble) → D (integrate)`.

With the parser/IR seam, that chain is ~1.0 + 0.8 + 1.75 + 1.0 + 0.4 ≈ **5 days of work spread across three
owners and overlapped**, which is why the wall-clock is ~4 days. Without the seam it was one person for 1.8
days and everything downstream started later.

## Blocking rules

1. **Need an interface that does not exist?** Write the test with the exact field names from `raw-module.md` /
   `data-model.md` / the contract, hand it to the owner, and keep going against your hand-built input.
2. **Blocked > 2 hours** → post the failing test and the interface you need. The owner lands the interface or
   hands you a 10-line stub within 2 hours.
3. **Never fork a behaviour.** Two behaviours for one interface costs a day at integration time and is the
   single most likely way this project misses its window.
4. **Report `fully lowered` as a boolean.** Never "about 90% done".

## Escalation (four people, four days — this is short)

| Question | Ask |
|---|---|
| "Is this IR shape real?" | Dump it. Read the fixture. Never guess. Twice this project designed against an imagined IR. |
| "Lower it or mark it `UNSUPPORTED`?" | `UNSUPPORTED`, then a limitations entry. The marker is always the safe answer. |
| "Is this number good enough?" | Is it produced by a script from a fixture, with a command that reproduces it? If not, it is not a number yet. |
| "Do we have time?" | Apply the published cut order in `plan.md`. Do not renegotiate it live. |
| "Which of us fixes this?" | The `layer` field on the diagnostic names the owner: `syntax` → A, `ir` → B, `descriptor` → C, `isa`/`device` → D. |
