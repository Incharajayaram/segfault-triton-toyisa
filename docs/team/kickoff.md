# Kickoff — read this before anything else

**Project**: given a declarative description of a tensor ISA, generate a Triton-IR-consuming backend for it
and register it as a device with `torch.compile`.
**Deliverable of record**: a real registered device with a coverage report and a measured cross-ISA transfer
report. Not a demo script.
**Budget**: ~4 working days for 4 people (the single-person estimate is ~6.5 days; the split below is why it
fits).

*Why the per-owner loads sum to ~12.4 person-days and not 6.5.* They should be higher, and the difference is
real: the single-person estimate folded the test corpora, the reports, the CLI, the second ISA and the
integration tests into "the blocks", and this assignment counts them as work. Capacity is 16 person-days
(4 × 4), so the assignment fits with slack. The wall-clock bound is not the sum — it is the critical path,
`fixtures → parser → to_ir + graph → recognizer → idioms → assemble → integrate`, which is ~4 days. Working
that number out instead of asserting it is the whole point of the budget rule below.

## The claim we are entitled to make

> Given a declarative tensor-ISA description, a `torch.compile` backend for it can be generated automatically
> and registered as a PyTorch device, with measured operation coverage on a graduated kernel corpus,
> numerical validation against a reference, and the method's transfer to a second ISA measured.

Three things make that claim testable rather than merely true, and they are the three things that are easy to
skip. Do not skip them.

1. **The seam, not a harness.** `register_backend` + a `DeviceInterface` implementation. ~10 lines of
   registration against a documented API, with an in-tree template at
   `test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/compiler.py`.
2. **Alternatives and costs in the ISA.** Two DMA, two MAC, each with an admissibility predicate and a cost.
   An ISA with one instruction of each kind is a parameter file, and then the word "generate" is unearned.
3. **A second ISA.** Half a day, and it is the only falsifiable test of the word "declarative" in our title.

## The three non-negotiables

- **Never silently miscompile.** Anything we cannot lower becomes an explicit `UNSUPPORTED(op)` marker with
  its originating `loc` name. Anything the parser cannot read becomes `PARSE_UNSUPPORTED(line, col)`. No
  dropping, no guessing, no crash. A crash is a bug; a marker is a feature.
- **Never assert a number you have not produced.** Coverage, cost, transfer rate, latency: generated from the
  reports, never typed into prose. If it is not in `reports/`, it does not go in the write-up.
- **Never edit someone else's module.** Interfaces are frozen in
  `specs/001-triton-to-toy-isa/contracts/`. If you need a change, post in the group and the owner makes it.

## The split

We have a **parsing specialist** and an **IR specialist**, and they are different people. That is the best
starting position available: the front end was the one single-person bottleneck, and it becomes two halves the
moment there is a frozen seam between them. That seam is `RawModule`
(`specs/001-triton-to-toy-isa/contracts/raw-module.md`) — the parser's output is **text-level only**: types stay
strings, attributes stay unparsed, SSA names stay as written.

| Track | Owner | Modules | Load | Critical path? |
|---|---|---|---|---|
| **A — Parser** (text → `RawModule`) | parsing specialist | `ttir/lexer.py`, `ttir/parser.py`, `harness/`, `fixtures/`, `tests/integration/` (from day 2) | ~2.5 d **then floats** | **yes** — front of it |
| **B — IR + emitter** (`RawModule` → IR → program) | IR specialist | `ttir/ssa.py`, `ttir/types.py`, `ttir/to_ir.py`, `ttir/graph.py`, `canon/`, `emit/`, `cli.py`, `report/transfer.py` | ~3.4 d | **yes** — front-middle and tail |
| **C — Recognition & coverage** | recognition owner | `recognize/`, `idioms/`, `report/coverage.py` | ~3.0 d | **yes** — the middle |
| **D — ISA, seam & emulator** | ISA/runtime owner | `isa/`, `torch_backend/`, `emu/` | ~3.5 d | owns the day-1 proof; the integration tail |

**Why the parser and the IR model are two people and not one.** They are different skills — recursive-descent
and error recovery on the one hand, value numbering, region nesting and region-aware traversal on the other —
and they are different layers once the seam is fixed. Assigning both to one person made the critical path's
first link 1.8 days long and gave the whole team a single point of idleness. Split, the front end lands more
than a day earlier and both specialists stay in their own files.

**A is the float from mid-day 2** (the parser is 1.2 days). The float reinforces whichever of C or D is behind,
by writing tests and debugging — never by editing their files.

## Day 1 has zero blocking, on purpose

The three tempting orders (schema-first, parser-first, seam-last) all park someone idle or defer the only
integration risk to the end. Instead:

| Who | Day 1, first 4 hours | Why it is not blocked |
|---|---|---|
| A | **`RawModule` dataclasses committed before lunch**, then fixtures, then the parser happy path | nothing depends on anyone, and A owns the seam everyone else needs |
| B | `ssa.py`/`types.py` field definitions, then `build_ir` against a **hand-built `RawModule`** | the seam is frozen at lunch, so B never waits for the parser |
| C | `AccessDescriptor` committed before lunch, then the expectation table against a **hand-built `Module`**, then the `tts.make_tptr` conformance harness | a dataclass literal is enough to write every test |
| D | `toyisa1.yaml`, the predicate language, `validate_schema`, `select` — then the **day-1 smoke test**: hard-coded 64×64 matmul on the registered device | hand-built descriptors are the real input to the selector; the seam retires the only integration risk today |

Frozen before lunch on day 1 — after that, changes go through the owner, and an agreement in chat is not a
freeze (each item is **committed code with fields only, no logic**):

1. **`RawModule` / `RawOp` / `RawRegion` / `RawBlock` / `RawLoc`** — the parser/IR seam, text stays text (A)
2. `ttir/ssa.py` (`Module`, `Operation`, `Region`, `Block`, `SsaValue`, `TypeExpr`, `Loc`) — A→B (B owns)
3. `recognize/descriptor.py:AccessDescriptor` field set (C)
4. `emit/ir.py` (`Instr`, `Loop`, `Program`, `Operand`, `UnsupportedMarker`) (B)
5. the serialised program text format (header line + `LOOP`/`LOOPEND`) (B)
6. `isa/schema.py` (`IsaSchema`, `Instruction`, predicate term vocabulary) (D)

## Timeline (4 days, 4 people)

| Day | A (parser) | B (IR + emitter) | C (recognition) | D (ISA + runtime) |
|---|---|---|---|---|
| 1 | `RawModule` frozen + fixtures + parser happy path | `ssa`/`types` frozen; `build_ir` green on hand-built `RawModule` | descriptor frozen; expectation table + conformance harness | schema + predicate language + selector; **smoke test green** |
| 2 | syntax negatives; hardening; take over `tests/integration/` | `graph` + region-aware traversal; `canon`; `build_ir` on real text | descriptors on T0–T3 with exact fields; the walk | emulator + precision; wire the real pipeline |
| 3 | per-tier integration tests + fuzz pass (the float) | `emit/assemble` + region ordering; serialise/round-trip | idioms + annotation; **both true negatives** | per-tier end-to-end; fallback records; selection quality |
| 4 | evidence appendix; edge-case traceability; float | CLI + transfer report | coverage + limitations document | ISA-2 + forced-edit list; reports |
| 5 | buffer — and the published cut order if needed | | | |

**Cut order under time pressure** (published, so nobody has to negotiate it in the moment): fuzz kernel →
canonicalisation entirely → ISA-2 → Tier-2 epilogue → Tier-3 fixture.
**Never cut**: the Tier-3 negative control, the coverage table, the true-negative test, the seam.

## Communication protocol

- **Group**: status, blockers, decisions. One thread per blocker; no side-channel design decisions.
- **A blocker is announced within 15 minutes of being hit**, with the failing command and its output. Nobody
  sits on a blocker overnight on a 4-day project.
- **Decisions are recorded in the repo**, in `specs/001-triton-to-toy-isa/research.md` (append a row to the
  decisions table) or as an issue. WhatsApp is for coordination; the repo is the source of truth. If a
  decision is not in the repo, it did not happen.
- **Branches**: `track1-frontend`, `track2-recognition`, `track3-isa-emit`, `track4-runtime`. Merge to `main`
  only when that track's contract tests pass.
- **Daily 10-minute stand-up**: what passed, what is blocked, what changed at an interface.

## Reading order (and what to read for your track)

Everyone, once: this file → `specs/001-triton-to-toy-isa/spec.md` (the requirements) →
`methodology-v2.pdf` (the method, 23 pages) → `research.md` §"Foundational truths" (10 things that are true
about this problem).

Then your own brief (`docs/team/track-N-*.md`) and your own contract file. Read the other tracks' contracts
only when you need to call into them — the contracts are the coordination device, and reading all eight is
how you avoid a two-hour conversation.

**T-minus reading, if there is 20 minutes before kickoff**: the audit (`AUDIT.md`). It is what the project
learned the hard way: v1 asserted a "5-day budget" four times without adding it up, and cited a paper section
that does not exist. Every rule above exists to keep us from repeating that.

## Ground rules that are not obvious

- **The fixtures are the interface, not the live compiler.** `ttir` syntax changes between Triton releases.
  We parse frozen fixtures. Regenerating them is a deliberate, reviewed act (`--force`) that records the new
  version hash.
- **Do not trust an IR shape you have not dumped.** Twice this project was designed against an imagined IR.
  The loop structure, the `tf32` literal and the 13 `tt.splat`s / 6 `tt.broadcast`s all came from dumping the
  real thing. Dump first, design second.
- **A flat topological sort over the def-use graph is wrong.** Both `tt.load`s must stay inside `scf.for`,
  `scf.yield` terminates the block, and the accumulator is loop-carried. Traversal is region-aware.
- **`tf32` is a literal in the corpus.** A float32 NumPy emulator cannot match torch numerics. Implement the
  declared precision and the declared accumulation order, and record the tolerance derivation.
- **The percentage is not the metric.** "Fully lowered: yes/no" is the metric, reported per tier and per ISA.
  A kernel with one unlowered op early in a chain can show 90% and be useless.
