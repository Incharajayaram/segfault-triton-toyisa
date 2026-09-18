# WhatsApp pack — what to send, to whom, in what format

Everything below is ready to copy-paste. Sizes are real: the whole pack is **under 700 KB**, so all of it fits
in three or four messages with no compression worry.

Send in this order, one message at a time, so the pin lands at the bottom and stays at the top of the chat:

| # | Where | What | Attachments |
|---|---|---|---|
| 1 | Group | **Message 1** — the kickoff (below) | `methodology-v2.pdf` only |
| 2 | Group | **Message 2** — repo bootstrap + ground rules | none (paste the commands as text) |
| 3 | Group | **Message 3** — the reading list | `kickoff.md`, `spec.md`, `plan.md`, `tasks.md`, `research.md` |
| 4–7 | DM to each person | **Messages 4–7** — one template per person | their track brief + their contract(s) |
| 8 | Group | **Message 8** — the day-1 handoff checklist | none |

**Pin message 1.** Everything else is scrollable.

---

## Message 1 — PINNED (send to the group, attach `methodology-v2.pdf`)

```text
PROJECT KICKOFF — read this before you open anything else.

What we're building
Given a declarative description of a tensor ISA, generate a Triton-IR-consuming
backend for it and register it as a device with torch.compile. So: user writes
normal PyTorch -> torch.compile -> Triton IR -> our pipeline -> toy ISA program ->
runs on our emulated device. Nobody has closed this seam: industry hand-builds
Triton backends (Meta/MTIA, Microsoft/triton-shared, Cambricon, Intel), and the
academic automation (ACT, TensorLift) targets XLA, which PyTorch does not use.

The prize, quantified: 2,590 ATen operators vs 209 Inductor lowering rules.
Four organisations hand-built this layer separately.

Deliverable of record: a REAL registered torch.compile device with a coverage
report and a measured cross-ISA transfer report. Not a demo script.

Budget: ~4 working days for the 4 of us. Single-person estimate is ~6.5 days -
the split below is why it fits. Capacity is 16 person-days; the assigned loads sum
to ~12.4, so there is slack, and the wall-clock bound is the critical path.

THE THREE NON-NEGOTIABLES
1. Never silently miscompile. Anything we can't lower becomes UNSUPPORTED(op) with
   its loc name; anything we can't parse becomes PARSE_UNSUPPORTED(line,col).
   No dropping, no guessing, no crash. A crash is a bug; a marker is a feature.
2. Never assert a number we haven't produced. Coverage/cost/transfer/latency come
   from reports/. If it's not in reports/, it doesn't go in the write-up.
3. Never edit someone else's module. Interfaces are frozen in the contracts. Need
   a change? Post it, the owner makes it.

THE SPLIT - and why the parser and the IR model are two different people
We have a parsing specialist and an IR specialist. That is the best possible
starting position: the front end was the one single-person bottleneck, and it
becomes two halves the moment there is a frozen seam between them.

The seam is RawModule (contracts/raw-module.md). The parser's output is
TEXT-LEVEL ONLY: types stay strings, attributes stay unparsed, SSA names stay as
written, terminators are just ops. The IR specialist turns that into real IR:
value numbering, multi-result binding, type interpretation, region nesting,
def-use graph, region-aware traversal.

Track A - Parser (text -> RawModule)              A   2.5 d  then FLOATS
  ttir/lexer.py, ttir/parser.py, harness/, fixtures/, tests/integration/ (from D2)
Track B - IR model + emitter                      B   3.4 d  CRITICAL PATH
  ttir/ssa.py, ttir/types.py, ttir/to_ir.py, ttir/graph.py, canon/, emit/, cli.py
Track C - Recognition + coverage                  C   3.0 d  CRITICAL PATH
  recognize/, idioms/, report/coverage.py
Track D - ISA, seam and emulator                  D   3.5 d  owns the day-1 proof
  isa/, torch_backend/, emu/

Assign the parsing specialist to A and the IR specialist to B. Do not put them on
the same file: the seam exists so that they don't have to be.

WHY DAY 1 HAS ZERO BLOCKING
A: commit the RawModule dataclasses BEFORE LUNCH, then fixtures, then the parser
   happy path. A owns the seam everyone else needs.
B: ssa.py field definitions, then build_ir against a HAND-BUILT RawModule. The seam
   is frozen at lunch, so B never waits for the parser.
C: descriptor fields committed, then the expectation table against a HAND-BUILT
   Module, then the tts.make_tptr conformance harness.
D: tritonflow1.yaml + predicate language + selector (a descriptor is a plain dataclass,
   so the whole selector is testable today), then the day-1 SMOKE TEST: hard-coded
   64x64 matmul running on the registered device.

TIMELINE: D1 interfaces + foundations + smoke test | D2 parser green + descriptors +
emulator | D3 integration + true negatives | D4 reports + ISA-2 + write-up | D5 buffer.

CUT ORDER (already decided, don't renegotiate it live):
fuzz kernel -> canonicalisation -> ISA-2 -> Tier-2 epilogue -> Tier-3 fixture.
NEVER CUT: Tier-3 negative control, coverage table, true-negative test, the seam.

RULES FOR THE CHAT
- Blockers posted within 15 minutes of being hit, with the failing command + output.
- A is the FLOAT from mid-day 2: it reinforces whichever of C or D is behind, by
  writing tests and debugging - never by editing their files.
- Decisions get recorded in the repo (research.md or an issue). WhatsApp is for
  coordination; the repo is the source of truth. If it isn't in the repo, it
  didn't happen.
- Branches: track-a-parser, track-b-ir, track-c-recognition, track-d-isa-runtime.
  Merge to main only when that track's contract tests pass.
- Stand-up: 10 minutes, same time daily. What passed / what's blocked / what
  changed at an interface.

Reading order: docs/team/kickoff.md -> spec.md -> methodology-v2.pdf (23pp) ->
research.md section "Foundational truths" -> your own track brief + your contracts.
```

---

## Message 2 — repo bootstrap + ground rules (group, no attachment)

```text
SETUP (do this today, it's ~15 minutes)

git clone <your-repo-url> segfault && cd segfault
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'      # pyyaml, numpy, pytest, triton==3.7.1, torch==2.12.1

Reference repos - do NOT clone all of these unless you need them (426 MB total):
  repos/clone.sh does shallow clones (--depth 1) of triton-shared, triton-linalg,
  intel-xpu-backend-for-triton and pytorch. pytorch alone is 342 MB. Only C needs
  triton-shared (for the make_tptr dialect definition) and only D needs pytorch
  (two files: DeviceInterface and torch_openreg/compiler.py).

The three files people always end up asking about:
  torch/_dynamo/device_interface.py            <- the 30 slots we implement
  torch/_dynamo/backends/registry.py           <- register_backend
  test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/
      compiler.py                              <- the template we copy the shape of

GROUND RULES THAT AREN'T OBVIOUS
- Fixtures are the interface, not the live compiler. ttir syntax changes between
  Triton releases. Regenerating fixtures is a reviewed act (--force) that records
  the version hash. Don't regenerate them to make a test pass.
- Don't trust an IR shape you haven't dumped. Twice this project was designed
  against an imagined IR. Dump first, design second.
- In track A's layer, TEXT STAYS TEXT. The moment the parser decides what a type
  means, we have two dialects and integration day is spent reconciling them.
- A flat topological sort over the def-use graph is WRONG. Both tt.loads must stay
  inside scf.for; scf.yield terminates the block; the accumulator is loop-carried.
- tf32 is a literal in our corpus. A float32 NumPy emulator cannot match torch
  numerics. Implement the declared precision AND the declared accumulation order.
- Coverage is a boolean first ("fully lowered: yes/no"), reported per tier and per
  ISA. A node percentage can read 90% on a kernel that is useless.
- Every track must be testable without its upstream being finished. That is what
  makes this parallel. If you're blocked, you're missing a hand-built input, not
  a colleague.
```

---

## Message 3 — the reading list (group, attach the seven files)

Attach: `docs/team/kickoff.md`, `docs/team/testing-ci.md`,
`specs/001-triton-to-tritonflow/spec.md`, `specs/001-triton-to-tritonflow/plan.md`,
`specs/001-triton-to-tritonflow/tasks.md`, `specs/001-triton-to-tritonflow/research.md`,
and the `Makefile` pasted as text.

```text
THE SIX FILES THAT MATTER (in reading order)

kickoff.md   - mission, split, protocol, timeline. 10 min.
spec.md      - 5 user stories, 35 requirements, 9 measurable success criteria.
               This is what "done" means. 20 min.
plan.md      - every module and every function, the build order, the schedule,
               the 3 justified deviations. 20 min.
tasks.md     - 82 tasks with IDs, exact file paths, and a requirement traceability
               matrix at the bottom (Phase 0, 7 of them, is already done). 15 min.
research.md  - the 10 questions we grilled ourselves on, the 10 foundational
               truths, and 12 rejected alternatives with reasons. Read the
               "Foundational truths" table first - it will stop you rediscovering
               them the hard way. 20 min.

Also worth 10 minutes, and it will save arguments later: docs/team/parallelism.md
(what parallelises, what can't, and the three disciplines that make it work).

MANDATORY, before you write one line of code: docs/team/testing-ci.md. It is how
we all build and test against the same numbers. Read S1, S4, S6, S8 and S9; the
rest is reference. Then run:

    make bootstrap
    make ci

Both are green today, before any pipeline code exists. That is the point.

Don't read the contracts now. Read YOUR OWN when you start your track - it is your
definition of done. There are nine; nobody needs all nine.
```

---

## Messages 4–7 — the four DMs (one per person)

**Do not paste a long brief into a chat.** The brief *is* the document; the DM is the pointer plus the one
thing you want them to do in the next hour. Attach their brief and their contract(s), and send this template
with the four brackets filled in:

```text
You're Track [A/B/C/D] — [track name].

Own: [their paths]
Your contracts (attached): [their contract files]
Your load: ~[N] days. Critical path: [yes/front of it/no, and why].

YOUR FIRST HOUR
[the one day-1 action that unblocks everyone else or proves something today]

The seam that keeps you unblocked: [the frozen dataclass they build against, and
who owns it].

Two things that will bite you if nobody says them out loud:
1. [the gotcha from their brief]
2. [the second gotcha]

Full details in your brief (attached). Do not read the other three tracks'
contracts unless you are calling into them.
```

Filled in, the four **first-hour** lines are:

| DM to | First hour | Their seam / substitute |
|---|---|---|
| **A** — parser | Commit the `RawModule` dataclass fields **before lunch**, then fixtures, then the four-fixture happy path | You *own* the seam. `tests/unit/test_parser_syntax_negative.py` is your half of the negative corpus |
| **B** — IR + emitter | Commit `ssa.py`/`types.py` fields, then get `build_ir` green against a **hand-built `RawModule`** — the parser does not exist yet and does not need to | `RawModule` (A freezes it at lunch); `AccessDescriptor` (C) for the emitter |
| **C** — recognition | Commit `AccessDescriptor` fields, then write the expectation table against a **hand-built `Module`**, then read triton-shared's `TTS_Op<"make_tptr">` (nine arguments, not three) | `Module`/`SsaValue` (B freezes at lunch) — build them as literals |
| **D** — ISA + runtime | `tritonflow1.yaml` + predicate language + selector, tested on hand-built descriptors — then get the **smoke test green**: a real torch op on the registered device | Nothing upstream. Hand-built descriptors *are* the real input to the selector |

---

## Message 8 — day-1 handoff checklist (group, no attachment)

```text
DAY 1 CHECKLIST - post each one in the group as it lands, then we're unblocked

[ ] A  RawModule dataclasses FROZEN + committed      -> unblocks B (and the whole seam)
[ ] B  ssa.py / types.py frozen + committed          -> unblocks C and D
[ ] A  fixtures generated (4 .ttir + VERSIONS.txt)   -> unblocks everyone's real input
[ ] C  AccessDescriptor frozen + committed           -> unblocks D's selector
[ ] D  tritonflow1.yaml + validate_schema passing        -> proves the decision machinery works
[ ] D  SEAM SMOKE TEST GREEN: real torch op on our device

End of day 1 we should be able to say, with a command that reproduces it:
"a torch.compile'd matmul runs on a device we generated, and our parser fails
cleanly on malformed input."

Six boxes, four people, and none of them waits for another box to be ticked first -
that is the whole point of the seam. If any box is unticked, say so at stand-up with
the failing command. Do not spend day 2 silently working around a missing interface.
```

---

## The reference papers (rename them before sending)

Three of the four source PDFs are in the project folder under arXiv/date names nobody will recognise. Rename
before sending, or they will never be opened:

| Current name | Send as | Size | What it is / who reads it |
|---|---|---|---|
| `2510.09932v2.pdf` | `ACT-OOPSLA26.pdf` | 3.2 MB | ACT: auto-generating ML compiler backends from ISA descriptions, for **XLA**. Source of the α/β split (§3), equality saturation (§5), integer constraint programming for addressing (§6), the cost model (§7), soundness (§3.3). **D reads §5–§7; everyone reads §3.3.** |
| `2604.13523v3.pdf` | `TensorLift-ICCAD26.pdf` | 1.1 MB | RTL → TAIDL ISA semantics via MLIR. Source of the 4-phase / 8-pass pipeline (Table 2: `detect-mac`, `detect-clamp`, `reconstruct-loops`, …) and "passes annotate rather than rewrite" (§3.2). **B and C read Table 2 and §3.2.** |
| `2608.00325v2.pdf` | `Triton-MTIA-Meta.pdf` | 0.5 MB | Meta's **hand-built** production Triton backend for MTIA-2i. The architecture template (Fig. 4), and the evidence that this layer is expensive: 16 kernels ported, 5 needing rewrites (§8.1), and the modulo pointer arithmetic case (§8.1) that is our Tier-3 negative control. **Everyone skims §3.3; D reads Table 1 and §7.** |
| `methodology.pdf` | `methodology-v1-archived.pdf` (optional) | 0.2 MB | The original plan. Send only if someone asks why v2 exists — `AUDIT.md` answers it better. |

## File reference table

*`specs/.../` below is shorthand for `specs/001-triton-to-tritonflow/`.*

| File | Send to | Format | Size | Purpose |
|---|---|---|---|---|
| `methodology-v2.pdf` | group | PDF | 570 KB | the method of record, 24 pp |
| `docs/team/testing-ci.md` | **group, mandatory** | .md | 15 KB | the unified test, benchmark and CI protocol — must be read before any code |
| `Makefile`, `pyproject.toml`, `requirements-dev.txt` | group | paste as text | 4 KB | `make bootstrap` / `make ci`; no Triton, no torch, no GPU |
| `fixtures/GOLDEN.json` | group | paste as text | 8 KB | the canon: 4 fixtures, 182 operations, the only source of expectations |
| `tools/snapshot_fixtures.py`, `fixtures_lib.py` | group | paste as text | 14 KB | `make golden` / `make golden-check` — the CI gate on fixture drift |
| `.github/workflows/ci.yml` | group | paste as text | 4 KB | every CI job is a `make` target |
| `bench/cases.yaml`, `bench/adapter.py` | **D** | paste as text | 5 KB | the 32-row bench matrix and the one place bench touches the pipeline |
| `docs/team/kickoff.md` | group | .md (or paste) | 13 KB | pinned brief: mission, split, protocol |
| `docs/team/ownership.md` | group | .md | 10 KB | module→owner matrix, freeze list, blocking rules |
| `docs/team/parallelism.md` | group | .md | 12 KB | what parallelises, what can't, the three disciplines |
| `specs/.../spec.md` | group | .md | 20 KB | requirements, user stories, success criteria |
| `specs/.../plan.md` | group | .md | 25 KB | components, every function, build order |
| `specs/.../tasks.md` | group | .md | 22 KB | 82 tasks + 7 pre-build + traceability matrix |
| `specs/.../research.md` | group | .md | 24 KB | the 10 grilled questions, 10 truths, rejected alternatives |
| `specs/.../contracts/raw-module.md` | **A, B** | .md | 8 KB | the parser/IR seam — both owners need it |
| `specs/.../contracts/ttir-parser.md` | A | .md | 4 KB | graph + fixtures portion of A's contract |
| `specs/.../contracts/access-descriptor.md` | C | .md | 4 KB | C's definition of done |
| `specs/.../contracts/coverage-report.md` | C | .md | 6 KB | coverage, selection and transfer report shapes |
| `specs/.../contracts/assembler.md` | B | .md | 4 KB | emission, region ordering, round-trip |
| `specs/.../contracts/isa-schema.md`, `selector.md` | D | .md | 9 KB | schema validity, fail-closed predicates, selection |
| `specs/.../contracts/torch-seam.md`, `emulator.md` | D | .md | 15 KB | the seam and the precision policy |
| `specs/.../data-model.md` | A, B, C, D | .md | 17 KB | ISA YAML in full, descriptor, program format, reports |
| `specs/.../edge-cases.md` | A, B, C | .md | 16 KB | the 100 cases with required behaviour |
| `specs/.../quickstart.md` | group | .md | 8 KB | the five commands, with expected output |
| `.specify/memory/constitution.md` | group | .md | 8 KB | the 5 binding principles |
| `AUDIT.md` | optional | .md | 24 KB | the 17 falsified claims — why the ground rules exist |
| `repos/clone.sh` | C, D | paste as text | 4 KB | shallow-clone script for the reference repos |

**Do not send `repos/`** — it is 426 MB (`pytorch` alone is 342 MB). Paste `repos/clone.sh` and the URLs
instead, or just point at the repo.

## Format rules for WhatsApp

- **PDF for anything longer than a screen.** WhatsApp renders `.md` as plain text, so tables turn into
  pipes-and-dashes soup and `**bold**` shows literally. Only `methodology-v2.pdf` exists as a PDF today;
  everything else is markdown, which is fine in a markdown viewer or on a laptop.
- **`.yaml`/`.tex`/`.py`: paste as text or send as `.txt`.** Never as the primary reference — nobody reads YAML
  in a chat bubble.
- **Send each contract to exactly one person.** A contract is that person's definition of done; broadcasting
  all nine invites everyone to have an opinion about everyone's interface, which is the coupling the split
  exists to avoid. The exception is `raw-module.md`, which A and B co-own as producer and consumer.
- **Never paste a brief into the chat.** Attach it. Pasted text scrolls away and gets edited in replies.
- **One decision per message, one thread per blocker.** Four people, four days — the chat cannot be the place
  where the interface lives.
- **The repo is the source of truth.** If a WhatsApp decision is not copied into `research.md` or an issue
  within the hour, it does not exist and it will be re-litigated on day 3.
- Everything here is ≤ 570 KB, so no WhatsApp compression concerns (documents up to 100 MB are fine).
