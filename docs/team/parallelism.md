# Does this split actually parallelise? — an honest audit

**Short answer: yes for days 1–3, with three disciplines that the plan was one edit short of stating. Day 4
converges on one person, and there is one hard serialization point that cannot be removed.**

The question to ask is not "are the modules separate?" — that is easy. It is:

> **Can you run your tests without your upstream track being finished?**

If yes, you are working in parallel. If no, you are waiting politely, and no amount of module ownership fixes
it. Below is the audit, track by track.

## 1. The substitution table — what makes each track independent

Every track's input is a *dataclass*, not a working system. That is the whole reason this parallelises: a
hand-built input costs 20 minutes and buys a day of unblocked work.

| Track | Upstream input it needs | Hand-built substitute | Available from |
|---|---|---|---|
| **A** Parser | a `ttir` file | the fixtures, or a string literal in a test | itself, ~1 h in |
| **B** IR + emitter | a `RawModule` | **write one as a literal** — `RawOp`/`RawModule` are frozen dataclasses | hour 0 |
| **B** (emitter) | an `AccessDescriptor` + an `AnnotationSet` | **construct both literals** | hour 0 |
| **C** Recognition | a `Module` with regions and SSA values | **build one in the test** — `Module`/`Operation`/`SsaValue` are frozen dataclasses | hour 0 |
| **D** ISA + selector | nothing at all | hand-built descriptors are the *real* input | hour 0 |
| **D** (emulator) | a `Program` | **construct one literal** — the `Instr`/`Loop` dataclasses are frozen | hour 0 |

So all four tracks can be *developed and tested* on day 1 with nothing from anyone else. That is the property
the user is asking about, and it does hold — and it holds for the front end too, because the parser and the IR
model are separated by the `RawModule` seam (`contracts/raw-module.md`), which means the IR specialist is not
waiting for the parser specialist.

**What does not hold**: validating against the *real* upstream. C's recogniser validated on a real parsed
module waits for A's parser plus B's `build_ir`. That is not a flaw to engineer away — a fake cannot prove the
parser works. It is a *handover*, and the rule below handles it.

## 2. The three disciplines the plan needed

### Discipline 1 — interfaces are committed code on day 1, not agreements

"Freeze `ssa.py` before lunch" is not enough. If `ssa.py` does not exist, C writes their own `SsaValue`, D
writes a second `AccessDescriptor`, and integration day is spent reconciling three dialects.

**Rule**: before lunch on day 1, each of these files exists in the repo with **field definitions only, no
logic**:

| File | Owner | Frozen content |
|---|---|---|
| `ttir/parser.py` (types only) | A | `RawModule`, `RawOp`, `RawRegion`, `RawBlock`, `RawLoc` — **the parser/IR seam** |
| `ttir/ssa.py` | B | `Module, Function, Operation, Region, Block, SsaValue, TypeExpr, Loc, Attr` |
| `recognize/descriptor.py` | C | `AccessDescriptor`, `DescriptorResult` (the three-variant union) |
| `emit/ir.py` | B | `Instr, Loop, Program, Operand, UnsupportedMarker` |
| `emu/precision.py` | D | `PrecisionPolicy` fields |

Twenty lines each. This is the single highest-leverage hour of the project, and it is now **six items, not
five**, because the parser/IR seam is an interface like any other. Without it the two front-end specialists
would be one bottleneck instead of two parallel halves.

### Discipline 2 — every track has a fake upstream and keeps it honest

The fake is where the parallelism comes from, and it is also the way this goes wrong: you end up testing your
fake. The fix is not to avoid fakes — it is to make the real input an obligation:

- Contract tests are **parametrised over two inputs**: `["handbuilt", "real"]`.
- The `"real"` parameter is `@pytest.mark.xfail(strict=False)` until the upstream lands, then it becomes a hard
  failure if it does not pass.- **By end of day 3, no contract test may run only against a hand-built input.** A — the float from mid-day 2 —
goes through the suite and each `"real"` row must be green. This is the float's highest-value job: it is the one
person whose task is exactly "make the fakes stop being necessary".

That converts the hand-built input from a permanent second dialect into a temporary scaffold with an expiry
date.

### Discipline 3 — a blocked person escalates in 2 hours, not at stand-up

Four people, four days. A half-day of silent waiting is 12% of the project. The escalation rule is not
politeness, it is the mechanism that keeps the parallelism real:

- blocked > 2 hours → post the failing test in the group, with the exact interface you need;
- the upstream owner either lands the interface within 2 hours or **hands you a 10-line stub** so you can
  proceed (stubs are deleted when the real implementation lands, and the stub's contract test is the one that
  catches it).

## 3. Who waits on whom — with the mitigation

| Waits | On | When | Mitigation that removes the wait |
|---|---|---|---|
| B, C, D (real-input validation only) | A's fixtures | day 1, hour 1 | A owns them and does them first; B/C/D use literals in their own tests |
| B (`build_ir` on real text) | A's parser output | day 2 | **the seam is the mitigation**: `RawModule` is frozen day 1, so B writes and tests `build_ir` against a hand-built `RawModule` from hour 0 |
| C (real validation) | A's parser + B's `build_ir` | day 2 | develop against hand-built `Module`s from hour 0; C needs *a* module, not a fuzz-hardened parser |
| C (graph traversal) | B's `graph.py` | day 2 | the graph API is frozen day 1; C's walk tests build their own `DefUseGraph` until it lands |
| B (`assemble`) | C's descriptors + A/B's regions | day 3 | build the emitter against literal `AccessDescriptor` + a literal `AnnotationSet`; assembly needs the *interface*, not real values |
| D (`lower_and_run`) | A + B + C | day 3–4 | **this one cannot be faked** — it is the end-to-end test. It is one 3-hour task, not a track, and D has the ISA, selector, device, interface and emulator to finish first |
| B (transfer report) | D's forced-edit list | day 4 | D records edits as they happen, so the list exists before the report is written |
| D (ISA-2 run) | the whole pipeline | day 4 | intentionally last, and deliberately second in the cut order — if the pipeline is late, ISA-2 is the first thing to drop |

## 4. The three genuine serialization points

These cannot be engineered away, and it is important to say so rather than pretend the plan is embarrassingly
parallel:1. **Fixtures (~1 hour, day 1).** Everyone's *real* input. A owns them (they need them most), it needs a Triton
environment, and it is the first thing that happens. Cost of the gate: an hour.
2. **The parser's happy path (day 2).** A's parser is the largest single risk and it gates the *real* validation
of B's `build_ir` and C's recogniser. Mitigation: **happy path first, hardening second**. Land "parses the four
fixtures" on day 2 morning; push fuzz-hardening, exotic syntax and the 200-deep nesting case to day 3. B and C
need a module from a clean fixture, not a parser that survives adversarial input. And note the seam's payoff
here: B's `build_ir` is *already written and tested* against a hand-built `RawModule` when the parser lands, so
this gate costs hours instead of a day.
3. **Integration (day 3–4).** One person wires the real pipeline behind the seam. By definition this waits for A,
B and C. It is 3 hours of work; D owns it because D owns the device and the seam, and A takes over
`tests/integration/` from day 2 so the per-tier tests are not a single person's bottleneck either.

Together these leave **roughly one person-day of unavoidable serialization** across four days. Everything else
parallelises.

## 5. Where the spare capacity actually is

Loads: **A ≈ 2.5 d, B ≈ 3.4 d, C ≈ 3.0 d, D ≈ 3.5 d** over a 4-day window. The spare capacity is **A**, and it
is deliberate: the parser is 1.2 days, so A is free from mid-day 2 and becomes the float.

The critical path is `A (fixtures) → A (parser happy path) → B (to_ir + graph) → C (recognition → idioms) →
B (assemble) → D (integrate)`.

**Staffing consequence**: if the critical path is at risk on day 2, A reinforces it — by writing tests and
debugging in C's or D's area, never by editing their files. And never onto ISA-2, which is second in the cut
order and therefore already the designated sacrifice.

## 6. Parallelism health check (end of day 2)

If all five of these are true, the split is working as designed. If any is false, fix the process, not the
plan:

- [ ] All four frozen interface files exist and were committed on day 1
- [ ] All four tracks ran their own tests on day 2 **without waiting** for another track
- [ ] The parser parses all four fixtures (happy path), and nobody is fuzzing it yet
- [ ] No contract test runs only against a hand-built input after day 3
- [ ] Nobody has been blocked more than 2 hours without a stub or an escalation

## 7. How to tell it is *not* parallelising

Three symptoms, each with its cause:

| Symptom | Actual cause | Fix |
|---|---|---|
| "I'm waiting for X to finish" | the hand-built substitute was never written | write the fake first; it is 20 minutes |
| Integration day discovers three incompatible data models | interfaces were agreed in chat but not committed on day 1 | Discipline 1, retroactively: freeze the dataclasses today |
| Everyone is editing `recognize/` in the last two days | the critical path is late and everyone piles onto it | apply the published cut order; add at most **one** helper to the critical path, and take them from ISA-2 |
