# Research & Decision Log: Triton-IR-to-Declarative-Toy-ISA

**Phase 0 output** for `001-triton-to-toy-isa`. Input: `methodology.pdf` + `AUDIT.md` +
`methodology-v2.tex`, plus primary sources cloned under `repos/` and Triton 3.7.1 / PyTorch 2.12.1
installed locally.

**Method.** The `interview-me` protocol was run reflexively: hypothesis with a confidence number first,
then one question at a time, each with the answer I *wanted* to give attached, because the wanted answer is
the one that hides the assumption. Where an answer could be checked, it was checked before it was written
down. Every claim below carries its evidence.

---

## The restate (intent, in the user's own framing)

- **Outcome**: a working vertical slice that takes a declarative ISA description and produces a
  `torch.compile`-registered backend for it, with transferability measured across two ISAs.
- **User**: whoever must bring a custom accelerator to PyTorch without hand-writing 2,590 operator kernels —
  and the reviewer who has to believe it.
- **Why now**: four organizations have hand-built this layer separately (Meta, Microsoft, Cambricon, Intel);
  the academic automation (ACT, TensorLift) targets XLA, which PyTorch does not use.
- **Success**: a real `torch` op runs on the generated device within declared tolerance; coverage is reported
  honestly per tier and per ISA; the second-ISA transfer numbers exist, whatever they are.
- **Constraint**: ~6.5 days, no MLIR/LLVM build, one author, four corpus kernels.
- **Out of scope**: operator coverage at vendor scale, production performance, multi-bank memory, learned
  cost models, being a drop-in Triton pass.

---

## Foundational truths

Ten statements that are true of this problem regardless of how the project is built. Each one was verified
and each one constrains a design decision.

| # | Truth | Evidence | Design consequence |
|---|---|---|---|
| T-1 | **The reduction structure is already in the IR.** Triton emits `scf.for` with `iter_args` around `tt.dot`; nothing must be reconstructed. | Tier-1 fixture: exactly one `scf.for`, one `iter_args`, one `tt.dot`, one `scf.yield`. | An entire TensorLift phase (C6/C7) collapses into pattern recognition. This is the project's budget argument. |
| T-2 | **`ttir` text is not a stable interface.** Its syntax changes between Triton releases. | triton-shared pins `triton-hash.txt`; its PR #345 is "Update triton to e44bd1c…". | Fixtures are the interface, not the live compiler. Pin the version, snapshot the text, treat regeneration as a reviewed act. |
| T-3 | **The parser is the critical path.** Not the ISA, not the emulator. | ~18 distinct op names in Tier 1, but region nesting, multi-result ops, layout-typed strings and a `#loc` table to handle; no parse-failure path existed in v1. | Hand-rolled parser with a hard failure route (`PARSE_UNSUPPORTED`); build it first and test it on all four tiers before anything else. |
| T-4 | **A flat topological sort is wrong.** | In Tier 1 both `tt.load`s must stay inside `scf.for`, `scf.yield` terminates the block, and the accumulator is loop-carried. | Traversal must be region-aware. v1's "ordering is free because the IR is a dataflow graph" was false. |
| T-5 | **Float addition is not associative.** | Definitional; and the corpus contains a tf32 dot. | Accumulation order is a schema field. Without it, any tolerance is unfalsifiable — a mismatch can always be blamed on reassociation. |
| T-6 | **`tf32` is a literal in the corpus, not a footnote.** | Tier-1 `tt.dot` carries `inputPrecision = tf32`. | A float32 emulator cannot match torch numerics. Per-dtype tolerance with a recorded derivation is mandatory. |
| T-7 | **Coverage as a node fraction is gameable.** | One unannotated op early in a chain invalidates every downstream annotation while the percentage stays high. | `fully lowered` is a boolean, reported before any percentage; the annotated fraction is labelled an upper bound. |
| T-8 | **An ISA with one instruction per kind is a parameter file.** | v1 declared one DMA, one MAC, one elementwise → nothing to choose → generator degenerates. TAIDL's `add_instruction` already takes `cost`, `update`, `constraints`. | Two DMA + two MAC variants, each with an admissibility predicate and a cost. |
| T-9 | **The cell is empty, and empty for a boring reason.** | Meta (MTIA), Microsoft (triton-shared), Cambricon (triton-linalg), Intel (xpu) all target **Triton, hand-built**. ACT and TensorLift auto-generate for **XLA-HLO IR**. Search found no counter-example. | The claim is a *composition* claim, not a novelty-of-technique claim. Frame it as such without hedging into "already done". |
| T-10 | **There is a second seam nobody names.** | `native_functions.yaml` has 2,590 `- func:` entries; Inductor's lowering surface is 209 `register_lowering()` rules. | The 2,590 are a separate problem from the device plumbing. Say which one is being attacked. |

---

## R1. What is the unit of value — the kernel or the operator surface?

**Wanted answer.** "The kernel: we lower a matmul correctly." It is concrete, it is demonstrable, and it is
what the pipeline actually does.

**Why that answer is the one that sounds right.** It is measurable in isolation, and every failure mode of the
project shows up as a kernel that does not lower.

**Harder answer.** The kernel is the unit of *work*; the operator surface is the unit of *value*. A vendor
does not need one matmul lowered; they need a model to run, which is a coverage question across ~209 lowering
rules that stand in for 2,590 hand-written kernels. Measured on one kernel, this project is indistinguishable
from a class exercise.

**Decision.** Keep the kernel as the unit of implementation, make the operator surface the unit of the claim:
the artifact is a registered device with a coverage report, not a program that lowers a matmul. This is why
US1 (the seam) outranks US2 (recognition) in priority, although US2 is where the work is.

**Evidence.** `aten/src/ATen/native/native_functions.yaml`: 2,590 `- func:` entries;
`torch/_inductor/lowering.py`: 209 `register_lowering()` (265 across `torch/_inductor`); Triton-MTIA Table 1
reports 94.5%/94.7% success across 60 model types and states Inductor kernels were 50% of layers.

---

## R2. Is the deliverable a compiler pass or a program — and does "a program" make it worthless?

**Wanted answer.** "A program is fine, because the hard part is the analysis." This is the comfortable answer,
and I gave it in an earlier draft by listing it under "limitations" rather than deciding it.

**Harder answer.** It is a Python program operating on frozen text. Every reference implementation — ACT,
triton-shared, triton-linalg, intel-xpu — is a C++ MLIR pass. The artifact therefore cannot be dropped into
Triton's pipeline and is not comparable to any of them. That is a real narrowing and it belongs in the claim,
not in an apology.

**But the reason matters, and the obvious reason is false.** The justification "MLIR bindings would
reintroduce the build-from-source burden" is **false**: the shipped wheel contains them.

```
triton._C.libtriton.ir.parse_mlir_module + ir.load_dialects
  T1 matmul: PARSED OK   T3 modulo: PARSED OK      (verify/verify_mlir_bindings.py)
```
The *true* reason is that the Python-side traversal API is crippled: `operation` has no `walk`, `region` has
no `get_block`, `region` is not subscriptable. You can parse but you cannot walk.

**Decision.** Hand-roll the parser and traversal. Reject the build-cost justification; use the traversal-API
justification. State in the claim and in the limitations document that this is a text tool that does not plug
into Triton's pipeline.

**Confidence.** 92% — the remaining 8% is the risk that the MLIR bindings carry an undocumented traversal
entry point.

---

## R3. What makes the word "generate" true rather than "template"?

**Wanted answer.** "It generates code from a description, so it's a generator." That is a definitional dodge:
a `for` loop over three instructions is also "generating code from a description".

**Why that answer sounds right.** It is technically true, and the distinction only matters to someone who has
seen a real generator.

**Harder answer.** Generation requires a *decision*. With one DMA, one MAC and one elementwise instruction
there is nothing to decide; the schema supplies values, not structure. ACT's formulation is the standard:
enumerate the admissible lowerings, take the minimum-cost one (ACT §7). Admissibility and cost are what turn a
description into a decision procedure.

**Decision.** Two DMA and two MAC instructions; a decidable admissibility predicate per instruction; an
explicit cost; a `constraints` field retained in narrow form (and `update` dropped, stated). Report selection
quality against an exhaustive oracle, so that "it chose" is measurable and the gap to optimal is visible.

**Evidence.** TAIDL's own signature is `add_instruction(instruction, computation_attr, addressing_attr, cost,
update, constraints)` — the parameters v1 dropped are exactly the ones that make ACT's selection work.

---

## R4. What is the actual structure of the IR we must recognise?

**Wanted answer.** `tt.addptr ← tt.splat/tt.broadcast ← arith.muli/arith.addi over tt.make_range/get_program_id`.

**Harder answer.** **That list omits the case the demonstration kernel depends on.** Measured on
Triton 3.7.1:

```
%acc_25:3 = scf.for %_k = %c0_i32 to %1 step %c1_i32
    iter_args(%a_ptrs_34 = %a_ptrs_14, %b_ptrs_35 = %b_ptrs_24, %acc_36 = %acc)
  %a = tt.load %a_ptrs_34
  %acc_37 = tt.dot %a, %b, %acc_36, inputPrecision = tf32
  %a_ptrs_38 = arith.muli %sak, %c32_i32
  %a_ptrs_40 = tt.addptr %a_ptrs_34, %a_ptrs_39
  scf.yield %a_ptrs_40, %b_ptrs_43, %acc_37
```
The in-loop operands are `scf.for` `iter_args` advanced by a **constant** and re-threaded by `scf.yield`.
Missing from v1's list: `scf.for` iter_args, `scf.yield`, `tt.expand_dims` (6+ occurrences in Tier 1).
Without them the idiom matcher's precondition fails on the one kernel it exists to demonstrate.

**Also falsified:** v1's claim of "redundant splat/broadcast chains" to canonicalise. Measured: 13 `tt.splat`,
6 `tt.broadcast`, 10 `arith.muli`, 5 `tt.addptr`, and **0** `arith.extsi`/`arith.trunci`. The splats encode
required broadcast expansion (`rm[:,None]*sam + rk[None,:]*sak`); there is no redundancy and no sign-extension
round-trip. Canonicalisation is therefore re-scoped to vocabulary closure and taken off the critical path.

**Decision.** The recogniser's required op set is taken from the dumped fixture, not from the paper. A
mandatory loop-recurrence test (iter_args → `tt.load` → descriptor) exists as a regression against exactly
this omission.

**Confidence.** 97% — measured directly, on the pinned version.

---

## R5. Which reference work is the competitor, and which is the template?

**Wanted answer.** "Triton-MTIA and the Triton-family repos are the competitors; ACT and TensorLift are the
theory." This inverts which side of the landscape each thing sits on.

**Harder answer.** Two different axes were being conflated.

| Work | Org | Targets | Relationship |
|---|---|---|---|
| Triton-MTIA | Meta | **Triton**, hand-built from the ground up | architecture template (§3.3 four-stage flow) |
| triton-shared | Microsoft | **Triton** | archived; source of `tts.make_tptr` semantics, used as a conformance oracle |
| triton-linalg | Cambricon | Triton → Linalg | comparison only |
| intel-xpu-backend | Intel | Triton | comparison only |
| **ACT** | UIUC | **XLA-HLO IR** | source of the α/β split, soundness vocabulary, cost-driven selection |
| **TensorLift** | Michigan | **XLA** (RTL → TAIDL) | source of the four-phase/eight-pass pipeline and annotate-don't-rewrite |

So: **industry hand-builds Triton backends; academia auto-generates XLA backends.** ACT and TensorLift are the
theoretical sources, not competitors; MTIA is the architecture template and *also* the best evidence that the
layer is expensive (a compiler from the ground up, 16 kernels ported, 5 requiring rewrites).

**Consequence that v1 missed.** MTIA *reuses* triton-shared's analysis pass — "we leverage the open-source
Triton IR analysis pass from triton-shared [31]" (§3.3), with `%desc = tts.make_tptr …` in its Listing 2(d).
v1 built a negative reference out of an archived repo while borrowing the architecture of the one paper that
**depends** on it. Corrected: `tts.make_tptr` is used as a conformance oracle for our recogniser (FR-009), and
the dependency is stated rather than avoided.

---

## R6. Is the "missing cell" real — or is it empty for a boring reason, which would be a problem?

**Wanted answer.** "It's real, so the project is novel." Correct, and it is what makes the project worth doing.

**Harder answer.** It is real, and it is empty for a boring reason: two communities never needed each other's
half. That is *good news*, not bad — an empty cell for a boring reason does not have to be won from an
incumbent. But it means the framing must be a composition claim, which is why the abstract says "a working
vertical slice" rather than "the missing cell in the landscape".

**Evidence.** Search and repo inspection found no work taking a declarative ISA description and emitting a
Triton-IR-consuming backend. MTIA is hand-written; the automation is XLA-side.

**Decision.** Frame as composition + measurement. Do not claim technique novelty (the technique is
ACT's/TensorLift's, already published). Claim: *this composition, on this IR, with the seam closed and the
transfer measured.*

---

## R7. What does "correct" mean when the IR says `tf32`?

**Wanted answer.** "Record the tolerance used." That is what v1 said and it is not enough, because the
tolerance is not a free parameter — it is determined by the IR.

**Harder answer.** `inputPrecision = tf32` is a literal attribute of the corpus dot. A float32 NumPy emulator
**cannot** match torch/triton numerics, because the tf32 path truncates mantissa bits on the multiply inputs.
So either the emulator reproduces the declared precision explicitly, or the tolerance is derived from the
precision difference and stated per dtype with the derivation recorded. Separately, T-5 makes accumulation
order part of the contract: the emulator must reduce in the order the ISA declares, or the tolerance cannot
distinguish rounding from reassociation.

**Decision.** Emulator implements declared `inputPrecision` and declared accumulation order. Integer and
low-precision paths: exact match. Float paths: within tolerance, tolerance derived per dtype, derivation
recorded in the report. Triton-MTIA §8.1 independently reports mixed-precision handling as a source of
out-of-tolerance results requiring explicit casting, which corroborates that this is a real failure class and
not pedantry.

---

## R8. Where does the device boundary actually sit in PyTorch, and how much of it must be implemented?

**Wanted answer.** "It's huge — a vendor needs thousands of kernels." That is how v1 implicitly framed it, by
aiming at a bespoke NumPy harness instead.

**Harder answer.** There are two surfaces, and conflating them overstates the problem by an order of
magnitude.

- **Device integration**: registration, runtime, streams/events, allocator, RNG, serialization, autocast,
  distributed `ProcessGroup`, profiler stubs, autoload. **Bounded.** PyTorch's own in-tree reference
  (`OpenReg`) implements it at *stub level by design*: "not to implement a fully functional,
  high-performance PyTorch backend, but to serve as a minimalist reference implementation for mechanism
  verification", implementations "should be **STUB-level**".
- **Compiler backend**: a `torch.compile` backend plus the device abstraction the compiler uses.
  `DeviceInterface` spans **30 method slots across 4 nested classes** (`device`, `Event`, `Stream`,
  `Worker`), most with usable defaults. Verified in `torch/_dynamo/device_interface.py`.
- **Eager operator library**: 2,590 ATen schemas — a *separate* problem, and the one the compiled path
  largely removes.

**Decision.** Target the compiler backend. Copy the structure of `torch_openreg/compiler.py`, which contains
both a `register_backend` entry point and an `OpenRegInterface(DeviceInterface)` subclass. Demote the NumPy
interpreter from "the front door" to "the device emulator", the role `libopenreg.so` plays. Prove the seam on
day 1 with a hard-coded kernel.

**Evidence.** `torch/_dynamo/backends/registry.py` (`register_backend`),
`torch/_dynamo/device_interface.py` line 40, the OpenReg README, and the OpenReg `compiler.py`.

---

## R9. What must be reused rather than built?

**Wanted answer.** "Build it: the whole point is that we can generate the backend." Reuse feels like cheating
in a project whose thesis is automation.

**Harder answer.** Distinguish *what the project is testing* from *what the project is demonstrating*. The
thesis is that the ISA→backend seam can be automated; it is not that `DeviceInterface` must be re-derived, or
that a structured-pointer analysis must be re-invented. Reusing:

- **`tts.make_tptr` semantics** as a conformance oracle (FR-009) — an oracle is *stronger* evidence than a
  from-scratch reimplementation, and it is what stops the recogniser from being a private dialect.
- **`torch_openreg/compiler.py`** as the registration pattern (FR-023/024).
- **Triton's own `scf.for`/`tt.dot` structure** rather than reconstructing loops (T-1).

The line is: reuse the *contracts* and the *seam*; build the *pipeline the project is claiming to have
automated*.

**Decision.** Reuse is stated explicitly in the limitations document per item, with the reason, so that
"reused" can never be mistaken for "built".

---

## R10. What must never happen?

**Wanted answer.** "It shouldn't crash." Too weak: a crash is loud and fixable.

**Harder answer.** The failure that voids the project is a **silent one**: a lowering that produces a
plausible program which computes something else, or a report claiming coverage it does not have. ACT states
the requirement directly (§3.3): a generator must never silently miscompile. The second silent failure is a
report: `fully lowered = true` on a kernel with an unlowered operation at the head of a chain.

**Decision.** Soundness is a constitution-level principle (Principle II), not a feature. Every unlowered
operation becomes a first-class `UNSUPPORTED` marker carrying its originating `loc`; every unparseable input
becomes `PARSE_UNSUPPORTED` naming line and column; coverage is a boolean before it is a percentage; the fuzz
pass asserts "no crash, always a marker" (SC-008).

---

## Additional decisions (alternatives considered and rejected)

| # | Decision | Alternative rejected | Why |
|---|---|---|---|
| D1 | `register_backend` + `DeviceInterface` | bespoke NumPy harness (v1) | the harness cannot be invoked by any framework; the value is unmeasurable |
| D2 | hand-rolled parser + traversal | MLIR Python bindings for traversal | bindings parse but do not walk (no `walk`/`get_block`); *not* build cost, which is false |
| D3 | `tts.make_tptr` as conformance oracle | runtime dependency on triton-shared | repo is archived (2025-12-05); an oracle gives the semantics without the dependency |
| D4 | own four-stage architecture (MTIA's shape) | adopt MTIA's pipeline wholesale | MTIA's middle-end *is* triton-shared's pass, which we cannot take |
| D5 | two DMA + two MAC variants with costs | one of each (v1) | no alternatives → no decision → not a generator |
| D6 | greedy enumeration + min cost | equality-saturation search | addressing is integer constraint programming in ACT (§6), not equality saturation (§5); search does not fit the budget |
| D7 | analytic per-instruction cost | learned cost model (IR2Vec-style) | needs a closed vocabulary first; retained as future work with the analytic model as the baseline to beat |
| D8 | second ISA inside the plan | second ISA as future work (v2) | a self-referential generator/ISA pair tests only itself; half a day buys the falsification |
| D9 | YAML + one rule module per ISA | a full DSL + grammar (TAIDL) | TAIDL is a Python DSL with an ANTLR4 grammar and a semantics language; the fidelity claim is narrowed instead of overstated |
| D10 | region-aware traversal | flat topological sort (v1) | both `tt.load`s must stay inside `scf.for`; `scf.yield` terminates; accumulator is loop-carried |
| D11 | fixtures frozen against a pinned Triton | parse the live compiler output | `ttir` syntax is not stable (T-2); fixtures are the regression baseline, and that limitation is written down |
| D12 | tests labelled by class (unit/contract/integration/transfer/negative) | a single end-to-end suite | a failure must localise to one component instead of surfacing as "the demo is wrong" |

## Open items carried into `plan.md`

- **ISA-2 corpus (US4 / block 7)**: re-express the four existing kernels in ISA-2, or author two new kernels? Default:
  re-express all four (cheap, and the comparison is apples-to-apples against ISA-1's coverage), plus one new
  kernel if block 7 has budget. Decided at block 7 and recorded here.
- **Block 7 is in the published cut order** as the second thing to drop. This is deliberate: a transfer rate
  measured on two of four kernels is still a measurement, and the limitation is recorded.
