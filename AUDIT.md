# Complete Audit: `methodology.pdf`

**Method:** every checkable claim was tested against primary sources — the three source
PDFs, the cloned repositories (`repos/`), and the installed Triton 3.7.1 / PyTorch 2.12.1.
Where a claim was about generated IR, it was tested empirically (`verify/*.py`).
Line numbers refer to `methodology.pdf` (`/tmp/methodology_numbered.txt`).

**Verdict: 7/10 (B).** The technical core is *empirically confirmed*. The literature
grounding is unusually accurate (49 claims verified). What is wrong is the **evidence used
to justify design decisions** — §1.4's negative reference fails on 3 of 3 specific claims and
describes an archived repo; §4.1 and §12.2 rest on two provably false statements; §4.3/§5.1/§5.2
were designed against an *imagined* IR that the real dump disproves.

---

## PART 1 — FALSIFICATIONS AND UNSUPPORTED CLAIMS (17)

### F1. MLIR bindings "build-from-source burden" — **FALSE**
**Claim (L277–283):** a hand-rolled parser is "preferable to depending on MLIR Python
bindings, which would reintroduce the exact build-from-source burden the methodology is
designed to avoid per §3.1–3.2."

**Evidence:** the stock pip wheel has them.
```
triton._C.libtriton.ir.parse_mlir_module(path, ctx) + ir.load_dialects(ctx)
  T1 matmul: PARSED OK -> module
  T3 modulo: PARSED OK -> module
```
Zero build cost. (`verify/verify_mlir_bindings.py`.)

**But the conclusion survives for a different reason:** the Python-side traversal API is
crippled — `operation` has no `walk`, `region` has no `get_block`, `region` is not
subscriptable. You can parse but cannot walk. Hand-rolling is defensible; the *stated reason*
is false.

### F2. "TensorLift §9.2" does not exist — **CITATION FALSE**
**Claim (L304–307):** canon reduction "reported the same way TensorLift reports its
92.9%/41.2% figures (§9.2)."

**Evidence:** TensorLift has **five** sections only — §1 Intro, §2 Background, §3 Pipeline,
§4 Evaluation, §5 Related Work. There is no §6–§9. The figures live in **§4.2 / Figure 4 /
Table 3**. The *numbers* (92.9% PE, 41.2% VTA) are correct; the location is fabricated.

### F3. "Pipeline cost in §4.2" — **WRONG SECTION**
**Claim (L486–487):** "TensorLift reports its one-time per-accelerator pipeline cost
(§4.2, "Pipeline cost")."

**Evidence:** `Pipeline cost. Extraction and lifting are a one-time, per-accelerator cost...`
sits between **§4.1 Experimental Setup** (L575) and §4.2 Semantic Lifting Effectiveness (L649).
It is in **§4.1**. (Two-column layout makes this slightly ambiguous; the §4.2 attribution is not
supported.)

### F4. triton-shared "crashes on unstructured access" — **FALSE at HEAD**
**Claim (L139–142, repeated L319–321, L623–624).**

**Evidence — an unstructured path exists and is tested:**
- `lib/Conversion/TritonToUnstructured/TritonToUnstructuredPass.cpp` — 595 lines
- `lib/Conversion/UnstructuredToMemref/UnstructuredToMemrefPass.cpp` — 422 lines
- `python/examples/test_modulo.py`, `test_gather_scatter.py`, `test_index_select.py`
- `test/Conversion/TritonToUnstructured/` — gather tests, vector-add, softmax, matmul

**And the exact Tier 3 pattern is explicitly handled:**
```
include/triton-shared/AnalysisStructured/PtrAnalysis.h (L162–181)
  // ptr + (row_offsets[:,None] % mod_offset + some_number) + row_indices[:None]
  // (row_offsets[:,None] % mod_offset + some_number) is structured and has modulo.
```
`tts.make_tptr`'s `shape` field is documented as *"the boundary by which addresses wraps
around (constant zero indicates no wrap-around in the corresponding dimension)"* — i.e.
triton-shared has a first-class representation for **the methodology's Tier 3 negative control**.

### F5. triton-shared "monolithic hard-to-extend pass" — **UNSUPPORTED**
**Claim (L141).** `triton_shared.cc` is **8 lines**. No file, doc, or issue makes this claim.
The genuine maintainability smell is different and unreported: **two parallel pointer
analyses** — `lib/Analysis/PtrAnalysis.cpp` (1375 L) and
`lib/AnalysisStructured/PtrAnalysis.cpp` (1959 L).

### F6. "cited directly by its own maintainers" — **NOT SUPPORTED, and the repo is ARCHIVED**
**Claim (L142).** What the maintainers actually documented is generic:
> "the prototype only supports lowering programs that have structured memory access patterns"
> "The analysis is still in its early stage and does not support all scenarios"

Neither of those is "fragile", "crashes", or "monolithic". The referenced 2023 talk
(*Triton for All*, Ian Bearman, Microsoft) exists but I could not verify its content.

**The repository is dead:**
```
README.md line 1: **This repository is no longer maintained.**
last commit: 2025-12-05  "Update README with maintenance notice (#367)"
```
Framing an archived project's "maintainers" as an active negative reference is outdated.

### F7. Structured-access descriptor is not "three-field" — **INACCURATE**
**Claim (L315–320):** descriptor = "base pointer, per-dimension stride, per-dimension shape —
the same three-field addressing-attribute shape ... the same shape triton-shared's pointer
analysis attempts to recover."

**Evidence:** triton-shared's descriptor is larger.
```
TTS_MakeTensorPtrOp (TritonStructuredDialect.td L33):
  arguments = base, sizes, strides, offsets, shape,
              static_strides, static_offsets, static_shape, order   // 9 fields
PtrState (AnalysisStructured/PtrAnalysis.h L52):
  offsets, sizes, strides   // + base
```
Not `base/stride/shape`. Notably `offsets` and `order` are absent from the methodology's
descriptor, and `shape` there means *wraparound boundary*, not tile shape.

### F8. §5.2's op-shape list omits the case §6.1 depends on — **INCOMPLETE**
**Claim (L327–329):** expected op shapes are
`tt.addptr ← tt.splat/tt.broadcast ← arith.muli/arith.addi over tt.make_range/tt.get_program_id`.

**Evidence — the real in-loop operand pointers (Triton 3.7.1):**
```
%acc_25:3 = scf.for %_k = %c0_i32 to %1 step %c1_i32
    iter_args(%a_ptrs_34 = %a_ptrs_14, %b_ptrs_35 = %b_ptrs_24, %acc_36 = %acc)
  %a = tt.load %a_ptrs_34
  %acc_37 = tt.dot %a, %b, %acc_36
  %a_ptrs_38 = arith.muli %sak, %c32_i32
  %a_ptrs_40 = tt.addptr %a_ptrs_34, %a_ptrs_39
  scf.yield %a_ptrs_40, %b_ptrs_43, %acc_37
```
The operands are **`scf.for` iter_args advanced by a constant and re-threaded by `scf.yield`**.
Missing from the list: `scf.for` iter_args, `scf.yield`, and `tt.expand_dims` (6+ occurrences in
T1, essential to the index expression). Without iter_arg/`scf.yield` handling, §6.1's
precondition — *"two operands each already resolved to a structured-access descriptor"* —
**fails on the one kernel it exists to demonstrate.**

*Mitigating:* the increment is a **constant** (`sak*32`), not IV-derived, so no affine analysis
is needed — but the recurrence case is simply not named.

### F9. §4.3's "redundant splat/broadcast chains" premise — **UNSUPPORTED**
**Claim (L65–70, L295–303):** ttir has "redundant `tt.splat`/`tt.broadcast` chains" to collapse,
"same motivation as TensorLift's sign-extension collapsing".

**Evidence (T1 matmul, Triton 3.7.1):** `tt.splat`=13, `tt.broadcast`=6, `arith.muli`=10,
`tt.addptr`=5, `tt.expand_dims`=6+, extend/trunc=**0**. The splat/broadcast sequence encodes
`rm[:,None]*sam + rk[None,:]*sak` — **required broadcast expansion**, not redundancy. TensorLift's
A1/A2 fold sign-extension and `trunci/ext` round-trips; **ttir has neither**. The analogy has no
referent, and the §9.2 "IR reduction" metric will read near zero.

### F10. ACT's addressing phase is not equality saturation — **FALSE**
**Claim (L623–626):** "full generality would require **ACT-style equality saturation over
addressing expressions**."

**Evidence:** ACT's phases are split exactly the other way.
```
§5  Phase 1: Tensor Instruction Selection Using Equality Saturation   → computational attrs (α)
§6  Phase 2: Tensor Memory Allocation using Integer Constraint Programming → addressing attrs (β)
```
Addressing is **integer constraint programming**, not equality saturation. The methodology gets
this right in §0 (L17–19) and then inverts it in §12.2.

### F11. "ordering is free" — **OVERSTATED**
**Claim (L407–415):** ordering comes from the def-use graph's topological order; "Triton IR
already is a dataflow graph, so this ordering is free."

**Evidence:** the matmul's ordering is **region/loop order**, not flat dataflow: both `tt.load`s
must stay *inside* `scf.for`, `scf.yield` must be the block terminator, and the accumulator is
loop-carried. A flat topological sort over the def-use graph flattens the region hierarchy and
will mis-order or hoist them. Ordering is *derivable*, but requires region-aware traversal.

### F12. α attributes: extracted or pre-fixed? — **INTERNAL CONTRADICTION**
- L366–371 (§6.2): "extract tile shape and dtype (the computational/α attributes) **from the
  matched idiom**"
- L398–403 (§7.1): MAC templates "filling in the tile-shape/dtype computational attributes
  **fixed at schema-design time in §2.2**"

Both cannot be the source of truth. For a one-instruction toy ISA they coincide, but the document
asserts them as independent mechanisms.

### F13. "fixed-shape kernel" vs multi-shape testing — **INCONSISTENT**
- L41: scope cut = "fixed-shape kernel"; L243: "**fixed-shape single-block** matmul"
- L549–550: test "the exact block shape used in extraction, **one non-square shape**, and one
  shape at a **different block-size configuration**"

Either the kernel is fixed-shape or three shapes are exercised. (Conflating *block* shape with
*problem* shape is the likely intent, but it is never clarified.)

### F14. `TensorReduce` — **ORPHAN CITATION**
**L394:** "the same annotate/assemble separation **TensorReduce** and TAIDL/ACT both rely on."
TensorReduce appears exactly once in the document, carries no citation, and is not one of the four
source papers. Unsupported in-document.

### F15. "both benchmark against a hand-written reference" — **HALF TRUE**
**Claim (L205–212).** TensorLift: **true** ("matches hand-written Gemmini kernels, 1.014×
geometric mean"). ACT: compares against **expert-written kernel libraries and production
compilers** (e.g. neuronx-cc), not a hand-written reference artifact. "Both" overreaches.

### F16. D8 conflation — **INACCURATE**
**Claim (L91–96):** D8 classifies args "exactly as TensorLift classifies scratchpad vs.
accumulator roles." D8 actually *"Classifies each memref argument by its load/store footprint,
labels scalar arguments as control attributes"*. Scratchpad/accumulator is TensorLift's **data
model**, not D8's output.

### F17. "memref-like argument" in ttir — **TERMINOLOGY MISMATCH**
**Claim (L92).** At extraction time ttir has **no `memref` values** — targets are
`tensor<64x64x!tt.ptr<f32>>`. `memref` appears only after TensorLift's own lifting. Carrying the
term over to ttir is a category slip.

---

## PART 2 — VERIFIED TRUE (49)

### Literature attribution & shape (all correct)
| # | Line | Claim | Evidence |
|---|---|---|---|
| T1 | L11 | TAIDL/ACT = Jain et al. | TensorLift ref [20] TAIDL, [21] ACT — Devansh Jain et al. |
| T2 | L17–18 | ACT = equality saturation + constraint programming | ACT §5, §6 |
| T3 | L22 | TensorLift = Gao et al. | Ruijie Gao, Haoran Jin, Jirong Yang, Nathaniel Bleier |
| T4 | L23–25 | TensorLift 4-phase pipeline | "fall into four phases (Table 2)" |
| T5 | L26–29 | prove-cheap + differential + annotate-don't-rewrite | Z3 SMT core; golden simulator rest; L567 |
| T6 | L31 | Triton-MTIA = Zhu et al. | Haishan Zhu (first author) |
| T7 | L32–35 | MTIA 4-stage flow | Fig. 4 |
| T8 | L36–37 | MTIA GEMM vs long-tail + coverage stats | §7.1.1, §7.1.3, Table 1 (94.5%/94.7%) |
| T9 | L119–130 | MTIA Fig. 4 as skeleton; DMA-vs-scalar → DMA-vs-unsupported | MTIA §3.3 |
| T10 | L60 | TensorLift Table 2 = eight passes | confirmed, exact |
| T11 | L64 | A1/A2 = canonicalization | A1 canon-bitmanip, A2 narrow-types |
| T12 | L72–73 | **B3–B5 = MAC, control specialization, clamp** | detect-mac, specialize-control, detect-clamp — exact |
| T13 | L78–89 | C6/C7 inverted; Triton already emits the loop | C6 reconstruct-loops, C7 lift-to-linalg; **empirically confirmed** |
| T14 | L91–96 | D8 = metadata emission | D8 emit-taidl-metadata |
| T15 | L98–101 | ACT §3 = problem formalization, tiled kernel input, α/β, soundness/completeness | §3.1 "Tiled Kernels: Input", §3.3, L485–487 |
| T16 | L103–107 | α/β split | ACT L443–445; TAIDL API has `computation_attr`/`addressing_attr` **parameters** |
| T17 | L109–112 | pii machinery + equality saturation are ACT's | ACT L164, L544, §5.4 |
| T18 | L114–117 | soundness/completeness vocabulary | ACT §3.3 |
| T20 | L170–171 | TAIDL has `add_data_model` | `taidl/accelerator.py:37` |
| T21 | L177–178 | TAIDL has `add_instruction` + `add_semantics` | `accelerator.py:40`, `instruction.py:55` |
| T22 | L188–191 | accumulate flag, no ISA loop primitive | consistent: ACT tiles inner loops, TAIDL describes per-tile ops |
| T23 | L215–219 | ttir syntax not stable across releases | triton-shared pins `triton-hash.txt`; their PR #345 = "Update triton to e44bd1c…" |
| T25 | L231–237 | TensorLift PE vs Load/Store Controllers, different numbers | Table 3 modules: PE, Execute/Load/Store Ctrl, TensorGemm, TensorAlu |
| T29 | L285–293 | TensorLift keeps `scf.if`/named memrefs over phi nodes | L507, L512 |
| T30 | L291 | loc names / iter_args / scf.for bounds preserved | **empirically: 169 locs; locs carry variable names** (`loc("a_ptrs")`, `loc("pid_m")`) |
| T31 | L302–303 | TensorLift Phase A = sign-extension collapse | A1 |
| T32 | L306 | 92.9% / 41.2% | abstract |
| T34 | L384–385 | TensorLift §3.2: "passes annotate rather than rewrite" | **verbatim**, L567 |
| T35 | L387–389 | `tensorlift.mac` / `taidl.*` annotations | present in TensorLift |
| T36 | L411–413 | TensorLift §3.3 recovers ordering from FSM register updates | §3.3 "Stage 3: TAIDL Assembly" |
| T37 | L430–433 | TensorLift opaque-fallback quote | **near-verbatim**, L570 |
| T38 | L434–435 | ACT soundness requirement | §3.3 |
| T39 | L454–455 | TensorLift **§4.4 "Specification Completeness"** | exact |
| T40 | L458–459 | TensorLift reports per-module | Table 3 |
| T41 | L466–467 | TensorLift **§4.2/Table 3** | §4.2 Semantic Lifting Effectiveness; Table 3 |
| T42 | L470 | TensorLift Figure 4 before/after lines | Figure 4 caption |
| T43 | L473 | ACT **§8.3–§8.6** | Comparative Evaluations |
| T44 | L474–475 | TensorLift §4.4 hand-written reference | L572 |
| T46 | L541–543 | TensorLift Z3 / golden-simulator dual track | verified |
| T48 | L595–601 | TensorLift/ACT explicit about verification bounds | verified |

### Empirically confirmed (the project's core bets)
| # | Line | Claim | Result |
|---|---|---|---|
| **T24** | L221–229 | **GPU-free ttir via `triton.compile()` + explicit target — go/no-go gate** | **PASSES.** 160 lines of ttir, no driver/GPU |
| **T27** | L243–245 | **Tier 1 = `tt.dot` inside `scf.for` with `iter_args`** | **EXACT MATCH** — 1 `scf.for`, 1 `iter_args`, 1 `tt.dot`, 1 `scf.yield` |
| **T33** | L351–364 | **§6.1 MAC pattern**, quoted form `%acc:N = scf.for … iter_args(…) { %acc' = tt.dot %a,%b,%acc …; scf.yield … }` | **EXACT MATCH** |
| T26 | L239–242 | Tier 0 vector add, no compute idiom | 0 `tt.dot`, 0 `scf.for` — true negative holds |
| T28 | L252–260 | Tier 3 = modulo wraparound = MTIA §8.1 pattern | `arith.remsi` survives into ttir; MTIA §8.1 verified verbatim |
| T45 | L525–528 | Tier 0 has no `tt.dot` | confirmed |
| T49 | L45–48 | "nothing today takes a declarative ISA description and emits a Triton-IR-consuming backend" | **SURVIVES.** ACT→XLA only; MTIA/triton-shared/triton-linalg/intel all hand-written |

**The two headline technical bets (§3.2 gate, §6.1 pattern) are both correct.** That is the
backbone of the project and it holds.

---

## PART 3 — COULD NOT VERIFY (honest gaps)

| Line | Claim | Reason |
|---|---|---|
| L40–43 | "The SegFault brief" scope cut | Not among the four supplied PDFs; no such document in the folder |
| L142 | the 2023 talk containing the three failure-mode statements | Talk exists (Ian Bearman, Microsoft, *Triton for All*) but video content is not machine-verifiable |
| L393–394 | `TensorReduce`'s annotate/assemble separation | Orphan citation; not a source paper; no reference given |
| L152–158 | §1.5 inheritance map | An intended internal deliverable, not a factual claim |

---

## PART 4 — WHAT IS *NOT THERE* IN THE PROJECT (independent of the above)

These are scope/architecture gaps in the plan itself, not errors in the prose.

1. **No parse-failure path.** §8's `UNSUPPORTED` mechanism covers *semantic* non-match. Nothing
   covers ttir the hand-rolled tokenizer cannot parse (nested attr dicts, multi-result ops, region
   structure, `#loc` tables). §10.6 requires "fails cleanly" — that needs a `PARSE_UNSUPPORTED`
   route too.
2. **The YAML is a parameter file, not a description language.** §2.2 hard-codes exactly one DMA,
   one MAC, one elementwise. The ISA description supplies *values*, it does not drive generator
   *structure*. TAIDL by contrast is a Python DSL + an ANTLR4 grammar (`taidl/antlr4/IDLV2Parser.py`)
   with `set_inputs`/`set_outputs`/`add_semantics`, `cost`, `update`, `constraints` — the
   methodology's §2.2 drops `cost`/`update`/`constraints`, which is precisely what makes ACT's
   cost-driven §7 selection work.
3. **No reusable pass interface.** Every reference implementation is a **C++ MLIR pass**
   (triton-shared, triton-linalg, intel-xpu, MTIA). This plans a Python text tool. Legitimate (it
   avoids building MLIR) but the artifact cannot be dropped into Triton's pipeline and is not
   comparable to the references. §12 does not say this.
4. **No schedule.** "5-day budget" is invoked four times (L87, L335, L450, L612) but never laid
   out. A hand-rolled parser + NumPy interpreter + toy ISA + assembler + 4-kernel corpus + fuzz
   kernel exceeds it.
5. **Coverage metric is gameable.** §9.1 counts annotated *op nodes*. One unannotated op early in
   the chain invalidates every downstream annotation while the percentage stays high. Needs
   "kernel fully lowered: yes/no" and/or largest fully-lowered subgraph.
6. **§9.3's structural equivalence is the wrong shape.** "same instruction count and shape" will
   flag *valid* alternative lowerings as failures — which the same paragraph concedes.
7. **tf32 makes the tolerance question mandatory, not optional.** The matmul ttir carries
   `tt.dot … inputPrecision = tf32`. A NumPy fp32 interpreter cannot match torch/triton numerics.
   §11.3 says only "record the tolerance used."
8. **Fixtures test regression, not generality.** Pinning Triton + freezing fixtures means the
   parser is only ever exercised against one version's output. §10.1's "independent of a live
   Triton install" is a strength for CI and a blind spot for robustness.
9. **No transferability test.** The ISA is designed by the same person who writes the generator,
   so the ISA→generator loop is self-referential. Generality is asserted, never tested. One extra
   toy ISA would convert the demo into evidence.
10. **§1.4's premise contradicts its own source.** MTIA — the methodology's "target compiler
    shape" — **reuses** triton-shared's pass: *"we leverage the open-source Triton IR analysis
    pass from triton-shared [31]"* (MTIA §3.3), and its Listing 2(d) shows `%desc = tts.make_tptr
    %x_ptr, …`, i.e. triton-shared's op. The methodology borrows MTIA's architecture then rebuilds
    the one component MTIA actually reused.

---

## PART 5 — RATING

| Dimension | Grade | Basis |
|---|---|---|
| Literature fidelity (attributions, pipeline shapes, quotes) | **A−** | 49 claims verified; only ACT-phase conflation (F10) and section-number slips (F2, F3) |
| Core technical feasibility | **A** | §3.2 gate and §6.1 pattern both empirically confirmed on Triton 3.7.1 |
| Auditability / verifiability | **A** | every claim is checkable against primary sources — this audit was possible at all because the document commits to specifics |
| Fit to the *actual* IR | **D+** | §4.3 premise false (F9), §5.2 list omits iter_args (F8), §5.1 descriptor wrong (F7), ordering claim overstated (F11) |
| Negative-reference accuracy (triton-shared) | **D** | 3 of 3 specific claims unsupported/false (F4, F5, F6); subject repo is archived |
| Internal consistency | **C** | α-attribute contradiction (F12), fixed-shape vs multi-shape (F13) |
| Novelty framing | **B−** | survives search (T49) but rests on a partly-wrong landscape characterization (§1.4) |
| Scope realism | **C+** | no schedule; parser + interpreter + 6 kernels in 5 days is optimistic |

### **Overall: 7/10 — B**

**The plan is sound. The evidence used to justify parts of it is not.**

Nothing is *conceptually* wrong: the gap is real, the target is unclaimed, the two load-bearing
technical bets are verified, and the testing discipline (§10 — true-negative tests, idempotency,
Tier 3 negative control, recorded tolerances) is better than most published work in this space.

What fails is §1.4, which builds a "negative reference" out of an **archived** repository using
three claims that its source does not support — while ignoring the two facts that actually matter:
triton-shared already implements both the structured descriptor (`tts.make_tptr`) *and* an
unstructured fallback, and the paper the methodology borrows its architecture from **reuses** that
implementation rather than rebuilding it.

### Fix list, in priority order

1. **Rewrite §1.4** from the real README text and the real archive notice; drop "crashes" and
   "monolithic". Replace with the verifiable facts: incomplete op coverage (see issues #201, #233,
   #243, #298, #309), two parallel pointer analyses, archived 2025-12-05.
2. **Fix §5.2/§6.1** from `/tmp/T1_matmul.ttir`: add `scf.for` iter_args, `scf.yield`,
   `tt.expand_dims`; state that increments are constants.
3. **Fix §4.1's reason** (bindings ship; traversal API doesn't) and **§4.3's premise** (no redundancy
   to collapse in ttir — pick a different canonicalization, or drop it and the §9.2 metric).
4. **Fix §12.2's ACT phase swap** (addressing = constraint programming, α = equality saturation).
5. **Fix §5.1's descriptor** (base + sizes + strides + offsets + shape + order) and decide
   explicitly whether to piggyback on `tts.make_tptr` semantics as a conformance oracle.
6. **Fix citations** §9.2 → §4.2; §4.2 → §4.1; cite or drop `TensorReduce`.
7. **Add**: a second toy ISA (transferability), a parse-failure path, a schedule, a
   "fully-lowered yes/no" metric, and an explicit statement that this is a Python tool that does
   not plug into Triton's pipeline.
