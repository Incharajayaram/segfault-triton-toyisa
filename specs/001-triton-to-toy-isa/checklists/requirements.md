# Requirements Quality Checklist: Triton-IR-to-Declarative-Toy-ISA

**Purpose**: reviewer-owned review of the *specification*, not the implementation. These items ask whether the
requirements are complete, unambiguous, measurable, traceable and consistent — the questions a reviewer should
be able to answer before any code is written.
**Created**: 2026-09-14
**Feature**: `specs/001-triton-to-toy-isa/spec.md`
**Review ownership**: mark `[x]` only when the reviewer determines the requirement-quality criterion is
satisfied. `[x]` means "this requirement is well-formed", never "the code is done".

## Requirement completeness

- [ ] CHK001 Every user story has an "Independent Test" that a single test file can satisfy without the other stories
- [ ] CHK002 Every functional requirement names an observable behaviour, not an internal mechanism
- [ ] CHK003 The device integration surface is bounded in the spec: which `DeviceInterface` slots must work, which may raise, and where the reasons are recorded
- [ ] CHK004 The eager fallback floor is acknowledged as a requirement (FR-025), not left implicit
- [ ] CHK005 The ISA description format is specified as a contract (fields, constraint language, mandatory accumulator semantics) rather than as an example
- [ ] CHK006 The failure route for unparseable input is a requirement (FR-001) with a named diagnostic type, not an implementation note
- [ ] CHK007 Both true-negative cases (Tier 0, Tier 3) are requirements (FR-010, FR-033), not test trivia
- [ ] CHK008 The transfer experiment specifies what must transfer *unchanged* and where edits are permitted
- [ ] CHK009 The limitations document is a deliverable with a required cross-reference per entry (FR-031)

## Requirement clarity and measurability

- [ ] CHK010 Every success criterion is measurable from an artifact: a number, a boolean, or a file
- [ ] CHK011 SC-004 ("100% of operands") states its denominator explicitly, so partial success cannot be reported as success
- [ ] CHK012 SC-006 does not require a flattering transfer rate — only that the numbers and the edit locations exist
- [ ] CHK013 The tolerance requirement states that the derivation is recorded, not merely the value (FR-022)
- [ ] CHK014 "Generation latency" is defined as a one-time per-shape cost, distinct from per-compilation runtime (SC-007)
- [ ] CHK015 The coverage metric's ordering is normative: boolean before any fraction (FR-027)
- [ ] CHK016 The word "generate" is operationally defined (admissible enumeration + minimum cost), not asserted
- [ ] CHK017 "Fully lowered" has a definition that is checkable from the emitted program (data-model invariant 4)

## Consistency and traceability

- [ ] CHK018 Every FR maps to at least one task in `tasks.md`, and every task cites its FR or SC
- [ ] CHK019 Every contract in `contracts/` lists its satisfying FRs
- [ ] CHK020 The schedule in `plan.md` matches the block list in `methodology-v2.tex` (including the honest 6.5-day total)
- [ ] CHK021 The cut order is stated once and referenced, never re-derived differently in two documents
- [ ] CHK022 `data-model.md` and the contracts agree on field names (`strides` vs `stride`, `offsets` present, `shape` = wraparound boundary)
- [ ] CHK023 No document claims a reduction that `research.md` T-8-adjacent evidence contradicts
- [ ] CHK024 The claim in `spec.md` matches the claim stated in `methodology-v2.tex` §1 word for word in substance
- [ ] CHK025 ISA-2's differences are enumerated in exactly one place (`data-model.md` §1.3) and referenced elsewhere

## Coverage of the 100 edge cases

- [ ] CHK026 Every EC ID has a verification cell containing either a test path or a named disposition
- [ ] CHK027 The 27/20/10/15/10/10/8 distribution is stated as it is (not as first planned), with the reason for the two additions to group A
- [ ] CHK028 Every `UNSUPPORTED`-disposition row is exercised by the fuzz pass
- [ ] CHK029 Every `LIMITATIONS` row appears in the generated limitations document with a cross-reference
- [ ] CHK030 No EC row's required behaviour is "undefined" or "TBD"
- [ ] CHK031 The edge cases cover the failure classes the audit found (parser totality, loop-carried operands, tf32, gameable coverage, schema with no alternatives)

## Constitution compliance

- [ ] CHK032 Principle I: every quantitative claim in every artifact has a reproduction command or file+line
- [ ] CHK033 Principle II: no path in the spec drops or approximates an operation without a marker
- [ ] CHK034 Principle III: the second ISA is a scheduled block (not future work) and its measurement is required
- [ ] CHK035 Principle IV: the seam is the artifact of record and is proven before the pipeline is wired to it
- [ ] CHK036 Principle V: every explicit non-goal is recorded with the full-generality reference
- [ ] CHK037 The Complexity Tracking table names the three justified violations and the simpler alternative rejected for each

## Evidence integrity (the specific failure class of v1)

- [ ] CHK038 No section number, line number, or file path appears anywhere in the spec set that was not opened and verified
- [ ] CHK039 The citations that v1 got wrong (TensorLift §9, "Pipeline cost" §4.2, ACT's addressing phase) are correct or absent
- [ ] CHK040 The triton-shared reference describes the archived repository and its real limits (documented incompleteness, two parallel pointer analyses) rather than "crashes" or "monolithic"
- [ ] CHK041 The MLIR-bindings decision cites the traversal-API reason, not the false build-cost reason
- [ ] CHK042 The alpha/beta attribute contradiction (extracted vs pre-fixed) is resolved in one direction and stated once
- [ ] CHK043 "Block shape" and "problem shape" are distinguished wherever both could be meant
- [ ] CHK044 No orphan citations: every work named is one of the four sources or is cited with a locator
- [ ] CHK045 The `TensorReduce` reference is either cited or removed, not carried over from v1

## Notes

- Add findings inline, with the requirement ID they affect.
- A checked item means the requirement is well-formed; `tasks.md` and the tests track whether it is met.
- Items CHK038–CHK045 exist because every one of them was a real defect in the previous revision. If a future
  revision re-introduces one, the checklist is where it should be caught.
