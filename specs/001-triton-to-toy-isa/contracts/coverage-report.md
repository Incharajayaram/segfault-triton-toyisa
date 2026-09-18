# Contract: reports (coverage, selection, transfer)

**Module**: `src/triton_tritonflow/report/` (`coverage.py`, `transfer.py`)
**Consumers**: the write-up, the reviewer, the limitation document.
**Requirements**: FR-027 … FR-031, SC-002, SC-003, SC-006, SC-009.

## Interface

```python
coverage_report(runs: list[PipelineRun]) -> CoverageReport   # one run per (isa, tier)
transfer_report(runs: list[PipelineRun], edits: list[Edit]) -> TransferReport
render_markdown(report) -> str                                # prose is generated from the table
limitations_document(non_goals: list[NonGoal]) -> str
```

## Postconditions

1. **The boolean comes first (FR-027).** For every (ISA, tier): `fully_lowered` is computed and reported
   *before* any fraction, and is `True` iff every value-producing operation is annotated or provably elided.
   A kernel with one unlowered operation at the head of a dataflow chain reports `fully_lowered = false` even
   when its node fraction is high. This is the specific v1 defect being fixed.
2. **The fraction is labelled.** `annotated_node_fraction` is emitted with the label `upper bound`, in the data
   structure and in the rendered text, so it cannot be quoted as a coverage number.
3. **Unsupported inventory is actionable (FR-028).** Every marker carries the op name, the reason, and the
   originating `loc` name where available, so a reader can find the operation in the original Python source.
4. **Per ISA, per tier, never merged (FR-027).** There is no single aggregate coverage number in any output.
5. **Selection quality is reported (FR-029)** against both the exhaustive oracle and the hand-written reference
   program: chosen cost, oracle minimum, gap, and the rejected alternatives with the predicate that rejected
   each.
6. **Transfer is a first-class report (FR-030).** One row per pipeline stage per ISA, `transferred: bool`, and
   the edit location when false. `edits_outside_schema_and_rules` is a required field and is expected to be
   non-empty on a first attempt; an empty list without a stage-by-stage justification is treated as a
   measurement error, not a success.
7. **Reports are generated from data, never written by hand (Constitution, workflow rule 5).** `render_markdown`
   consumes the structures; if a number appears in prose it came from a report.
8. **Limitations are cross-referenced (FR-031).** Every explicit non-goal appears with the source work that
   solves it at full generality, or an explicit `no known implementation` — never a bare "future work".
9. **Generation cost is a separate axis (SC-007).** Wall-clock per kernel is reported as a one-time
   per-kernel-shape cost, following TensorLift §4.1's framing, and never as per-compilation runtime.

## Rendered shape (fixed column order)

```
| ISA      | Tier    | fully lowered | largest subgraph | annotated (≤) | unsupported | latency ms |
|----------|---------|---------------|------------------|---------------|-------------|------------|
| tritonflow1  | t0_vecadd | yes         | 1.00             | 1.00          | –           | 12.4       |
| tritonflow1  | t3_modulo | NO          | 0.31             | 0.72          | 1 (loc("x_ptr")) | 19.8  |
```

## Failure modes

| Case | Required behaviour |
|---|---|
| A tier with no runs recorded | row absent **and** a warning; no implicit zero |
| A stage missing from the transfer report | `transfer_report` raises naming the stage; the report is never silently partial |
| A non-goal with no cross-reference | `limitations_document` emits `no known implementation`, prompting a review decision |
| Prose contains a number absent from the report | caught by the write-up checklist (`checklists/requirements.md`) |

## Tests

- **Contract** (`tests/contract/test_reports.py`): schema of every report; the "boolean first" ordering; the
  `upper bound` label present; the gameable-case fixture reporting `fully_lowered = false` with a high
  fraction.
- **Golden**: rendered markdown for the corpus matches a checked-in snapshot (so a change in metrics is
  visible in review, not silent).
- **Completeness**: the staged transfer report contains exactly the pipeline stage list from `plan.md`.
