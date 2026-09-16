# bench/ — T-L5

Publishes numbers. **Never gates a merge.** The full protocol is normative in
[`../docs/team/testing-ci.md`](../docs/team/testing-ci.md) S11; this file is the
operating summary.

```bash
make bench                              # -> bench/results.json
python3 bench/report_table.py           # -> the markdown results table
```

## Files

| File | Role |
|---|---|
| `cases.yaml` | the case matrix: 4 tiers × 8 metrics = 32 rows, plus the protocol values |
| `adapter.py` | the **only** place bench touches the pipeline; owned by track D |
| `run.py` | the runner; emits every row, always |
| `report_table.py` | renders `results.json` as the table used in the report and the CI summary |
| `results.json` | generated artifact; the input to `report/` and to nothing else |

## Hard rules

1. All 32 rows are emitted every run. A metric that cannot be computed is
   `status: "unavailable"` **with a reason**, never a deleted row.
2. Protocol is fixed: `seed=24173`, `repeats=5`, `warmup=1`, `perf_counter_ns`,
   `median`, `allow_gpu=false`. `run.py` exits non-zero if any of these drift.
3. `fully_lowered` (boolean) is reported before any fraction. `node_fraction` is
   labelled an upper bound and is never the headline.
4. Rows are never merged across tiers or across ISAs.
5. Timings are comparable within a run only. No cross-machine speed claims.
6. Deterministic inputs come from `adapter.make_inputs()`, which uses the single
   declared seed. A bare `np.random.*` call in `bench/` is a bug.

## Current state

`adapter.lower_fixture()` returns `None` until the pipeline lands (US1), so all
32 rows report `unavailable` with `pipeline not implemented`. That is the correct
state before the build, and the runner still produces a valid artifact — which is
what makes the first real number a diff against a known baseline rather than a
new file.
