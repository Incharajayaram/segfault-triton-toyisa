# Unified testing, benchmarking and CI protocol

Normative. If this document and a person's habit disagree, this document wins.

The problem it solves: four people building four tracks will silently diverge on
*what a correct answer is*. Then a merge looks fine, the demo runs, and the
numbers in the report do not reproduce. Every rule below exists to make one of
those divergences impossible rather than unlikely.

---

## S1. The one rule

> **A test may not contain an expectation that was typed by hand.**

Every expected value comes from one of two places:

| Source | Contains | Who may edit |
|---|---|---|
| `fixtures/*.ttir` | the frozen inputs | nobody directly; regenerate via `make fixtures` |
| `fixtures/GOLDEN.json` → `observations` | machine-derived facts about those inputs | nobody; `make golden` rewrites it |
| `fixtures/GOLDEN.json` → `intent` | what each tier was *designed* to exhibit | reviewed change only |

A literal like `assert ops["tt.dot"] == 1` is allowed only when the same number is
asserted by `intent`. `assert len(splats) == 13` is banned — that is an
observation, and it belongs in `GOLDEN.json`.

## S2. Why: the three divergences this prevents

1. **Byte-divergence.** `ttir` embeds the absolute path of the file that produced
   it: `#loc = loc("/home/dev/proj/verify/dump.py":31:0)`. Two people generating
   the same kernel produced **different bytes**, so every hash, diff and
   golden-file test failed for whoever did not generate it first. Verified on the
   real corpus before normalisation; `--check` now fails if the path returns.
2. **Number-divergence.** One person asserts 13 splats, another asserts 14, both
   from memory, both pass locally. Fixed by deriving the count once.
3. **Version-divergence.** The fixtures are an artifact of Triton **3.7.1**.
   A Triton upgrade changes the IR, so re-deriving observations would hide a real
   corpus change behind a green run. `snapshot_fixtures.py` refuses to re-derive
   while `provenance.triton` differs, and the CI job that touches Triton is
   `workflow_dispatch` only.

## S3. The canon

```
fixtures/
├── raw/                 as dumped, unnormalised, absolute paths and all (provenance)
├── *.ttir               normalized: loc("<abs path>":L:C) -> loc("LOCFILE":L:C)
├── GOLDEN.json          schema, provenance, observations, intent
└── VERSIONS.txt         python / triton / torch version + the regeneration command
```

`GOLDEN.json` is the only file in the repository a person must not hand-edit in
the `observations` block. Observed facts about the corpus as shipped:

| tier | lines | ops | op vocab | dialects | multi-result | region depth | `tt.dot` | `scf.for` | design |
|---|---|---|---|---|---|---|---|---|---|
| `t0_vecadd` | 51 | 20 | 13 | arith 5, tt 15 | 0 | 2 | 0 | 0 | true-negative control |
| `t1_matmul` | 160 | 63 | 17 | arith 23, scf 2, tt 38 | 1 | 3 | 1 | 1 | primary; loop-carried pointers |
| `t2_matmul_relu` | 174 | 70 | 19 | arith 25, scf 2, tt 43 | 1 | 3 | 1 | 1 | epilogue after the accumulator |
| `t3_modulo` | 67 | 29 | 13 | arith 9, tt 20 | 0 | 2 | 0 | 0 | unstructured; must not resolve |

Also recorded per fixture: SHA-256, `loc_defs`/`loc_uses`, `loc_names`
(`a_ptrs`, `b_ptrs`, `c_ptrs`, `acc`, `M`, `N`, `K`, …), `has_tf32`,
`max_region_depth`, `ops_total`, full op histogram. `t1_matmul` carries 93 `#loc`
definitions and 64 uses; `t3_modulo` carries `arith.remsi`.

Commands:

```bash
make golden         # re-derive observations (never touches intent)
make golden-check   # CI gate: fixtures and GOLDEN.json agree; names the drift
```

## S4. The test taxonomy

Five layers. The layer is fixed by the **directory**, and the marker is applied
automatically by `tests/conftest.py` — you never write `@pytest.mark.unit`, so a
test cannot be filed in the wrong layer by forgetting to annotate it.

| Layer | Directory | Marker | Runs without | Exists to answer |
|---|---|---|---|---|
| T-L1 unit | `tests/unit/` | `unit` | all upstream stages | "is this one function right?" |
| T-L2 contract | `tests/contract/` | `contract` | the pipeline | "does this stage honour its documented interface?" |
| T-L3 integration | `tests/integration/` | `integration` | Triton, torch, GPU | "do two real stages compose?" |
| T-L4 e2e | `tests/e2e/` | `e2e` | Triton, GPU | "is the corpus fully lowered and numerically right?" |
| T-L5 bench | `bench/` | `bench` | — | "what is the number?" — **never gates** |

`make test` runs T-L1…T-L4. `make ci` adds `lint`, `golden-check`, `test-map`.

**Layer ownership.** T-L1 belongs to the track that owns the module. T-L2 belongs
to the track that owns the *contract*, not the stage behind it — so the test that
judges your work is written against the interface you promised, which is what
makes the interface real. T-L3/T-L4 are owned by track D.

## S5. The mapping law (R5)

Enforced by `tools/check_test_map.py`, wired as `make test-map`:

1. every `src/tritonflow/**/<module>.py` has `tests/unit/test_<module>.py`
2. every `specs/.../contracts/<name>.md` has `tests/contract/test_<name>.py`
   (`-` → `_`)
3. no module imports `triton` outside `harness/`, and no module imports `torch`
   outside `torch_backend/` — always enforced, in every mode

Law 3 is the reason the suite runs on a bare laptop and in CI without CUDA: the
pipeline is a text-to-text program, and the one place Triton is allowed is the
fixture extractor.

Mode | laws 1–2 | used by
---|---|---
report (default) | warnings | day 1–2, before modules exist
`--strict` | hard failure | CI job `test-map-strict`, enabled with `STRICT_REAL=1`

Today: 0 modules, 9 contracts, 8 contract tests pending, 0 violations.

## S6. The two input classes

Every T-L2 contract test is parametrised over both:

```python
def test_something(input_class):            # input_class is "handbuilt" | "real"
```

* **`handbuilt`** — the test builds its input as a typed literal. It runs from
  hour 0, before any upstream stage exists. This is what lets four people work in
  parallel: nobody is blocked on anybody.
* **`real`** — the test consumes the output of the real upstream stage. Until
  that stage lands it is an `xfail`, which is a *dated* promise, not an excuse.

The deadline mechanism is explicit and binary:

```python
# tests/conftest.py
def real_pipeline_available() -> tuple[bool, str]:
    from tritonflow.ttir.parser import parse_raw
    from tritonflow.ttir.to_ir import build_ir
```

* unavailable + `STRICT_REAL=0` → `xfail`, reason printed in the run summary
* unavailable + `--strict-real` → **hard failure**, message naming the deadline

**Flipping the switch** is one line in `.github/workflows/ci.yml`, `env.STRICT_REAL`,
at the end of day 3. That single edit converts every `real` row in the suite from
a promise to a gate. Today: 0 real rows, so the flip is free and the suite is
green at hour 0.

## S7. Determinism rules

Non-negotiable, because a flaky number is worse than a missing one.

| Rule | Why |
|---|---|
| No GPU in any test or bench. `-m gpu` tests exist and never run in CI | register allocation on a toy device is emulated, not accelerated |
| No network, no clock, no timezone, no hostname in an assertion | |
| Every random input comes from `np.random.default_rng(SEED)`, `SEED = 24173` in `bench/cases.yaml` and `bench/adapter.py` | one seed, two places, checked by the runner |
| No timing assertion in a test. Timings live only in T-L5 | a timing assertion is a flake with a schedule |
| Fixtures have LF endings, one trailing newline, no trailing whitespace | CRLF makes every hash platform-dependent |
| `loc(...)` source paths normalised to `LOCFILE` | S2.1 |
| Fixture regeneration is manual, never a push | S2.3 |

## S8. How to run

```bash
make bootstrap       # pip install -e '.[dev]'  — no Triton, no torch, no GPU
make ci              # lint + golden-check + test-map + tests  (what CI runs)
```

| Target | Use |
|---|---|
| `make test-unit` | while writing one module |
| `make test-contract` | before declaring a stage done |
| `make test-integration` | track D, and anyone merging |
| `make test-e2e` | before the demo |
| `make test-real` | `--strict-real`: check your promise is kept |
| `make golden` | after changing `fixtures/raw/` |
| `make golden-check` | before pushing anything that touches `fixtures/` |
| `make lint` | ruff, same config as CI |
| `make bench` | writes `bench/results.json`; never blocks a merge |

**The daily loop, four commands:**

```bash
make test-unit && make test-contract && make lint && make golden-check
```

Anything else is optional. If a check is not in that list and not in `make ci`, it
is not enforced, so do not rely on it.

**Day 1, first command per track** — all four work before the identity of any
upstream module is known:

```bash
# A (parser)      make test-contract   # negative corpus + descriptor table
# B (IR/emit)     make test-unit       # build_ir against a literal RawModule
# C (recognition) make test-unit       # recognise() against a hand-built Module
# D (seam, ISA)   make test-unit       # selector against hand-built descriptors
```

## S9. PR definition of done

Tick all seven. A PR is not ready because the tests pass.

- [ ] `make ci` is green locally, not just in CI
- [ ] new module → `tests/unit/test_<module>.py` exists (R5 law 1)
- [ ] touched a contract → the contract's test file is updated in the *same* PR
- [ ] any new expectation lives in `GOLDEN.json`, not in the test body (S1)
- [ ] contract tests are parametrised over `input_class` (S6)
- [ ] no new import of `triton`/`torch` outside its allowed package (R5 law 3)
- [ ] if `fixtures/` changed: `fixtures/VERSIONS.txt` updated and `GOLDEN.json` diff reviewed by a second person

## S10. CI

`.github/workflows/ci.yml` — every job is a `make` target, so a red CI run is
reproducible with one local command.

| Job | Blocks merge | Runs | Notes |
|---|---|---|---|
| `lint` | yes | ruff | same config as `make lint` |
| `canon` | yes | `make golden-check` | stdlib only, no pip install |
| `test-map` | yes | report mode | laws 1–2 warn until day 3 |
| `test` | yes | `pytest` on py 3.12 / 3.13 / 3.14 | no Triton, no torch, no GPU |
| `test` (strict) | yes, after day 3 | `pytest --strict-real` | gated on `STRICT_REAL` |
| `test-map-strict` | yes, after day 3 | laws 1–2 hard | gated on `STRICT_REAL` |
| `bench` | **no** | `make bench` | `continue-on-error`; uploads `bench/results.json`; table goes to the run summary |
| `fixtures-refresh` | n/a | `workflow_dispatch` only | installs Triton, uploads new fixtures as an artifact, never opens a PR |

`bench` deliberately does not gate. A performance number that can block a merge
gets deleted by whoever is blocked, and then there is no number.

## S11. Benchmark protocol

`bench/cases.yaml` is the case matrix: 4 tiers × 8 metrics = **32 rows**. The
runner writes all 32 every time. A row it cannot compute is recorded as
`unavailable` with a reason — never omitted — so the published table cannot
quietly lose a metric.

| Metric | Unit | Headline | Formula |
|---|---|---|---|
| `fully_lowered` | bool | **yes** | every value-producing op is annotated or provably elided |
| `node_fraction` | ratio | no, **upper bound** | annotated value ops / all value ops — gameable, never the headline |
| `largest_subgraph` | ratio | yes | largest connected annotated subgraph / dataflow-relevant graph |
| `cost_ratio_vs_reference` | ratio | yes | selected total cost / hand-written reference cost |
| `cost_ratio_vs_oracle` | ratio | yes | selected cost / min over exhaustively enumerated valid lowerings |
| `instruction_count` | count | no | emitted instructions + raw `ttir` op count, informational |
| `generation_ns` | ns | yes | median of 5 `perf_counter_ns`, 1 warm-up discarded |
| `parity_max_rel_err` | ratio | yes | max relative error vs eager, under the derived tolerance |

Rules:

1. **Report the boolean before the fraction.** A high `node_fraction` with
   `fully_lowered = false` is a failure, and the ordering makes that legible.
2. **Never merge rows across tiers or across ISAs.** Coverage and selection
   quality are per tier *and* per ISA; a merged figure hides exactly what the
   experiment exists to expose.
3. **Timings never cross machines.** Within-run comparisons only. No "we are
   2× faster than X" claim from wall-clock on a laptop.
4. **Cost ratios, not instruction counts**, are the selection-quality claim —
   instruction count would flag a legitimate alternative lowering as a failure.
5. **`bench/results.json` is the artifact; prose is generated from it.** The
   table is rendered by `bench/report_table.py` and pasted into the report.
6. **A negative or missing number is reported, not removed.** `unavailable` is
   an honest state; a deleted row is not.

`bench/results.json` schema (validated by the runner, consumed by T-L4 and the
report):

```json
{
  "schema": 1,
  "provenance": { "utc": "...", "python": "...", "platform": "...", "seed": 24173,
                  "gpu": false, "corpus_hashes": { "t1_matmul": "c0ce8ee37cdd..." } },
  "protocol":   { "seed": 24173, "repeats": 5, "warmup": 1,
                  "clock": "perf_counter_ns", "statistic": "median",
                  "config_sha256": "..." },
  "results":    [ { "id": "t1_matmul.fully_lowered", "tier": "t1_matmul",
                    "metric": "fully_lowered", "unit": "bool", "headline": true,
                    "status": "ok|unavailable|error", "value": null, "reason": null } ]
}
```

The runner refuses to start if `(seed, repeats, warmup) != (24173, 5, 1)` or if
`allow_gpu` is true. That is not paranoia: it is the only thing standing between
two people's numbers being comparable.

## S12. Tolerance policy

Derived, never chosen. `t1_matmul`/`t2_matmul_relu` carry
`inputPrecision = tf32` as a literal attribute, so a float32 emulator cannot
match torch/Triton numerics unless the truncation is reproduced.

| Path | Comparison | Tolerance |
|---|---|---|
| integer (`arith.remsi`, indices, `make_range`) | exact | 0 |
| fp32, no `tf32` in the kernel | within tolerance | `atol=1e-6, rtol=1e-5` |
| fp32 with `inputPrecision = tf32` | within tolerance | `atol=1e-3, rtol=1e-2`, derived via `precision.derive_tolerance` |
| low precision emitted by the ISA | within tolerance | per-dtype, recorded with its derivation |

Both the value **and** the derivation string are recorded next to the result, so
a residual mismatch is attributable to precision rather than to reassociation of
a non-associative reduction. The reduction order is the order the ISA schema
declares (`precision.accumulate(values, order)`).

## S13. When something is red

| Symptom | Meaning | Action |
|---|---|---|
| `golden-check` fails on `sha256` | someone edited a fixture by hand or regenerated on another Triton | restore, or regenerate properly and get the `GOLDEN.json` diff reviewed |
| `golden-check` fails on `ops.*` only | the fixture text is right and the observation is stale | `make golden`, then read the diff |
| `intent/... expect_dot` mismatch | the corpus no longer means what it claims | the fixture is wrong, not the intent |
| `test-map` strict fails | a module landed without its test | write the test |
| `--strict-real` fails | the deadline passed and the upstream stage is still missing | land the stage, or move the deadline in a reviewed commit — never silently by reverting `STRICT_REAL` |
| `bench` row is `unavailable` | not a failure | it is a fact; quote it as one |

Ownership of `fixtures/GOLDEN.json`: **nobody owns it.** It is regenerated by a
tool. The only person who may commit a change to its `intent` block is someone
who can point at the fixture bytes that justify it.
