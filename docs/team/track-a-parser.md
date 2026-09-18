# Track A — Parser (text → `RawModule`)

**Owner**: A (parsing specialist) · **Load**: ~2.5 days, then the float · **Critical path: the front of it**

## Mission

Turn `ttir` text into a **strictly syntactic** structure, and make the failure route total: for *any* input,
either a `RawModule` or a diagnostic with an accurate line and column. Never an exception, never a hang.

The word to hold on to is **syntactic**. You deal in tokens and text. You do not decide what a type means, you
do not resolve an SSA name, you do not know that `scf.yield` is a terminator. `RawOp.result_types` holds
`"tensor<64x64xf32>"` as a **string**. That restraint is the entire reason you and B can work at the same time
without touching the same file.

## You own

```
src/tritonflow/ttir/lexer.py                    tokenize, strip_comments
src/tritonflow/ttir/parser.py                   parse_raw -> RawModule
src/tritonflow/harness/extract_fixtures.py      extract, TIER_KERNELS
fixtures/*                                         (generated once, then immutable)
tests/unit/test_parser_syntax_negative.py          your half of the negative corpus
```

**Your contract**: `specs/001-triton-to-tritonflow/contracts/raw-module.md` — the producer half. It is the
definition of done, item by item.
**Your tasks**: T001, T003, T005, T006, T007, T034, T036 (syntax half), T037 (syntax half), T073, T075.

## Day 1 — the seam, then the happy path

1. **Commit the `RawModule` dataclasses before lunch.** `RawModule`, `RawOp`, `RawRegion`, `RawBlock`,
   `RawLoc` — field definitions only, no logic. B writes `build_ir` against them from hour one. This is item 1
   on the project's freeze list and the highest-leverage hour you will spend.
2. **Fixtures, first thing.** You own `harness/extract_fixtures.py` and `fixtures/` because you need them more
   than anyone:
   ```
   python -m tritonflow.cli extract --out fixtures --force
   ```
   Expect 160 lines of `ttir` for matmul, 51 for vector-add, 67 for modulo, no GPU, no driver. Post
   "fixtures ready". 8 of the 27 lines of `inputPrecision = tf32` correctness in this project start here.
3. **Happy path, then hardening.** Land "parses the four fixtures" today or first thing tomorrow. Exotic
   syntax, `dense<…>`, 200-deep nesting and adversarial input come **after** the happy path — B needs a module
   from a clean fixture, not a parser that survives fuzzing.

## Day 2 — syntax negatives and hardening

`tests/unit/test_parser_syntax_negative.py` — your half of the corpus, each asserting an accurate line/column
and no exception: unterminated attribute dict (EC-011 vs EC-025), unknown token in a grammar position, truncated
module (EC-025), 200-deep nesting (EC-027), CRLF/BOM handling (EC-019, EC-020), minified single-line module
(EC-021), comments-only (EC-002), empty input (EC-001). The *semantic* invalidity cases (EC-005 arity mismatch,
EC-026 duplicate SSA name, bad type text) belong to B's file — don't take them, and don't fix them if you see
them fail; the `layer` field on the diagnostic says whose they are.

## Day 3–4 — the float role

The parser is ~1.2 days, which makes you the **float** from mid-day 2. Reinforce, but only by adding tests and
debugging — never by editing C's or D's files:

| If this is behind | Do this |
|---|---|
| C's recognition | write true-negative tests against Tier 0 and Tier 3, and a descriptor table for the hand-built cases |
| C's idioms | build the adversarial fuzz kernel and run the "marker, never a crash" pass over all 100 EC inputs |
| D's integration | debug the end-to-end per-tier failures with the fixture in one hand and the program text in the other |
| B's emitter | do **not** touch `emit/` — write the round-trip property test (200 generated programs) and hand it over |

Then: the evidence appendix (every quantitative claim + its reproduction command) and the edge-case
traceability check (SC-009, T075).

## Definition of done

- [ ] `RawModule` dataclasses committed day 1, before lunch, and not changed without announcing
- [ ] All four fixtures parse into a `RawModule`; `diagnostics == []`; op counts match the snapshot
- [ ] Type text, attribute values and SSA names are **verbatim** — no normalisation anywhere in your code
- [ ] Every `RawOp` carries `line`/`col`; diagnostics are accurate to the line (column where cheap)
- [ ] Nested regions preserved as nested `RawRegion`s; terminators are just ops with an index recorded
- [ ] `%acc_25:3` becomes one `RawOp` with **three** entries in `results` (placeholder names allowed)
- [ ] 27 syntax-negative inputs → diagnostics, accurate positions, zero tracebacks (SC-008)
- [ ] Deterministic: same bytes → equal `RawModule`, across processes and `PYTHONHASHSEED`
- [ ] `parse_raw` never raises and never hangs on any input, including 200-deep nesting

## Gotchas

- **Text stays text.** The first time you "helpfully" parse a type string into a shape, you have created a
  second dialect and B will find out on day 3.
- **Span accuracy is not cosmetic.** `PARSE_UNSUPPORTED(<line>:<col>)` is a claim we make in the write-up; a
  hand-wavy position makes it false.
- **Deep nesting must not recurse without bound.** EC-027 is real: 200-deep regions. Either iterative descent
  or a stated limit reported as a diagnostic.
- **Fixtures are the interface, not the live compiler.** `ttir` syntax moves between Triton releases. Do not
  regenerate fixtures to make a test pass; regeneration is a reviewed act that records the version hash.
- **No runtime Triton.** Only `harness/extract_fixtures.py` may import Triton (T004).

## If you are blocked

You should not be, at any point. You have nothing upstream of you except the fixtures, which you own. If B
reports that `build_ir` cannot consume your output, the seam contract is the arbiter — read it together before
either of you changes a field name.
