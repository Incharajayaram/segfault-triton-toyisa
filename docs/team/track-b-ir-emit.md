# Track B — IR model + emitter (`RawModule` → IR → program)

**Owner**: B (IR specialist) · **Load**: ~3.4 days · **Critical path: the front-middle and the tail**

## Mission

Two halves of one job, both about **structure**:

1. Turn the parser's syntactic `RawModule` into a real IR: resolve SSA names, bind multi-result operations,
   parse type text, nest regions, bind `loc`s, and build the def-use graph. This is where the one thing nobody
   else can do lives — **region-aware traversal**.
2. Emit the program, preserving the loop structure you recovered: an operation inside `scf.for` emits inside
   the `Loop`, and `scf.yield` becomes `iter_args` re-threading, not an instruction.

Nobody else can do either half, which is why you are on the critical path twice.

## You own

```
src/triton_toyisa/ttir/ssa.py               Module, Function, Operation, Region, Block, SsaValue, TypeExpr, Loc, Attr
src/triton_toyisa/ttir/types.py             type text -> TypeExpr (shape, dtype, address space)
src/triton_toyisa/ttir/to_ir.py             build_ir(RawModule) -> ParseResult
src/triton_toyisa/ttir/graph.py             build_def_use, walk_region, topo_within_region, iter_loops
src/triton_toyisa/canon/canonicalize.py     idempotent only
src/triton_toyisa/emit/ir.py                Instr, Loop, Program, Operand, UnsupportedMarker
src/triton_toyisa/emit/assemble.py          assemble, order_regions, emit_instr, check_constraint
src/triton_toyisa/emit/disasm.py            serialize, deserialize, disassemble
src/triton_toyisa/cli.py                    compile/report/transfer subcommands
src/triton_toyisa/report/transfer.py        the per-stage transfer table
tests/unit/test_parser_semantic_negative.py your half of the negative corpus
```

**Your contracts**: `contracts/raw-module.md` (consumer half), `contracts/assembler.md`, and
`data-model.md` §2 (the IR entity table).
**Your tasks**: T035, T037 (semantic half), T038, T039, T058, T059, T072, T074.

## Day 1 — never wait for the parser

`build_ir` consumes a `RawModule`. That is a frozen dataclass, so **you can write and test the entire IR
construction on day 1 against hand-built literals** — no parser, no fixtures needed.

1. Commit `ssa.py` and `types.py` field definitions (freeze item 2) before lunch.
2. Write `build_ir` against a hand-built `RawModule` containing an `scf.for` with `iter_args` and a `tt.dot`.
3. Write the semantic negatives: EC-005 (multi-result arity mismatch), EC-026 (duplicate SSA name), invalid
   type text, `loc` key missing from the table. These are yours, not A's — the `layer="ir"` field on the
   diagnostic says so.

Then, the moment A's parser lands, re-run the same tests with the `["handbuilt", "real"]` parametrisation from
`parallelism.md` Discipline 2: the `real` row starts as `xfail` and becomes a hard failure once it passes.

## Day 2 — graph, and the thing that is wrong in every first implementation

`build_def_use`, `walk_region`, `topo_within_region`, `iter_loops`.

- `Operation.results` is a **list**. `%acc_25:3 = scf.for …` binds three values.
- `topo_within_region` orders **within a region only**. It must never move an operation out of its region.
- `canonicalize` is idempotent and nothing more. There is no node-count reduction to achieve: measured, the
  corpus has 13 `tt.splat`, 6 `tt.broadcast`, and **0** sign-ext/trunc — the splats are required broadcast
  expansion. A canonicalisation pass justified by "redundancy" would be optimising for a problem that is not
  there.

## Day 3 — emission

`order_regions` (**region-aware**), `emit_instr` (a marker on any failure), `check_constraint` (re-validate the
chosen instruction's constraint independently — do not trust the selector transitively), `serialize` /
`deserialize` with a header carrying `isa_name`, `schema_version`, `total_cost`.

The specific wrong answer to avoid: a flat topological sort over the def-use graph. Both `tt.load`s must stay
inside `scf.for`, `scf.yield` is the block terminator, and the accumulator is loop-carried. A flat order
flattens the region hierarchy and will hoist or mis-order them.

## Day 4 — CLI and the transfer table

`cli.py` subcommands, then `report/transfer.py`: one row per pipeline stage per ISA, the edit location for each
non-transfer, and `edits_outside_schema_and_rules` as a **required** field (an empty list needs a stage-by-stage
justification, not a shrug).

## Definition of done

- [ ] `build_ir` green on hand-built input on day 1, and on A's real parser output by day 2 (no `xfail` left after day 3)
- [ ] `SsaValue`s resolve; every operand has exactly one definition; block args have `def_op is None`
- [ ] Multi-result ops bind every result; wrong arity is an `ir`-layer diagnostic
- [ ] `loc` names recovered (`loc("a_ptrs")`, `loc("pid_m")`) and kept on operations
- [ ] `walk_region` descends into nested regions; `topo_within_region` never crosses a region boundary
- [ ] Region-preservation property test: ops inside the input's `scf.for` == `Instr`s inside the output's `Loop`
- [ ] `deserialize(serialize(p)) == p` on the corpus and on 200 generated programs
- [ ] Emitted header's `total_cost` equals the recomputed cost
- [ ] Determinism: 3 processes, different `PYTHONHASHSEED`, byte-identical program text

## Gotchas

- **`scf.yield` is not an instruction.** It is how the loop re-threads its `iter_args`. Emitting it, or hoisting
  a load out of the loop, are both bugs that produce a plausible-looking wrong program.
- **Text stays text in A's layer, and only here does it become semantics.** If you find yourself lexing, the
  seam has been violated — talk to A rather than patching their file.
- **No reduction claim in `canon`.** See day 2. The test is idempotence, and the honest metric is "no change".
- **`serialize` must be byte-stable.** A version mismatch on `deserialize` is a refusal, never a reinterpretation.
- **The CLI is not a wrapper.** `compile` prints the program *and* the selection report; the report is evidence,
  not decoration.

## If you are blocked

You are not, on day 1 — that is the point of the seam. Later: C's `AccessDescriptor` (fields frozen day 1) and
D's `Program` consumption (fields frozen day 1). If a field name has to change, it is a group post and a
30-second conversation, not a local rename.
