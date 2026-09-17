# Changes to `triton-generator-merge`

Audit and repair pass. Every item below was reproduced before it was changed and
re-verified after. Two findings that could not be closed are recorded in
`KNOWN_GAPS.md` rather than worked around.

## The headline defect: `lower.py` was not a compiler

`src/triton_toyisa/lower.py` — the module `cli.py`, `bench/adapter.py`,
`verify/verify_end_to_end.py` and the contract tests all depend on — was a lookup
table keyed on the fixture's *filename*:

```python
if tier == "t0_vecadd":
    instrs = [Instr(name="LDG" if isa_name == "vortex_rvgpu" else "DMA1D",
                    cost=128.0, ...)]
```

Every instruction, operand and cost was a literal. "Cross-ISA transfer" was a
ternary on a string. The recogniser, the schema and the selector were imported
nowhere in the file, despite its docstring naming all three.

Consequences, all reproduced:

* **`t3_modulo` reported "Successfully compiled" on every ISA.** It is the
  corpus's negative control — its modulo wraparound makes the memory access
  unstructured and it must be refused. `unsupported=[]` was hardcoded. A silent
  miscompile of the exact kind the project's own first principle forbids.
* **`t1`/`t2` crashed with an uncaught traceback** on toyisa1 and toyisa2: the
  table emitted a `DMA1D` carrying no memory operand, which the emulator
  correctly refused at run time.
* Identical costs (435.2, 448.0) across three supposedly different ISAs.

`lower.py` is now a thin driver over the production chain — `parse_module →
build_def_use → annotate → assemble` — with the real schema and the real
selector (`selector` is deliberately not passed, so `assemble`'s own
`_default_selector → isa.select.select` runs rather than a stand-in that is
admissible by construction). The public surface (`make_inputs`,
`compute_reference`, `lower_fixture`, `RunContext`) is unchanged.

## Schema defects in `vortex_rvgpu.yaml`

Three instructions were declared in a way that made them win selections they
should never have been candidates for. All three are the same shape of bug: an
instruction that cannot lower an operand was declared as a candidate for it, and
undercut the correct instruction on cost.

| Instruction | Was | Effect | Now |
|---|---|---|---|
| `BARRIER` | `kind: memory`, trivially-true constraint, flat `0.01` cost | Won **every** memory selection; each load and store lowered to a fence and the program moved no data | `kind: control`, `rule: barrier` |
| `LDS` / `STS` | `rule: memory`, `0.10 * words` against LDG/STG's `0.50` | Won every global load; the emitted program read a scratchpad nothing had filled | `rule: scratch` |
| `LDG` / `STG` | identical in every field selection reads | Min-cost broke the tie by declaration order and chose `LDG` for stores; the emulator executed a store as a load | `direction: load` / `direction: store` |

`toyisa2`'s `LDG`/`LDS2D` are genuine global→scratch moves and correctly remain
`rule: memory` — the contrast is noted in the schema.

## New schema-language feature: instruction direction

Adding `direction` was the only way to fix the `LDG`/`STG` tie: renaming or
re-costing either one would just move which of the two wins.

* `Instruction.direction` (`load` | `store` | `None`) and `Instruction.serves()`.
  `None` serves both, so `toyisa1` and `toyisa2` needed no edit.
* `enumerate_candidates` records a direction mismatch as a **rejection** with a
  reason, not a pre-filter — `contracts/selector.md` postcondition 1 says nothing
  is pre-filtered, and "STG is not a load" belongs in the audit trail beside the
  cost-based rejections.
* `assemble._direction` derives it from the operation name: `tt.load` reads
  through its pointer operand, `tt.store` writes through its own, and nothing in
  the descriptor carries that. Threaded through `_call_selector`, which already
  dispatches by arity, so stand-in selectors are unaffected.
* `validate_schema` rejects any other value.

## Other fixes

* **`cli.py` imported `triton_toyisa.emit.serialize`, which does not exist** —
  `--out` crashed every time. Corrected to `emit.disasm`. The CLI was rewritten:
  a refusal is now a diagnostic with exit status 1 and named reasons, not an
  uncaught emulator traceback. Successful compiles print the parity result
  against the derived tolerance.
* **`test_end_to_end.py` asserted `ctx.unsupported == []` for `t3_modulo`** —
  encoding the miscompile as the expected behaviour. Rewritten: t3 refusal is now
  asserted on all three ISAs, along with the refusal reason and the fact that a
  refused program is never executed.
* **Zero-valued writes were discarded.** The grid merge used
  `where=value != 0`, so a program that legitimately wrote `0.0` was treated as
  not having written at all. Tier 2's ReLU clamps roughly half its output to
  exactly zero. Output buffers are now NaN-initialised so "written" is
  distinguishable from "untouched", and anything left NaN propagates into the
  error instead of being quietly zero-filled.
* **Execution failures are recorded, not raised.** `lower_text` promises never to
  raise; an emulator failure after a successful lowering is now recorded on
  `RunContext.execution_error`, kept deliberately distinct from `unsupported` so
  a pipeline defect cannot be reported as a limitation of the ISA.
* **Lint: 36 errors → 0.** `E741` fixed properly (renamed `l`); `E402` scoped
  per-file for the three scripts that put the project on `sys.path` before
  importing from it, matching the convention already in `pyproject.toml`.
* **`F821 Undefined name 'violations'`** in `schema.py`, introduced during this
  pass and caught by lint before it shipped.

## Verified state

```
              t0_vecadd    t1_matmul      t2_matmul_relu   t3_modulo
toyisa1       err 0.0      3.03e-4 OK     see G2           REFUSED
toyisa2       err 0.0      3.03e-4 OK     see G2           REFUSED
vortex_rvgpu  lowers       lowers         lowers           REFUSED
```

Tolerance is derived, never hardcoded: `K · (2·2⁻¹¹ + 2⁻²⁴)` = 0.0625 for K=64
on the tf32 path, `K · 2⁻²⁴` otherwise, read from the kernel's own declared
`inputPrecision`.

* `python3 run_tests.py` — **74 passed, 0 failed, 0 errors, 1 skipped**
  (was 63 passed, with the t3 miscompile asserted as correct)
* `python3 -m ruff check .` — **All checks passed**
* `python3 tools/snapshot_fixtures.py --check` — 4 fixtures, 182 ops, canon OK

`test_selection_differs_across_isas` fails if any two ISAs produce the same
instruction names at the same cost, which is what makes the transfer claim
falsifiable rather than asserted.

## Not done

See `KNOWN_GAPS.md`. In short: `vortex_rvgpu` lowers on every tier but cannot yet
execute (its per-operation elementwise instructions share one rule and one cost,
so everything lowers to `VADD` — the fix is an `ops` filter shaped exactly like
the `direction` one, and `Instruction.ops` is already parsed), and
`t2_matmul_relu` parity sits at 1.0575 against a 0.0625 bound, isolated to the
epilogue. Neither is asserted as passing and the tolerance was not widened to
hide the second.

No GPU was available in this environment; nothing here is GPU-verified.
