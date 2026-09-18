# Known gaps

Things this tree does not currently do, recorded so they are findings rather
than surprises. Each one is reachable from a test or a command, so none of them
can quietly stop being true.

## G1 — `vortex_rvgpu` lowers but does not execute

**Status:** lowering verified; execution blocked on one remaining schema-language
feature. The original cause (directional memory) is **fixed**; a second, distinct
cause remains.

### Fixed: directional memory instructions

`vortex_rvgpu` splits global memory into `LDG` (load) and `STG` (store). The two
were identical in every field selection reads, so `select` — which takes the
minimum cost and breaks ties by declaration order — chose `LDG` for stores as
well as loads, and the emulator executed a store as a load.

Instructions may now declare `direction: load | store`. `Instruction.serves()`
answers whether an instruction can lower an access in a given direction,
`enumerate_candidates` records a direction mismatch as a *rejection* (so it lands
in the audit trail beside the cost-based ones rather than being pre-filtered),
and `assemble._direction` derives the direction from the operation name —
`tt.load` reads through its pointer operand, `tt.store` writes through its own,
and nothing in the descriptor carries that. An instruction that declares no
direction serves both, so `tritonflow1` and `tritonflow2` needed no edit.

Verified: `t0_vecadd` on `vortex_rvgpu` now emits `LDG` for both loads and `STG`
(roles `dst`/`value`, no `defs`) for the store.

### Remaining: elementwise instructions are named but selected only by cost

`vortex_rvgpu` names its elementwise unit per operation — `VADD`, `VMUL`, `VSUB`,
`VMOD`, `VRELU`, `VCLAMP` — but they share the `elementwise` rule and the same
`0.05 * words` cost. Selection enumerates by rule and takes the minimum, so
**every** elementwise operation lowers to `VADD`: the `tt.splat` that builds a
pointer vector, the `tt.make_range`, the `tt.addptr`, the `arith.cmpi`. The
emulator then executes each as an addition, `%x_4` comes back a scalar instead of
a 1024-element pointer vector, and the following `LDG` fails to reshape size 1
into `(1024,)`.

`tritonflow1` and `tritonflow2` avoid this by having a single catch-all elementwise
instruction (`EPI`, `VPU`) whose behaviour the emulator takes from the
instruction's `source_ops` provenance. That is the "EPI overloading" the project
documentation lists as an abstraction leak — but it is what makes those two ISAs
executable, and Vortex's more honest per-operation naming is what exposes the
missing feature.

**The fix is the same shape as the direction fix, one level up.**
`Instruction.ops` already exists and is already parsed from YAML (`ops=tuple(
entry.get("op", ()))`), so the schema can already say `ops: [arith.addf]` on
`VADD`. What is missing is the filter: `enumerate_candidates` needs to reject an
instruction whose declared `ops` do not include the source operation, exactly as
it now rejects one whose `direction` does not match, and `assemble._select` needs
to pass `op.name` through. Re-costing the elementwise instructions would only
move which one wins the tie.

Until then `lower_text` records the failure on the result as
`RunContext.execution_error` rather than raising, and keeps it distinct from
`unsupported`: a kernel the ISA cannot express and a program the emulator choked
on are different findings, and collapsing them would let a pipeline defect be
reported as a limitation of the ISA.

Reproduce:

```bash
PYTHONPATH=src python3 -c "
from tritonflow.lower import lower_fixture
c = lower_fixture('t0_vecadd', isa_name='vortex_rvgpu')
print('lowered:', c.fully_lowered, '| execution_error:', c.execution_error)"
```

## G2 — Tier 2 parity is outside its derived tolerance

**Status:** open, quantified.

`t2_matmul_relu` lowers cleanly on `tritonflow1` and `tritonflow2` and its output is
correctly non-negative (`TestReluEpilogue` asserts this), but its maximum
relative error against the fp64 reference is **1.0575** against a derived
tolerance of **0.0625**.

It is specific to the epilogue: `t1_matmul` is the same matmul without the ReLU
and lands at 3.03e-4, comfortably inside the same bound. So the reduction and the
accumulation are right and something in the post-loop elementwise stage is not.

This is deliberately **not** asserted as passing. `TestNumericalParity` covers
t0 and t1; t2 is covered for lowering and for non-negativity only. Do not widen
the tolerance to make it green — the tolerance is derived from the reduction
length and the declared precision (`_tolerance` in `lower.py`), and a number
chosen to fit the result is not a bound.

Reproduce:

```bash
PYTHONPATH=src python3 -c "
from tritonflow.lower import lower_fixture
c = lower_fixture('t2_matmul_relu', isa_name='tritonflow1')
print(f'err={c.parity_max_rel_err:.6g} tol={c.tolerance:.6g}')"
```

## G3 — the corpus is four frozen fixtures

TTIR is read from `fixtures/*.ttir`, checked in and regenerated manually via
`make fixtures` (which needs Triton). There is no live interception of an
arbitrary `torch.compile` graph in this tree. That keeps the whole suite
runnable with neither Triton nor torch installed, which is the trade being made
on purpose — but it does mean "works on any PyTorch model" is not a claim this
tree supports.

## G4 — no GPU verification was performed in this environment

Any GPU-anchored numbers quoted elsewhere in the project's documentation were
not reproduced here; no accelerator was present. Everything in this file and in
the test suite is CPU-only and reproducible offline.

## G5 — the C++ emulator backend is present but unwired

**Status:** sources shipped, never built, never called.

`src/tritonflow/emu/cpp/` (six files: `machine.cpp/.h`, `bindings.cpp`,
`precision.h`, `ir_types.h`, `errors.h`) and a top-level `CMakeLists.txt` are in
the tree. Nothing builds them and, until this merge, nothing could have used
them:

* `pyproject.toml` declares `build-backend = "setuptools.build_meta"`. A
  `pip install .` never invokes CMake, so `_emu_cpp` is never produced.
* `emu/__init__.py` did not define `HAS_CPP`, so `tests/unit/test_emu_cpp.py`
  raised `ImportError` at collection and reported as a hard ERROR rather than a
  skip.
* `emu.exec.emulate()` had no `use_cpp` parameter, so both `test_emu_cpp.py` and
  `test_mvp_integration.py` called it with a keyword it did not accept.

Fixed in this merge, up to the build itself:

* `emu/__init__.py` now defines `HAS_CPP`, guarded, as the single place anything
  asks whether the extension is available.
* `emulate(..., use_cpp=...)` routes: `None` prefers C++ when built, `False`
  forces the NumPy reference path, `True` **raises** when the extension is
  absent rather than quietly answering with NumPy's numbers under a flag that
  asked for C++ — a parity test that silently falls back proves nothing.
* `tests/unit/test_emu_cpp.py` was converted from pytest to stdlib `unittest`
  (pytest is not a dependency; `run_tests.py` is the runner). All seven cases are
  preserved and now skip cleanly instead of erroring.

**Still to do:** switch the build backend to `scikit-build-core` so
`pip install .` compiles the extension, then run the parity suite with it built.
Until then `HAS_CPP` is `False` everywhere and the C++ path is dead code — it
should not be cited as a result.
