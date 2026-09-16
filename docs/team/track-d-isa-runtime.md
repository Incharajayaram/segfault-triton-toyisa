# Track D — ISA, selection, seam & emulator

**Owner**: D · **Load**: ~3.5 days · **Owns the day-1 proof and the one true convergence point**

## Mission

The largest track, and the one with the two things that make the project mean anything:

1. **The decision.** A schema with alternatives, admissibility predicates and costs; a selector that rejects with
   a reason and emits the minimum-cost admissible lowering. Without alternatives there is nothing to decide and
   the word "generate" is unearned.
2. **The seam.** A real `torch.compile` device — `register_backend` plus a `DeviceInterface` implementation —
   proven on day 1 with a hard-coded lowering. Not day 6: day 1, before anything else is wired.

## You own

```
src/triton_toyisa/isa/schema.py             IsaSchema, Instruction, load_schema, validate_schema, evaluate, cost_of
src/triton_toyisa/isa/select.py             Candidate, SelectionReport, enumerate_candidates, select, oracle_min
src/triton_toyisa/isa/rules/toyisa1.py      ISA_RULES
src/triton_toyisa/isa/rules/toyisa2.py      ISA_RULES (authored independently — see day 4)
src/triton_toyisa/isa/schemas/*.yaml
src/triton_toyisa/torch_backend/compiler.py           toyisa_backend, extract_ttir, lower_and_run, fallback
src/triton_toyisa/torch_backend/device_interface.py   ToyIsaInterface, register_interface
src/triton_toyisa/torch_backend/device.py             ToyDevice, DevicePtr
src/triton_toyisa/emu/exec.py                         emulate, apply, ProgramNotExecutable, ShapeMismatch
src/triton_toyisa/emu/precision.py                    PrecisionPolicy, derive_tolerance, tf32_truncate, accumulate, compare
```

**Your contracts**: `contracts/isa-schema.md`, `contracts/selector.md`, `contracts/torch-seam.md`,
`contracts/emulator.md`.
**Your tasks**: T002, T004, T009–T024, T047–T057, T060–T066.

**Note**: you **write the day-1 smoke test** (T015, inside `torch_backend/`), then **hand `tests/integration/`
to A on day 2** — the per-tier tests, the fuzz pass and the fallback tests are A's float work, run against your
contract. You keep the smoke test because it is the proof of your track.

## Day 1 — hour 0 you are unblocked; by hour 2 the project is real

You are the only track that can start at full speed with nothing from anyone, *and* the only one that can
produce a result nobody can argue with on day 1. Order:

1. **`toyisa1.yaml`** — it is written out in full in `data-model.md` §1.1 (DMA1D, DMA2D, MAC8, MAC16, EPI, with
   constraints and costs). Then the predicate language, `validate_schema`, `evaluate` (fail-closed),
   `cost_of` (zero-dimension guard), `select`/`enumerate_candidates`.
2. **Test the selector with hand-built descriptors.** A descriptor is a plain dataclass; C freezes it at lunch.
   You can prove a 64×64×32 tile picks `MAC16` over `MAC8` before the parser exists.
3. **The smoke test.** `register_backend` + `ToyIsaInterface` + `ToyDevice` + a hard-coded 64×64 matmul
   lowering. Template in-tree:
   `test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/compiler.py`. ~10 lines of
   registration against a documented API.
4. **`emu/precision.py`** — the tolerance derivation. Small, and it makes every later numerical claim honest.

## Day 2 — emulator, then the real pipeline

`emulate`/`apply` over B's `emit/ir.py` (fields frozen day 1): DMA → array slicing with the resolved descriptor;
MAC → `matmul`/`tensordot` over the tile; EPI → the ufunc. **`UNSUPPORTED` raises `ProgramNotExecutable`**, and
the seam routes that kernel to eager fallback.

Then wire `lower_and_run`: pipeline → emulator → `torch.Tensor`. This is the convergence point — it waits for
A, B and C, and it is 3 hours of work, not a track.

## Day 3 — integration and selection quality

Per-tier end-to-end, `FallbackRecord`s surfacing in coverage, `oracle_min` and the gap, the selection-quality
table. The fallback test is the one that matters most: an unlowerable op must run through eager and produce a
**correct** result, never a wrong number.

## Day 4 — ISA-2 and the reports

- `toyisa2.yaml`: scratchpad + accumulator banks, an outer-product/conv unit, strided 2-D DMA with an explicit
  offset pair, a `CLAMP` epilogue, and a **different** accumulation order (`k_blocked(4)`). The table is in
  `data-model.md` §1.3.
- `isa/rules/toyisa2.py`: **author it without opening `isa/rules/toyisa1.py`.** That is not style — it is the
  validity condition of the experiment. Put it in the commit message. Then hand B every edit you were forced
  to make outside `isa/schemas/` and `isa/rules/`; that list, not the percentage, is the result.
- Coverage and transfer tables (C and B own those files), then the write-up reads them. No number typed by hand.

## Definition of done

- [ ] Day-1 smoke test green: a real `torch` op compiles and runs on the registered device
- [ ] Device visible: `device_count`, `is_available`, `current_device`/`set_device` round-trip, `.to("toyisa")`
- [ ] `toyisa1.yaml` validates; six broken schemas each produce their specific violation
- [ ] Both DMA admissible → `DMA2D` chosen with `DMA1D`'s rejection reason recorded; stride not divisible by 4 →
      `DMA1D`; `m % 16 != 0` → `MAC8`; 64×64×32 → `MAC16`
- [ ] Nothing admissible → `no_admissible_lowering` → `UNSUPPORTED`, **never a default instruction**
- [ ] Greedy-vs-oracle gap reported when nonzero
- [ ] Parity recorded with tolerance **and derivation**; integer paths exact; `tf32` truncation implemented
- [ ] `ProgramNotExecutable` on any program containing `UNSUPPORTED`
- [ ] Unsupported op → eager fallback, correct result, `FallbackRecord` present
- [ ] Unimplemented `DeviceInterface` slots raise with a reason, and each reason lands in the limitations document
- [ ] ISA-2 schema + rules authored independently; the forced-edit list handed to B

## Gotchas

- **A one-instruction-per-kind ISA is a schema error, not a schema.** `validate_schema` rejects it. TAIDL's API
  takes `cost`, `update`, `constraints`; we keep cost and a narrow `constraints` and drop `update`, and we say so.
- **Accumulator semantics are schema fields**: `precision`, `reduce_dim`, `order`. Float addition is not
  associative, so without a declared order any tolerance mismatch can be blamed on reassociation and the
  tolerance becomes unfalsifiable.
- **The emulator must execute the emitted program**, not recompute from the source module. Otherwise we validate
  a shortcut and prove nothing. Role model: `libopenreg.so`, a CPU DSO emulating a CUDA-like device.
- **`tf32` is a literal in the corpus IR.** Truncate multiply inputs to tf32 mantissa width and reduce in the
  declared order. A float32 emulator calling its result "within tolerance" is hiding a structural difference.
- **A tolerance with no derivation is a defect**, same class as a crash.
- **Do not pre-empt the ISA-2 result.** A transfer rate below 100% is the *expected* outcome and the edit list is
  the deliverable. Reporting a suspiciously perfect number is worse than reporting 0.7.

## If you are blocked

Day 1: nothing. Day 2: C's descriptor fields (frozen lunch day 1) and B's `Program` fields (frozen lunch day 1).
Day 3: the pipeline — A, B and C all owe you something, which is why you are the escalation owner for the
integration slice: post which of the three is missing and what command fails. Day 4: nothing.
