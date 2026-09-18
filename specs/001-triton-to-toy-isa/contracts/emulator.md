# Contract: device emulator and precision policy

**Module**: `src/triton_tritonflow/emu/` (`exec.py`, `precision.py`)
**Consumers**: the PyTorch seam, differential tests. **Requirements**: FR-021, FR-022, SC-001.

## Interface

```python
emulate(program: Program, inputs: dict[str, np.ndarray],
        policy: PrecisionPolicy) -> dict[str, np.ndarray]

apply(instr: Instr, state: MachineState, policy: PrecisionPolicy) -> None

class PrecisionPolicy:
    input_precision: Literal["tf32", "ieee"]
    accumulation_order: str                 # read from the ISA schema, not chosen here
    def tolerance_for(self, dtype: str) -> float
    derivation: str                         # human-readable, recorded in the report
```

## Preconditions

- The program deserialised and its `schema_version` matches the loaded schema.
- `policy.input_precision` equals the `input_precision` captured by idiom detection from `tt.dot`.

## Postconditions

1. **It executes the program (FR-021).** The emulator consumes the emitted instruction stream — DMA, MAC, EPI,
   `UNSUPPORTED` — via `apply`. It does not re-derive the computation from the source module. This is what
   keeps validation honest: the artifact under test is the emitted program.
2. **`UNSUPPORTED` halts locally, not globally.** A program containing an `UNSUPPORTED` marker cannot be
   executed; the emulator raises `ProgramNotExecutable(marker)` and the seam routes that kernel to the eager
   fallback. It never guesses what the unsupported op meant.
3. **Declared precision is implemented (FR-022).** If `input_precision == "tf32"`, the emulator truncates the
   multiply inputs to tf32 mantissa width *before* multiplication. Running fp32 numpy arithmetic and calling
   the result "within tolerance" is a contract violation, because the difference is structural, not noise.
4. **Declared accumulation order is used (T-5).** The reduction follows `accumulation_order` from the schema
   (`k_major_sequential`, `k_blocked(4)`, …). Reassociation is therefore *not* an unmodelled error source: if
   the emulator and the reference disagree, the cause is precision, and the tolerance is meaningful.
5. **Tolerance is derived, and the derivation is recorded (FR-022).**
   - Integer and low-precision paths: exact (`tolerance == 0`).
   - Float paths: tolerance derived from the precision difference (tf32 mantissa truncation: 10 explicit bits
     vs 23) and the reduction length, written as a formula in `derivation` and stored in the report.
   A tolerance with no derivation is a contract violation.
6. **Deterministic (FR-004).** Same inputs, same program, same output bytes across runs and processes. The
   emulator may not use unordered parallelism.
7. **Side-effect free.** `emulate` does not mutate `program` and does not write outside the storage it owns.

## Reference computation

The differential test compares against **eager PyTorch** (`torch.matmul`), not against a second NumPy
implementation. Two hand-written references would only prove they agree with each other. Where the eager path
itself uses tf32 (`torch.backends.cuda.matmul.allow_tf32`), the test states which mode it ran in and the
tolerance is derived for that mode; the mode is recorded in the report next to the number.

## Failure modes

| Case | Required behaviour |
|---|---|
| Program with an `UNSUPPORTED` marker | `ProgramNotExecutable`; seam falls back to eager |
| Input shape disagrees with the descriptor | clear `ShapeMismatch` naming operand, expected, got |
| NaN/Inf propagation | no special-casing; result compared with the same tolerance rule |
| An instruction whose operand is out of its emulated storage | error naming the pointer and the size; no silent zero-fill |
| Emulator disagrees with eager beyond tolerance | test failure, with the derivation printed alongside the observed error |

## Tests

- **Contract** (`tests/contract/test_emulator.py`): per-instruction unit tests on hand-built programs; the
  precision policy applied to a known tf32 truncation case, with the expected numeric result computed
  independently.
- **Differential** (`tests/integration/test_numerics.py`): randomised inputs over the corpus shapes and a small
  dtype matrix; eager comparison; tolerance from the policy; derivation printed on failure.
- **Determinism**: 3 runs, byte-identical outputs.
- **Round-trip**: emulate a serialised-then-deserialised program and a directly-assembled one; results equal.
