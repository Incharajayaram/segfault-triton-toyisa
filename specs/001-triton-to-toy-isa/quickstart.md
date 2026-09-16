# Quickstart: Triton-IR-to-Declarative-Toy-ISA

Five commands, in order. Commands 3–5 need neither Triton nor PyTorch: they run on the frozen fixtures.
Command 5 needs PyTorch and demonstrates the claim.

## 0. Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'        # runtime: pyyaml, numpy;  dev: pytest, triton==3.7.1, torch==2.12.1
```

## 1. (Once) Regenerate the fixtures — needs Triton, needs no GPU

```bash
python -m triton_toyisa.cli extract --out fixtures --force
```

Writes `fixtures/t{0,1,2,3}_*.ttir`, `fixtures/fuzz.ttir`, `fixtures/ttgir_snapshot.txt` and
`fixtures/VERSIONS.txt`. This is the only step that touches a live Triton install; it refuses to overwrite
without `--force`, because regenerating fixtures is a reviewed act (FR-034).

Sanity check that the gate is real (no driver, no GPU):

```bash
python -m triton_toyisa.cli extract --check-gate      # prints: GPU-free extraction OK; keys [cubin,llir,ptx,source,ttgir,ttir]
```

## 2. Compile a frozen kernel to a toy-ISA program

```bash
python -m triton_toyisa.cli compile fixtures/t1_matmul.ttir \
    --schema src/triton_toyisa/isa/schemas/toyisa1.yaml \
    --out /tmp/t1.prog --selection-out /tmp/t1.selection.json
```

Prints the program:

```
; toyisa1  schema_version=1  kernel=matmul  cost=18432.0
LOOP k: 0..32 step 1 iter(%a_ptrs, %b_ptrs, %acc)
  DMA2D  dst=smem_a  src=a_ptrs  sizes=[64,32] strides=[32,1]
  DMA2D  dst=smem_b  src=b_ptrs  sizes=[32,64] strides=[64,1]
  MAC16  acc=acc     a=smem_a    b=smem_b    k=32
  ADV    reg=a_ptrs  by=32
LOOPEND
```

and the selection report, which is the evidence that the generator *chose*:

```json
{"chosen_cost": 18432.0, "oracle_min_cost": 18432.0, "gap": 0.0,
 "choices": [{"operand": "a_ptrs", "chosen": "DMA2D",
              "rejected": [{"name": "DMA1D", "by": "cost(1.00*words) > 0.60*words"}]}]}
```

Tier 3 is expected to differ, and that is the point:

```bash
python -m triton_toyisa.cli compile fixtures/t3_modulo.ttir --schema .../toyisa1.yaml --out /tmp/t3.prog
# UNSUPPORTED(tt.load) loc("x_ptr") reason="modulo wraparound"   -- a marker, not a crash
```

## 3. Execute the program on the device emulator

```bash
python -m triton_toyisa.cli emulate /tmp/t1.prog --inputs tests/data/t1_inputs.npz --out /tmp/t1.npz \
    --parity-out /tmp/t1.parity.json
```

`--parity-out` records the comparison, including the tolerance **and its derivation**:

```json
{"input_precision": "tf32", "accumulation_order": "k_major_sequential",
 "tolerance": 7.8e-04, "derivation": "tf32 mantissa truncation (10 explicit bits) over k=32",
 "max_abs_error": 5.1e-05, "verdict": "within tolerance"}
```

## 4. Generate the reports

```bash
python -m triton_toyisa.cli report  --corpus fixtures --schema .../toyisa1.yaml --out reports/coverage.md
python -m triton_toyisa.cli transfer --isa1 .../toyisa1.yaml --isa2 .../toyisa2.yaml \
    --corpus fixtures --out reports/transfer.md
```

The coverage table leads with the boolean, per ISA and per tier:

```
| ISA      | Tier      | fully lowered | largest subgraph | annotated (<=) | unsupported        |
|----------|-----------|---------------|------------------|----------------|--------------------|
| toyisa1  | t0_vecadd | yes           | 1.00             | 1.00           | –                  |
| toyisa1  | t1_matmul | yes           | 1.00             | 1.00           | –                  |
| toyisa1  | t3_modulo | NO            | 0.31             | 0.72           | 1 (loc("x_ptr"))   |
```

The transfer table leads with the per-stage edit list, not the rate:

```
| stage       | transferred | edit                                        |
|-------------|-------------|---------------------------------------------|
| parser      | yes         | –                                           |
| recogniser  | no          | isa/rules/toyisa2.py:88 (stride-1 assumed)  |
| assembler   | yes         | –                                           |
edits outside schema and rules: ["recognize/walk.py:112"]   transfer rate: 0.86
```

## 5. The claim: a real PyTorch op on the generated device

```bash
python -m triton_toyisa.cli serve &        # registers the backend and the DeviceInterface
python - <<'PY'
import torch, triton_toyisa.torch_backend  # side-effect: registers "toyisa"
x = torch.randn(64, 64); y = torch.randn(64, 64)
f = torch.compile(lambda a, b: torch.relu(a @ b))
out = f(x, y)
ref = torch.relu(x @ y)
print("device:", out.device, "max abs err:", (out - ref).abs().max().item())
PY
```

Expected: the op compiles through the registered backend, runs on the emulated toy-ISA device, and matches
within the tolerance printed in step 3.

## What "done" looks like

```bash
pytest -q                     # unit + contract + integration
pytest -q -m transfer         # the second-ISA experiment
```

- Every tier's expected outcome appears in `reports/coverage.md` (SC-002).
- Both true negatives pass: no MAC on `t0_vecadd` or `t3_modulo` (SC-003).
- Every EC ID in `edge-cases.md` resolves to a test or a disposition (SC-009).
- The transfer report exists, with numbers that may not be flattering (SC-006).

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `PARSE_UNSUPPORTED` on a fresh fixture | Triton version moved; the fixture was generated by a different release | check `fixtures/VERSIONS.txt`; regenerate deliberately with `--force` and record the new hash |
| `ProgramNotExecutable` | the program contains an `UNSUPPORTED` marker | expected for unstructured input; the seam routes that kernel to eager fallback |
| Parity failure with a small error | `input_precision` not applied | confirm the program's `input_precision` matches `PrecisionPolicy`; the derivation is in the parity JSON |
| Coverage percentage looks high but `fully lowered` is `NO` | that is the corrected metric working as designed | read the unsupported inventory, not the fraction |
