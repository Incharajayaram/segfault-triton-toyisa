# The Triton-ToyISA Retargetable Compiler Infrastructure

**A Schema-Driven, Fail-Closed Retargetable Compiler for Custom AI Accelerators and RISC-V GPGPUs**

---

## 1. Overview & Architectural Motivation

Traditional compiler backends (e.g., target-specific LLVM passes, vendor codegen pipelines) require **months of bespoke engineering** for each new chip architecture. As hardware teams iterate on systolic arrays, asynchronous DMA engines, and specialized register files, software toolchains continually lag behind silicon.

**Triton-ToyISA** solves this bottleneck through **declarative, schema-driven compilation**:
* **The YAML Schema IS the Target Architecture**: Hardware engineers specify instruction sets, memory hierarchies, timing expressions, constraints, and cost models in declarative YAML files.
* **Zero C++ Compiler Rewrites**: Target instruction selection, schedule costing, access descriptor synthesis, and address generation are computed dynamically from the schema.
* **Fail-Closed Mathematical Soundness**: If a kernel operation violates memory alignment, stride predicates, or lacks hardware coverage, the compiler emits formal `UNSUPPORTED` markers rather than silently emitting invalid assembly or guessing fallback semantics.
* **PyTorch 2.x Ecosystem Native**: Operates as a first-class `torch.compile` backend with seamless device registration and dynamic tensor shape padding.

---

## 2. Compilation Pipeline Architecture

```
                                  [ Input Sources ]
                                         │
                 ┌───────────────────────┴────────────────────────┐
                 ▼                                                ▼
     Triton-IR Text (.ttir)                             PyTorch FX Graph
  (TTIR Syntax & Attributes)                       (torch.compile / Dynamo Seam)
                 │                                                │
                 ▼                                                ▼
         [ TTIR Parser ]                              [ Dynamic Extractor / ]
    (Lexer, Ast, Ssa Names)                           [   FlagGems Bridge   ]
                 │                                                │
                 └───────────────────────┬────────────────────────┘
                                         ▼
                             [ Def-Use Graph & CFG ]
                                         │
                                         ▼
                            [ Access Descriptor Engine ]
                     (Affine Strides, Base Offsets, Multi-Dims)
                                         │
                                         ▼
                      [ Declarative Instruction Selector ]
                 ◄── Ingests Target Schema (`*.yaml` / Rules)
                 ◄── Evaluates Predicates & Admissibility
                 ◄── Minimum-Cost Scheduling & Tie-Breaking
                                         │
                                         ▼
                            [ Target IR Assembly Engine ]
                 (Loop Structures, Memory Descriptors, Ssa Binds)
                                         │
                                         ▼
                           [ Execution & Emulation Layer ]
                 ├── Fast Native C++ Engine (`_emu_cpp`)
                 ├── High-Precision NumPy Reference Engine
                 ├── DXA Asynchronous DMA & Multicast Engine
                 └── TCU Tensor Core Unit (WGMMA / WMMA / Sparsity)
                                         │
                                         ▼
                            [ Hardware Performance Model ]
                 (DRAM Transactions, Cache Lines, Bank Conflicts)
```

---

## 3. Supported Target Architectures

The framework provides three production-grade target models out of the box:

| Target Identifier | Architecture Family | Memory Model | Compute Core Units | Key Features |
|---|---|---|---|---|
| **`toyisa1`** | Streaming ASIC | Flat 1D Shared Memory | `MAC8`, `MAC16` Systolic Arrays, `EPI` | Deterministic pipeline, 1D DMA engine |
| **`toyisa2`** | Banked Scratchpad ASIC | 16-Bank Interleaved SRAM | `OPU32` Outer-Product Units, `VPU` | Bank-conflict cost arbitration |
| **`vortex_rvgpu`** | RISC-V SIMT GPGPU | Hierarchical GMEM + LMEM | `TCU_WGMMA32`, `TCU_WMMA16`, `TCU_WGMMA_SP32`, `VADD`, `VMUL`, `VRELU` | **Hopper-TMA DXA async copy**, multicast deduplication, K-major transpose scatter, 4 FEDP backends, 2:4 structured sparsity, MXFP8 microscaling |

---

## 4. Evaluation & Verification Results

### A. Full Test Suite Execution Summary
Executed across contract suites, integration harnesses, unit models, and C++ bindings:

```
============================= test session starts ==============================
platform linux -- Python 3.14.7, pytest-9.1.1, pluggy-1.6.0
collected 182 items

tests/contract/test_access_descriptor.py ......                          [  3%]
tests/contract/test_assembler.py ...                                     [  4%]
tests/contract/test_coverage_report.py .......                           [  8%]
tests/contract/test_emulator.py ........                                 [ 13%]
tests/contract/test_end_to_end.py .....                                  [ 15%]
tests/contract/test_fixtures.py ..............................           [ 32%]
tests/contract/test_fx_lower.py .........                                [ 37%]
tests/contract/test_isa_schema.py .....                                  [ 40%]
tests/contract/test_raw_module.py ...                                    [ 41%]
tests/contract/test_selector.py ....                                     [ 43%]
tests/contract/test_to_ir.py ....                                        [ 46%]
tests/contract/test_torch_seam.py ......                                 [ 49%]
tests/contract/test_ttir_parser.py ....                                  [ 51%]
tests/integration/test_dynamic_extraction.py ........................... [ 66%]
..                                                                       [ 67%]
tests/integration/test_flaggems_bridge.py ..........                     [ 73%]
tests/unit/test_emu_cpp.py .......                                       [ 76%]
tests/unit/test_hardware_models.py .......                               [ 80%]
tests/unit/test_no_triton_runtime.py .........                           [ 85%]
tests/unit/test_parser_syntax_negative.py ........                       [ 90%]
tests/unit/test_vortex_capabilities.py ..................                [100%]

============================= 182 passed in 17.45s =============================
```

### B. Standard Corpus Coverage & Refusal Verification

| Target ISA | Kernel Tier | Fully Lowered | Subgraph Coverage | Status / Refusal Reason | Modeled Latency |
|---|---|:---:|:---:|---|:---:|
| `toyisa1` | `t0_vecadd` | **YES** | 100% | Lowered to `DMA1D` + `EPI` | 5.3 ms |
| `toyisa1` | `t1_matmul` | **YES** | 100% | Lowered to `DMA1D` + `MAC16` + `EPI` | 12.3 ms |
| `toyisa1` | `t2_matmul_relu` | **YES** | 100% | Lowered to `DMA1D` + `MAC16` + `EPI` | 13.4 ms |
| `toyisa1` | `t3_modulo` | **REFUSED** | 99% | **Refused**: Unstructured pointer modulo wraparound | 4.9 ms |
| `toyisa2` | `t0_vecadd` | **YES** | 100% | Lowered to `LDG` + `VPU` | 4.5 ms |
| `toyisa2` | `t1_matmul` | **YES** | 100% | Lowered to `LDG` + `OPU32` + `VPU` | 14.3 ms |
| `toyisa2` | `t2_matmul_relu` | **YES** | 100% | Lowered to `LDG` + `OPU32` + `VPU` | 16.2 ms |
| `toyisa2` | `t3_modulo` | **REFUSED** | 99% | **Refused**: Unstructured pointer modulo wraparound | 6.3 ms |
| `vortex_rvgpu` | `t0_vecadd` | **YES** | 100% | Lowered to `LDG` + `VADD` | 5.4 ms |
| `vortex_rvgpu` | `t1_matmul` | **YES** | 100% | Lowered to `LDG` + `TCU_MMA32` + `VADD`/`VMUL` | 17.2 ms |
| `vortex_rvgpu` | `t2_matmul_relu` | **YES** | 100% | Lowered to `LDG` + `TCU_MMA32` + `VRELU` | 18.8 ms |
| `vortex_rvgpu` | `t3_modulo` | **REFUSED** | 99% | **Refused**: Unstructured pointer modulo wraparound | 7.4 ms |

> **Fail-Closed Guarantee**: `t3_modulo` serves as the negative control fixture across all targets. The compiler mathematically refuses unstructured modulo indexing on pointer calculations rather than silently generating invalid address strides.

### C. Cross-ISA Retargeting & Cost Transfer

| Kernel Tier | `toyisa1` Baseline Cost | `toyisa2` Banked Cost | Delta (%) | `vortex_rvgpu` SIMT Cost |
|---|:---:|:---:|:---:|:---:|
| `t0_vecadd` | 9,217.5 cycles | 7,374.0 cycles | **-20.0%** | 6,837.5 cycles |
| `t1_matmul` | 35,772.9 cycles | 28,402.0 cycles | **-20.6%** | 6,837.5 cycles |
| `t2_matmul_relu` | 42,076.9 cycles | 33,445.2 cycles | **-20.5%** | 7,120.0 cycles |
| `t3_modulo` | 1,400.0 cycles (refused) | 1,120.0 cycles (refused) | **-20.0%** | 1,120.0 cycles (refused) |

---

## 5. Getting Started & Installation

### Prerequisites
* Linux (x86_64 or aarch64)
* Python 3.10+
* CMake 3.18+ (for C++ acceleration module)
* PyTorch 2.1+

### Installation
Clone the repository and install dependencies in editable mode:
```bash
git clone https://github.com/Incharajayaram/segfault-triton-toyisa.git
cd segfault-triton-toyisa

# Install core dependencies
pip install -e .

# Or install with development and test dependencies
pip install -e .[dev]
```

### Building the Optional High-Performance C++ Emulator
The NumPy emulator is always active by default. For up to 10× faster kernel execution:
```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
cd ..
```

---

## 6. How to Run the Tests & Verifiers

### 1. Run Complete Pytest Suite
```bash
pytest
```

### 2. Run Formal Verification Pipeline (12 Contracts)
Verifies IR round-tripping, MLIR bindings, fixture canon, and def-use graphs:
```bash
python3 verify/run_all.py
```

### 3. Run PyTorch Seam Integration Suite
Verifies `torch.compile` backend registration, custom device visibility, and tensor data planes:
```bash
python3 test_mvp_integration.py
```

### 4. Run Automated 5-Stage Live Demonstration
```bash
bash demo_quick.sh
```

---

## 7. Interactive Demos & Visual Diff Explorers

We include multiple interactive tools to inspect how kernels lower across different chips:

### A. Terminal Interactive Diff Explorer
Step through side-by-side assembly listings and hardware cost models:
```bash
python3 demo_diff.py
```
To run unattended in automated mode:
```bash
python3 demo_diff.py --auto
```

### B. Interactive Web GUI Compiler Diff Explorer
Launch local HTTP server to view the standalone, single-file browser dashboard:
```bash
python3 -m http.server 8080
```
Open in your browser:
* **`http://localhost:8080/demo_diff.html`** or **`http://localhost:8080/showcase_diff_demo.html`**
* Features: Side-by-side target ISA comparison, interactive opcode highlights, cycle breakdowns, and refusal audits.

---

## 8. Usage Guide & APIs

### A. PyTorch `torch.compile` Backend Integration
```python
import torch
from triton_toyisa.torch_backend.compiler import toyisa_backend

class SimpleMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(64, 128)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(128, 32)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

model = SimpleMLP()
x = torch.randn(16, 64)

# Compile model targeting custom AI accelerator
opt_model = torch.compile(model, backend="toyisa")
output = opt_model(x)

# Inspect lowering plan, node coverage, and hardware statistics
plan = opt_model.toyisa_plan
print(f"Fully lowered: {plan.fully_lowered}")
print(f"Nodes lowered to chip: {plan.node_lowerings}")
```

### B. Command-Line Interface (`toyisa`)
Compile arbitrary TTIR fixtures directly from the terminal:
```bash
# Compile fixture to Vortex RVGPU
toyisa lower fixtures/t1_matmul.ttir --isa vortex_rvgpu

# Compile and print cycle cost analysis
toyisa cost fixtures/t0_vecadd.ttir --isa toyisa2

# Run full diagnostics on a kernel
toyisa diagnose fixtures/t3_modulo.ttir --isa toyisa1
```

### C. Programmatic Python API
```python
from triton_toyisa.lower import lower_fixture, lower_text
from triton_toyisa.isa.schema import load_builtin

# Lower canonical fixture against Vortex RVGPU target
ctx = lower_fixture("t1_matmul", isa_name="vortex_rvgpu")

print(f"Emitted instructions: {len(ctx.program.instructions())}")
print(f"Hardware compute cycles: {ctx.hardware_stats.compute_cycles}")
print(f"DRAM bytes requested: {ctx.hardware_stats.dram_bytes_requested}")
print(f"Numerical relative error: {ctx.parity_max_rel_err:.4e}")
```

---

## 9. Project Structure

```
segfault-triton-toyisa/
├── bench/                         # Benchmarking adapters, harnesses, and baseline results
│   ├── adapter.py                 # Pipeline benchmark harness
│   └── results.json               # Recorded latency, cost, and coverage metrics
├── docs/                          # Architecture documentation and published papers
│   ├── papers/                    # Reference literature (Vortex GPGPU, DXA, microarchitectures)
│   └── team/                      # Ownership, tracks, and methodology guides
├── fixtures/                      # Canonical TTIR kernel fixtures (t0 to t3)
├── specs/                         # Technical specs, RFCs, and formal contracts
├── src/triton_toyisa/             # Core Compiler Implementation
│   ├── canon/                     # IR Canonicalization passes
│   ├── emit/                      # Target assembly and IR emission
│   ├── emu/                       # Native execution & hardware simulation
│   │   ├── cpp/                   # C++ high-performance execution engine
│   │   ├── dxa.py                 # Vortex DXA asynchronous bulk copy engine
│   │   ├── tcu.py                 # Vortex TCU tensor core unit (WGMMA/WMMA)
│   │   └── exec.py                # Machine state emulator & runtime dispatch
│   ├── extract/                   # Dynamic Triton extraction & FlagGems bridge
│   ├── isa/                       # Declarative schema engine & instruction selectors
│   │   └── schemas/               # Hardware architecture YAML definitions
│   │       ├── toyisa1.yaml       # Streaming DMA ASIC
│   │       ├── toyisa2.yaml       # Banked Scratchpad ASIC
│   │       └── vortex_rvgpu.yaml  # Vortex RISC-V SIMT GPGPU
│   ├── recognize/                 # Affine loop and memory access descriptor synthesis
│   ├── report/                    # Coverage, transfer, and diagnostic generators
│   ├── torch_backend/             # PyTorch 2.x Dynamo Seam & compiler backend
│   └── ttir/                      # Triton-IR Lexer, Parser, AST, and SSA graph
├── tests/                         # Test Suites (182 tests)
│   ├── contract/                  # Seam and schema contract tests
│   ├── integration/               # Dynamic extraction & FlagGems bridge tests
│   └── unit/                      # Hardware models, parser, and Vortex capability tests
├── tools/                         # CLI utilities, live demos, and diff generators
└── verify/                        # Formal verification scripts (12 scripts)
```

---

## 10. License & Citation

Licensed under the Apache License 2.0. Developed as part of the **IICT CompilerTech Hackathon 2025**.
