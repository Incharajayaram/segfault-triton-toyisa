# TritonFlow Strategic Architecture & Positioning

**Authoritative Technical Scope, Claims, and Strategic Roadmap**  
*Project SegFault / TritonFlow — Compiler Engineering Track*

---

## 1. Executive Summary & Core Thesis

TritonFlow is a schema-driven compiler prototype that consumes Triton Tensor IR (TTIR), recovers structured geometric access descriptors and semantic compute idioms, and lowers them directly to target accelerator programs using declarative ISA schemas.

### The Defensible Wedge: Front Door & Retargetability
Prior work such as ACT (OOPSLA 2026, UIUC & NVIDIA) established the viability of automated backend generation from declarative ISA schemas, but operates strictly on **XLA-HLO**. TritonFlow solves the missing half of the modern AI compiler stack:
1. **Frontend Integration**: Meeting practitioners directly at **PyTorch `torch.compile(backend="tritonflow")`** and **Triton IR**, rather than requiring monolithic XLA graph imports.
2. **Schema-Driven Multi-ISA Retargetability**: Supporting $\ge 2$ distinct target ISAs (`tritonflow1`, `tritonflow2`, and the open-source `vortex_rvgpu` RISC-V GPGPU) from a single unified compiler codebase without hardcoded per-target lowering tables.
3. **Fail-Closed Safety**: Unmapped instructions and unstructured memory access patterns trigger transparent, eager PyTorch fallback with zero silent miscompilations.

---

## 2. What We Claim vs. What We Do NOT Claim

| Topic | What We Explicitly Claim | What We Do NOT Claim |
|---|---|---|
| **Ecosystem Seam** | Clean interception at `torch.compile` and TTIR parsing via a registered PyTorch backend seam | Universal coverage of the entire PyTorch ATen operator surface |
| **CUDA Interception** | We target Triton IR directly as an open intermediate representation | **No "CUDA hijack" claim**: GPU performance resides in closed binaries (`ptxas`, cuBLAS, CUTLASS); compiling CUDA source pulls in SIMT thread/warp baggage without inheriting closed vendor heuristics |
| **Backend Generality** | Declarative ISA schemas drive instruction selection via mathematical cost models and geometric constraints | We do NOT claim to have invented declarative ISAs from whole cloth (ACT established this for XLA-HLO); our contribution is bringing declarative schemas to Triton IR |
| **Performance** | Rapid accelerator bring-up (weeks to define a schema vs. months of C++ LLVM backend engineering) with cycle-accurate microarchitectural simulation | **No claim of matching hand-written performance without autotuning**: optimal matrix multiplication requires tiling search, double buffering, and software pipelining |
| **Addressing** | Full address math subsumption: pointer arithmetic, splats, ranges, and increments are elided and folded into hardware DMA/LDS descriptors | Emitting flat scalar SSA address instructions alongside vector memory loads |
| **Sparsity & Formats** | Strict data preconditions: 2:4 structured sparse units (`TCU_WGMMA_SP32`) and microscaled formats (`TCU_WGMMA_MXFP8`) are only selected when input data has explicit sparse/scale metadata | Misselecting sparse units for dense tensors merely because theoretical FLOP cost is lower |
| **Negative Control** | Transparent fail-closed behavior: unsupported addressing (e.g. modulo wraparound) generates explicit `UNSUPPORTED` markers and falls back cleanly to PyTorch eager execution | Silent miscompilation or faking execution of unsupported patterns |

---

## 3. Why Retargetability Requires Multiple ISAs

A critical finding of compiler verification:
> **A second ISA is the only falsifiable test of "declarative compilation."**

If a compiler supports only a single target (e.g. only Vortex RISC-V), it is impossible to prove whether lowering decisions are derived from the declarative schema or whether the pipeline has been subtly hardcoded to one architecture's idiosyncrasies.

TritonFlow maintains three distinct targets:
1. **`tritonflow1`**: Baseline single-bank flat-memory accelerator with contiguous and tiled DMA engines (`DMA1D`, `DMA2D`, `MAC8`, `MAC16`).
2. **`tritonflow2`**: Multi-bank partitioned memory architecture with inter-bank routing operations (`MOVE_BANK`, `OPU16`, `OPU32`).
3. **`vortex_rvgpu`**: Real, open-source RISC-V GPGPU architecture featuring:
   - Direct eXecution Accelerator (DXA) async bulk copy and multicast engine.
   - Tensor Core Unit (TCU) supporting WMMA, WGMMA, 2:4 structured sparsity, and MXFP8 microscaling.
   - 16-bank conflict modeling and hardware CSR performance counters.

The identical input TTIR compiles to all three targets, selecting the optimal instruction set defined by each schema's cost model.

---

## 4. Architectural Verification Across the Graduated Tier Ladder

| Tier | Kernel Idiom | Lowering Result | Emulation & Numerical Parity | Verification Status |
|---|---|---|---|:---:|
| **Tier 0** (`t0_vecadd`) | 1D Vector Addition | 4 instructions (`LDG`, `LDG`, `VADD`, `STG`); all 14 pointer/indexing ops elided | Exact bitwise match ($0.000	imes 10^0$ error vs. NumPy/PyTorch) | **PASS** |
| **Tier 1** (`t1_matmul`) | 2D Tiled GEMM ($64	imes 32 	imes 64$) | Tiled memory descriptors + Dense WGMMA (`TCU_WGMMA32`, 32c) | Bounded TF32 relative error $< 3.125	imes 10^{-2}$ | **PASS** |
| **Tier 2** (`t2_matmul_relu`) | GEMM + Fused Epilogue | Tiled DMA + WGMMA + `VRELU` elementwise fusion | Satisfies TF32 bound; verified strictly non-negative outputs | **PASS** |
| **Tier 3** (`t3_modulo`) | Modulo Address Wraparound | **Fail-Closed Negative Control**: generates 2 `UNSUPPORTED` markers | Transparent eager PyTorch fallback invoked; zero crashes | **PASS** |

---

## 5. Strategic Roadmap

1. **Short Term (Immediate / Current)**:
   - Address math elision in production compilation paths.
   - Data preconditions enforced for all sparse and microscaled tensor core units.
   - PyTorch fallback seam hardened for production inference pipelines.
2. **Medium Term**:
   - Integration of autotuning search loop for tile shape selection ($BM, BN, BK$) using the cycle-accurate emulator as cost oracle.
   - Equality-saturation rewrite rules for complex tensor contraction identities.
3. **Long Term**:
   - Automated schema derivation from RTL/Verilog accelerator definitions (TensorLift integration).
