# Coverage Report

**Generated**: 2026-09-18T07:07:35.508609  
**Pipeline Version**: 0.1.0

## Summary

- Total tiers tested: 12
- Fully lowered: 9/12
- ISAs tested: 3
- Average latency: 10.5 ms

## Per-Tier Coverage

| ISA | Tier | fully lowered | largest subgraph | annotated (≤) | unsupported | latency ms |
|---|---|---|---|---|---|---|
| tritonflow1 | t0_vecadd | yes | 1.00 | 1.00 | – | 5.3 |
| tritonflow1 | t1_matmul | yes | 1.00 | 1.00 | – | 12.3 |
| tritonflow1 | t2_matmul_relu | yes | 1.00 | 1.00 | – | 13.4 |
| tritonflow1 | t3_modulo | NO | 0.99 | 1.00 | 2 (x) | 4.9 |
| tritonflow2 | t0_vecadd | yes | 1.00 | 1.00 | – | 4.5 |
| tritonflow2 | t1_matmul | yes | 1.00 | 1.00 | – | 14.3 |
| tritonflow2 | t2_matmul_relu | yes | 1.00 | 1.00 | – | 16.2 |
| tritonflow2 | t3_modulo | NO | 0.99 | 1.00 | 2 (x) | 6.3 |
| vortex_rvgpu | t0_vecadd | yes | 1.00 | 1.00 | – | 5.4 |
| vortex_rvgpu | t1_matmul | yes | 1.00 | 1.00 | – | 17.2 |
| vortex_rvgpu | t2_matmul_relu | yes | 1.00 | 1.00 | – | 18.8 |
| vortex_rvgpu | t3_modulo | NO | 0.99 | 1.00 | 2 (x) | 7.4 |

> [!NOTE]
> `annotated (≤)` is an upper bound on coverage. `fully lowered` is the definitive metric.

## Instruction Mix

### tritonflow1 / t0_vecadd

- `EPI`: 15 (83.3%)
- `DMA1D`: 3 (16.7%)

### tritonflow1 / t1_matmul

- `EPI`: 55 (93.2%)
- `DMA1D`: 3 (5.1%)
- `MAC16`: 1 (1.7%)

### tritonflow1 / t2_matmul_relu

- `EPI`: 61 (92.4%)
- `DMA1D`: 4 (6.1%)
- `MAC16`: 1 (1.5%)

### tritonflow1 / t3_modulo

- `EPI`: 25 (100.0%)

### tritonflow2 / t0_vecadd

- `VPU`: 15 (83.3%)
- `LDG`: 3 (16.7%)

### tritonflow2 / t1_matmul

- `VPU`: 55 (93.2%)
- `LDG`: 3 (5.1%)
- `OPU32`: 1 (1.7%)

### tritonflow2 / t2_matmul_relu

- `VPU`: 61 (92.4%)
- `LDG`: 4 (6.1%)
- `OPU32`: 1 (1.5%)

### tritonflow2 / t3_modulo

- `VPU`: 25 (100.0%)

### vortex_rvgpu / t0_vecadd

- `VADD`: 14 (77.8%)
- `LDG`: 3 (16.7%)
- `VMUL`: 1 (5.6%)

### vortex_rvgpu / t1_matmul

- `VADD`: 44 (74.6%)
- `VMUL`: 10 (16.9%)
- `LDG`: 3 (5.1%)
- `VDIV`: 1 (1.7%)
- `TCU_MMA32`: 1 (1.7%)

### vortex_rvgpu / t2_matmul_relu

- `VADD`: 49 (74.2%)
- `VMUL`: 10 (15.2%)
- `LDG`: 4 (6.1%)
- `VDIV`: 1 (1.5%)
- `TCU_MMA32`: 1 (1.5%)
- `VRELU`: 1 (1.5%)

### vortex_rvgpu / t3_modulo

- `VADD`: 20 (80.0%)
- `VMUL`: 4 (16.0%)
- `VMOD`: 1 (4.0%)
