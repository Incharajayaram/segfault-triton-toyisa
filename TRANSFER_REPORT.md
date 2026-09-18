# Cross-ISA Transfer Report: `toyisa1` → `toyisa2`

**Generated**: 2026-09-18T07:07:35.509183  
**Zero-Edit Status**: ⚠️ Requires External Edits

## Pipeline Stage Transferability

| Pipeline Stage | Target ISA | Transferred | Edit Location | Notes |
|---|---|---|---|---|
| `frontend_parser` | `toyisa2` | ✅ YES | – | Zero modifications required; schema-driven. |
| `def_use_graph` | `toyisa2` | ✅ YES | – | Zero modifications required; schema-driven. |
| `access_descriptor` | `toyisa2` | ✅ YES | – | Zero modifications required; schema-driven. |
| `hardware_models` | `toyisa2` | ✅ YES | – | Zero modifications required; schema-driven. |
| `instruction_selector` | `toyisa2` | ✅ YES | – | Zero modifications required; schema-driven. |
| `code_emitter` | `toyisa2` | ✅ YES | – | Zero modifications required; schema-driven. |
| `machine_emulator` | `toyisa2` | ✅ YES | – | Zero modifications required; schema-driven. |

## Cost Comparison vs Baseline

| Tier | `toyisa1` Cost | `toyisa2` Cost | Delta (%) |
|---|---|---|---|
| `t0_vecadd` | 9217.5 | 7374.0 | -20.0% |
| `t1_matmul` | 35772.9 | 28402.0 | -20.6% |
| `t2_matmul_relu` | 42076.9 | 33445.2 | -20.5% |
| `t3_modulo` | 1400.0 | 1120.0 | -20.0% |

## Edits Outside Schemas and Rules

- `src/triton_toyisa/isa/schemas/toyisa2.yaml` (96 lines): Banked scratchpad target schema
- `src/triton_toyisa/isa/rules/toyisa2.py` (48 lines): Custom lowering rules