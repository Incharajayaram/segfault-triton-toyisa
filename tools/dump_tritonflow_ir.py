#!/usr/bin/env python3

"""
Compile a PyTorch matmul through the TritonFlow backend and dump the
generated TritonFlow representation.

This intentionally does NOT use torch.device("tritonflow").
The inputs remain ordinary CPU tensors; torch.compile() sends the
captured graph through the TritonFlow backend and emulator.
"""

from __future__ import annotations

from pathlib import Path

import torch

from triton_tritonflow.emu import HAS_CPP

OUTPUT = Path("tritonflow_generated.txt")


def dump_object(obj: object, name: str) -> str:
    """Dump useful string/repr attributes from an arbitrary backend object."""

    lines: list[str] = []

    lines.append(f"OBJECT: {name}")
    lines.append(f"TYPE: {type(obj)}")
    lines.append("")

    # Common names for generated code / IR.
    preferred = (
        "asm",
        "assembly",
        "code",
        "source",
        "ir",
        "ttir",
        "program",
        "text",
        "kernel",
        "instructions",
    )

    seen: set[str] = set()

    for attr in preferred:
        if not hasattr(obj, attr):
            continue

        try:
            value = getattr(obj, attr)
        except Exception as exc:
            lines.append(f"[{attr}: failed to read: {exc}]")
            continue

        seen.add(attr)

        lines.append(f"===== {attr} =====")
        lines.append(str(value))
        lines.append("")

    # Dump other public string-valued attributes.
    for attr in dir(obj):
        if attr.startswith("_") or attr in seen:
            continue

        try:
            value = getattr(obj, attr)
        except Exception:
            continue

        if isinstance(value, str) and value.strip():
            lines.append(f"===== {attr} =====")
            lines.append(value)
            lines.append("")

    # Finally include repr so that custom objects which don't expose
    # their code as a string are still visible.
    lines.append("===== repr(object) =====")
    lines.append(repr(obj))
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    print(f"PyTorch: {torch.__version__}")
    print(f"HAS_CPP: {HAS_CPP}")

    if not HAS_CPP:
        print(
            "WARNING: C++ emulator is not loaded. "
            "The TritonFlow NumPy/reference emulator may be used instead."
        )

    torch.manual_seed(0)

    # 128x64 @ 64x128 -> 128x128
    a = torch.randn(128, 64, dtype=torch.float32)
    b = torch.randn(64, 128, dtype=torch.float32)

    def kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, y)

    print("\nCompiling with TritonFlow...")
    compiled = torch.compile(kernel, backend="tritonflow")

    print("Executing...")
    result = compiled(a, b)

    # TritonFlow backend metadata exposed by your backend.
    plan = compiled.tritonflow_plan

    print()
    print("========================================")
    print("TritonFlow compilation plan")
    print("========================================")

    print(f"Fully lowered: {plan.fully_lowered}")

    if not plan.fully_lowered:
        print("\nFallbacks:")
        for fallback in plan.fallbacks:
            print(f"  {fallback}")

        raise SystemExit(
            "\nERROR: graph was not fully lowered by TritonFlow."
        )

    print("\nLowered kernels:")
    for kernel in plan.lowered:
        print(f"  {kernel.name}")

    # ------------------------------------------------------------
    # Dump generated TritonFlow representation.
    # ------------------------------------------------------------

    output: list[str] = []

    output.append("TritonFlow generated code / IR")
    output.append("=" * 80)
    output.append("")
    output.append(f"PyTorch version: {torch.__version__}")
    output.append(f"C++ emulator loaded: {HAS_CPP}")
    output.append(f"Fully lowered: {plan.fully_lowered}")
    output.append("")

    for i, kernel_obj in enumerate(plan.lowered):
        print()
        print("========================================")
        print(f"Kernel {i}: {kernel_obj.name}")
        print("========================================")

        text = dump_object(kernel_obj, kernel_obj.name)

        print(text)
        output.append(text)
        output.append("\n")

    OUTPUT.write_text("\n".join(output), encoding="utf-8")

    # ------------------------------------------------------------
    # Correctness check.
    # ------------------------------------------------------------

    expected = torch.matmul(a, b)

    max_err = (result - expected).abs().max().item()

    print("========================================")
    print("Correctness")
    print("========================================")
    print(f"Result shape: {tuple(result.shape)}")
    print(f"Max absolute error: {max_err:.3e}")

    if max_err > 1e-2:
        raise SystemExit(
            f"ERROR: TritonFlow result differs from PyTorch by {max_err:.3e}"
        )

    print()
    print("PASS")
    print()
    print(f"Generated representation saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
