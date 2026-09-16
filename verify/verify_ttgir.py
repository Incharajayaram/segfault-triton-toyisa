"""Enumerate the *entities* a Triton-level IR2Vec-style vocabulary would need.

Motivation: IR2Vec's vocabulary is (opcode, type, operand) over a flat LLVM IR.
MIR2Vec re-derives the vocabulary (machine opcodes, register classes) for Machine IR.
The question is what the equivalent re-derivation looks like for ttir/ttgir.

This dumps the two Triton IR levels and extracts the attribute/layout entities that
actually drive cost -- the things a naive opcode-only port would miss.
"""

import re
from collections import Counter

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.compiler import compile as tt_compile

TARGET = GPUTarget("cuda", 80, 32)


@triton.jit
def matmul(
    a_ptr, b_ptr, c_ptr, M, N, K,
    sam, sak, sbk, sbn, scm, scn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a_ptrs = a_ptr + (rm[:, None] * sam + rk[None, :] * sak)
    b_ptrs = b_ptr + (rk[:, None] * sbk + rn[None, :] * sbn)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _k in range(0, tl.cdiv(K, BK)):
        acc = tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), acc)
        a_ptrs += BK * sak
        b_ptrs += BK * sbk
    c_ptrs = c_ptr + (rm[:, None] * scm + rn[None, :] * scn)
    tl.store(c_ptrs, acc)


src = ASTSource(
    fn=matmul,
    signature={"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32",
               "M": "i32", "N": "i32", "K": "i32",
               "sam": "i32", "sak": "i32", "sbk": "i32", "sbn": "i32",
               "scm": "i32", "scn": "i32",
               "BM": "constexpr", "BN": "constexpr", "BK": "constexpr"},
    constexprs={"BM": 64, "BN": 64, "BK": 32},
)
c = tt_compile(src, target=TARGET)

LEVELS = ["ttir", "ttgir"]
OP = re.compile(r"\b((?:tt|ttg|tts|arith|scf|math|cf|gpu)\.[a-zA-Z_][\w.]*)")

for lvl in LEVELS:
    if lvl not in c.asm:
        print(f"--- {lvl}: NOT AVAILABLE ---")
        continue
    txt = c.asm[lvl]
    ops = Counter(OP.findall(txt))
    print("=" * 74)
    print(f"LEVEL: {lvl}   ({len(txt.splitlines())} lines, "
          f"{sum(ops.values())} op-instances, {len(ops)} distinct ops)")
    print("=" * 74)
    print("  distinct ops (vocabulary candidates, 'opcode' entities):")
    for o, n in ops.most_common():
        print(f"     {o:<34} x{n}")

    # the entities a pure-opcode port would MISS
    print("\n  attribute-style entities present at this level:")
    for pat, label in [
        (r"#\w+\s*=\s*#ttg\.blocked<\{[^}]*\}", "#ttg.blocked<{...}> layout"),
        (r"#\w+\s*=\s*#ttg\.mma<[^>]*>", "#ttg.mma<...> layout"),
        (r"#\w+\s*=\s*#ttg\.shared[^>]*>?", "#ttg.shared memory encoding"),
        (r"inputPrecision\s*=\s*\w+", "inputPrecision (e.g. tf32)"),
        (r"num_warps\s*=\s*\d+", "num_warps"),
        (r"num_stages\s*=\s*\d+", "num_stages"),
        (r"tensor<\d+x\w+>", "tensor shapes"),
        (r"sizePerThread|threadsPerWarp|warpsPerCTA|order", "layout sub-fields"),
        (r"tt\.make_range\s*\{[^}]*\}", "make_range bounds"),
        (r"scf\.for[^\n]*", "loop structure"),
    ]:
        found = re.findall(pat, txt)
        uniq = sorted(set(found))[:3]
        print(f"     {label:<34} : {len(found):>3}  {uniq if uniq else ''}")

    # shape cardinality: how many distinct tensor shapes?
    shapes = Counter(re.findall(r"tensor<[^>]+>", txt))
    print(f"\n  distinct tensor shape/type strings : {len(shapes)}")
    print(f"     {list(shapes)[:8]}")

    with open(f"/tmp/matmul.{lvl}.txt", "w") as f:
        f.write(txt)

print("\nwrote /tmp/matmul.ttir.txt and /tmp/matmul.ttgir.txt")
