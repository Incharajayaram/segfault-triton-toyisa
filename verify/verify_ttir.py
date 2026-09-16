"""Verify the ttir-level claims in methodology.pdf against a real Triton install.

Tests, in order:
  GATE  (methodology 3.2)  : can we get ttir with NO GPU, via an explicit target?
  T1    (methodology 6.1)  : is tiled matmul really `scf.for ... iter_args ... tt.dot`?
  T0    (methodology 3.3)  : vector add has no tt.dot  (true-negative for idiom detection)
  T3    (methodology 3.3)  : modulo pointer wraparound -> unstructured?
"""

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.compiler import compile as tt_compile

TARGET = GPUTarget("cuda", 80, 32)  # explicit target: no GPU / driver needed


# ---------------------------------------------------------------- kernels
@triton.jit
def vec_add(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


@triton.jit
def matmul(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
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
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK * sak
        b_ptrs += BK * sbk
    c_ptrs = c_ptr + (rm[:, None] * scm + rn[None, :] * scn)
    tl.store(c_ptrs, acc)


@triton.jit
def matmul_relu(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
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
    bias = tl.load(bias_ptr + rn)
    out = tl.maximum(acc + bias[None, :], 0.0)
    c_ptrs = c_ptr + (rm[:, None] * scm + rn[None, :] * scn)
    tl.store(c_ptrs, out)


@triton.jit
def modulo_wrap(x_ptr, out_ptr, M, N, stride_xm, stride_xn,
                stride_om, stride_on, BLOCK: tl.constexpr):
    # pointer arithmetic that wraps around the tensor via modulo
    rm = (2 + tl.arange(0, BLOCK)) % M
    rn = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + (rm[:, None] * stride_xm + rn[None, :] * stride_xn))
    tl.store(out_ptr + (rm[:, None] * stride_om + rn[None, :] * stride_on), x)


# ---------------------------------------------------------------- helper
def get_ttir(fn, signature, constexprs=None, opts=None):
    src = ASTSource(fn=fn, signature=signature,
                    constexprs=constexprs or {})
    kw = dict(target=TARGET)
    if opts:
        kw["options"] = opts
    return tt_compile(src, **kw).asm["ttir"]


PATTERNS = [
    "tt.dot", "scf.for", "scf.yield", "iter_args",
    "tt.load", "tt.store", "tt.addptr", "tt.splat", "tt.broadcast",
    "tt.make_range", "tt.get_program_id", "arith.muli",
    "arith.remsi", "arith.remui", "arith.remf", "tt.func",
    "#blocked", "loc(", "tt.tt_load_to_local", "ttg.", "cf.cond_br",
]


def probe(name, ttir):
    print(f"\n{'='*72}\n{name}   ({len(ttir.splitlines())} ttir lines)")
    print("=" * 72)
    for p in PATTERNS:
        n = ttir.count(p)
        if n:
            print(f"  {p:<24}: {n}")
    print("  [absent]               : "
          + ", ".join(p for p in PATTERNS if ttir.count(p) == 0))
    return ttir


results = {}

CASES = [
    ("T0_vec_add", "Tier 0 - vector add", vec_add,
     {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32",
      "n": "i32", "BLOCK": "constexpr"}, {"BLOCK": 1024}),
    ("T1_matmul", "Tier 1 - tiled matmul", matmul,
     {"a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32",
      "M": "i32", "N": "i32", "K": "i32",
      "sam": "i32", "sak": "i32", "sbk": "i32", "sbn": "i32",
      "scm": "i32", "scn": "i32",
      "BM": "constexpr", "BN": "constexpr", "BK": "constexpr"},
     {"BM": 64, "BN": 64, "BK": 32}),
    ("T2_matmul_relu", "Tier 2 - matmul + bias + relu epilogue", matmul_relu,
     {"a_ptr": "*fp32", "b_ptr": "*fp32", "bias_ptr": "*fp32",
      "c_ptr": "*fp32",
      "M": "i32", "N": "i32", "K": "i32",
      "sam": "i32", "sak": "i32", "sbk": "i32", "sbn": "i32",
      "scm": "i32", "scn": "i32",
      "BM": "constexpr", "BN": "constexpr", "BK": "constexpr"},
     {"BM": 64, "BN": 64, "BK": 32}),
    ("T3_modulo", "Tier 3 - modulo wraparound (negative control)", modulo_wrap,
     {"x_ptr": "*fp32", "out_ptr": "*fp32",
      "M": "i32", "N": "i32",
      "stride_xm": "i32", "stride_xn": "i32",
      "stride_om": "i32", "stride_on": "i32",
      "BLOCK": "constexpr"}, {"BLOCK": 16}),
]

for key, label, fn, sig, cx in CASES:
    try:
        results[key] = probe(label, get_ttir(fn, sig, cx))
    except Exception as e:
        print(f"\n{label}: {key} FAILED - {type(e).__name__}: {e}")

for k, v in results.items():
    with open(f"/tmp/{k}.ttir", "w") as f:
        f.write(v)
    print(f"wrote /tmp/{k}.ttir ({len(v.splitlines())} lines)")
