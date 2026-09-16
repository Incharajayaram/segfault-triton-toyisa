import os

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.compiler import compile as tt_compile


@triton.jit
def t0_vecadd(a_ptr, b_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    out = a + b
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def t1_matmul(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    pid_m = pid // tl.cdiv(N, BLOCK_SIZE_N)
    pid_n = pid % tl.cdiv(N, BLOCK_SIZE_N)
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator, input_precision="tf32")
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def t2_matmul_relu(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    pid_m = pid // tl.cdiv(N, BLOCK_SIZE_N)
    pid_n = pid % tl.cdiv(N, BLOCK_SIZE_N)
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator, input_precision="tf32")
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = tl.maximum(0.0, accumulator)
    c = c.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def t3_modulo(x_ptr, y_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)  # noqa: F841
    offs = tl.arange(0, BLOCK_SIZE)
    x_ptrs = x_ptr + (offs % N)
    x = tl.load(x_ptrs)
    tl.store(y_ptr + offs, x)



TIER_KERNELS = [
    (t0_vecadd, {'signature': {'a_ptr': '*fp32', 'b_ptr': '*fp32', 'out_ptr': '*fp32', 'n_elements': 'i32'}, 'constexprs': {'BLOCK_SIZE': 64}}),
    (t1_matmul, {'signature': {'a_ptr': '*fp32', 'b_ptr': '*fp32', 'c_ptr': '*fp32', 'M': 'i32', 'N': 'i32', 'K': 'i32', 'stride_am': 'i32', 'stride_ak': 'i32', 'stride_bk': 'i32', 'stride_bn': 'i32', 'stride_cm': 'i32', 'stride_cn': 'i32'}, 'constexprs': {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}}),
    (t2_matmul_relu, {'signature': {'a_ptr': '*fp32', 'b_ptr': '*fp32', 'c_ptr': '*fp32', 'M': 'i32', 'N': 'i32', 'K': 'i32', 'stride_am': 'i32', 'stride_ak': 'i32', 'stride_bk': 'i32', 'stride_bn': 'i32', 'stride_cm': 'i32', 'stride_cn': 'i32'}, 'constexprs': {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}}),
    (t3_modulo, {'signature': {'x_ptr': '*fp32', 'y_ptr': '*fp32', 'N': 'i32'}, 'constexprs': {'BLOCK_SIZE': 64}}),
]

def extract(out_dir, force=False):
    if not force and os.path.exists(os.path.join(out_dir, "t0_vecadd.ttir")):
        print("Use --force to overwrite")
        return
    os.makedirs(out_dir, exist_ok=True)
    TARGET = GPUTarget("cuda", 80, 32)
    
    # Keep track of ttgir for the snapshot
    ttgir_snapshot = []
    
    for kernel, kwargs in TIER_KERNELS:
        compiled = tt_compile(ASTSource(fn=kernel, signature=kwargs['signature'], constexprs=kwargs['constexprs']), target=TARGET)
        ttir = compiled.asm["ttir"]
        ttgir = compiled.asm["ttgir"]
        with open(os.path.join(out_dir, f"{kernel.__name__}.ttir"), "w") as f:
            f.write(ttir)
        ttgir_snapshot.append(f"--- {kernel.__name__} ---\n{ttgir}\n")
        
    with open(os.path.join(out_dir, "ttgir_snapshot.txt"), "w") as f:
        f.write("\n".join(ttgir_snapshot))
        
    with open(os.path.join(out_dir, "VERSIONS.txt"), "w") as f:
        f.write(f"triton=={triton.__version__}\n")
    print(f"Extracted fixtures to {out_dir}")
