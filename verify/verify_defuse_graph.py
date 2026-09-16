"""Show that a def-use graph over ttir is obtainable with ZERO extra build
burden, using only the MLIR bindings already inside the `triton` wheel.

This is the strongest test of methodology section 4.1/4.2, which asserts a
hand-rolled text parser is "preferable to depending on MLIR Python bindings,
which would reintroduce the exact build-from-source burden".
"""

from triton._C.libtriton import ir

ctx = ir.context()
ir.load_dialects(ctx)
mod = ir.parse_mlir_module("/tmp/T1_matmul.ttir", ctx)
print("parsed       :", type(mod).__name__)

top = mod.get_operation()
print("entry func   :", mod.get_entry_func_name())
print("top oplist   :", [op.name for op in top.walk()][:4])

print("\n--- walk: tt / scf ops of interest ---")
n = 0
for op in top.walk():
    name = getattr(op, "name", None)
    if not name:
        continue
    n += 1
    if name in ("tt.dot", "tt.load", "tt.store", "scf.for", "scf.yield",
                "tt.addptr", "tt.make_range", "tt.splat", "tt.expand_dims",
                "arith.remsi"):
        print(f"  {name:<16} results={len(op.results)} "
              f"operands={len(op.operands)}")
print(f"  total ops walked: {n}")

print("\n--- def-use graph (tt.dot operands -> defining op) ---")
for op in top.walk():
    if getattr(op, "name", None) == "tt.dot":
        for i, operand in enumerate(op.operands):
            owner = operand.get_owner() if hasattr(operand, "get_owner") else None
            loc = owner.location if owner is not None else None
            print(f"  tt.dot operand[{i}] <- {getattr(owner, 'name', '?')}"
                  f"   loc={loc}")
        break

print("\n--- location info on tt.load / tt.store (methodology 4.2) ---")
for op in top.walk():
    if getattr(op, "name", None) in ("tt.store", "tt.load"):
        print(f"  {op.name:<9} loc = {op.location}")
