"""Contract tests for emit.ir and emit.disasm using standard library unittest.

Validates Program construction, serialization/deserialization round-tripping,
loop region representation, and human-readable disassembly.
"""

from __future__ import annotations

import unittest

from triton_tritonflow.emit.disasm import deserialize, disassemble, serialize
from triton_tritonflow.emit.ir import (
    Imm,
    Instr,
    Loop,
    MemRef,
    Program,
    SourceRef,
    SsaRef,
)


class TestAssemblerContract(unittest.TestCase):
    """Real contract tests for the assembler and instruction stream."""

    def test_basic_program_round_trip(self) -> None:
        """Test serialize -> deserialize round trip for straight-line instructions."""
        p = Program(
            isa_name="tritonflow1",
            schema_version=1,
            kernel_name="test_vecadd",
            total_cost=64.0,
            inputs=("%x", "%y", "%out"),
            instrs=(
                Instr(
                    name="DMA1D",
                    cost=64.0,
                    loop=None,
                    defs=("%x_val",),
                    operands={"dst": SsaRef("%x_val"), "src": MemRef.of("global", "%x")},
                    constraint="in_bounds(base, length)",
                    constrained_on=None,
                    source_ops=(SourceRef("tt.load", 18, 4, "x_5"),),
                ),
            ),
        )

        ser = serialize(p)
        self.assertIn("PROGRAM", ser)
        self.assertIn("DMA1D", ser)

        p2 = deserialize(ser)
        self.assertEqual(p, p2)
        # Idempotence: serialize(deserialize(serialize(p))) == serialize(p)
        self.assertEqual(serialize(p2), ser)

    def test_program_with_loop_and_epilogue(self) -> None:
        """Test round trip for a program with a structured reduction loop and epilogue."""
        loop = Loop(
            id=0,
            induction_var="%k",
            lower=Imm(0),
            upper=Imm(32),
            step=Imm(1),
            iter_args=("%acc",),
            inits=("%c0",),
            results=("%res",),
            yields=("%acc_next",),
            body=(
                Instr(
                    name="MAC8",
                    cost=64.0,
                    loop=0,
                    defs=("%acc_next",),
                    operands={"acc": SsaRef("%acc"), "tile": Imm(8)},
                    constraint="aligned(base, 4)",
                    constrained_on=None,
                    source_ops=(SourceRef("tt.dot", 48, 8, "acc"),),
                ),
            ),
            source=SourceRef("scf.for", 45, 4, "loop"),
        )

        epilogue_instr = Instr(
            name="EPI",
            cost=32.0,
            loop=None,
            defs=("%final",),
            operands={"src": SsaRef("%res")},
            constraint="all_of(acc_dtype == op_dtype, in_bounds(base, length))",
            constrained_on=None,
            source_ops=(SourceRef("arith.addf", 65, 4, "c"),),
        )

        p = Program(
            isa_name="tritonflow1",
            schema_version=1,
            kernel_name="matmul_loop",
            total_cost=96.0,
            inputs=("%a", "%b", "%c0"),
            loops=(loop,),
            instrs=(),
            epilogue=(epilogue_instr,),
        )

        ser = serialize(p)
        p2 = deserialize(ser)
        self.assertEqual(p, p2)
        self.assertEqual(len(p2.loops), 1)
        self.assertEqual(len(p2.epilogue), 1)

    def test_disassembly_readable_format(self) -> None:
        """Test disassembly produces structured, annotated text."""
        p = Program(
            isa_name="tritonflow1",
            schema_version=1,
            kernel_name="test_disasm",
            total_cost=10.0,
            inputs=("%x",),
            instrs=(
                Instr(
                    name="DMA1D",
                    cost=10.0,
                    loop=None,
                    defs=("%val",),
                    operands={"src": SsaRef("%x")},
                    constraint=None,
                    constrained_on=None,
                    source_ops=(SourceRef("tt.load", 10, 2, "load_x"),),
                ),
            ),
        )
        dis = disassemble(p)
        self.assertIn("tritonflow1", dis)
        self.assertIn("DMA1D", dis)
        self.assertIn("load_x", dis)


if __name__ == "__main__":
    unittest.main()
