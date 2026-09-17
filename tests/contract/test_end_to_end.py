"""Contract tests for the end-to-end lowering pipeline.

Each case drives `lower_fixture`, which runs the production chain
(`parse_module -> build_def_use -> annotate -> assemble`) with the real schema
and the real selector. Nothing here asserts an instruction *name* the pipeline
was told to emit: the point is that the selector chose it from the schema's
candidate set, so the assertions are about coverage, refusal and numerical
parity.

The Tier 3 case is the corpus's negative control and earns its own note. An
earlier version of this file asserted `ctx.unsupported == []` for it -- encoding
a miscompile as the expected behaviour, because the lowering it tested was a
hardcoded table that emitted a `VMOD` instruction for a kernel whose modulo
wraparound makes its memory access unstructured. It must be refused, with the
originating name attached.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from triton_toyisa.isa.schema import load_builtin, validate_schema
from triton_toyisa.lower import lower_fixture

#: Every ISA the corpus is lowered against. `lower_fixture` takes the name and
#: changes nothing else, which is the cross-ISA transfer claim in one line.
ISAS = ("toyisa1", "toyisa2", "vortex_rvgpu")

#: Tiers expected to lower completely, and the one expected to be refused.
LOWERABLE = ("t0_vecadd", "t1_matmul", "t2_matmul_relu")
REFUSED = "t3_modulo"

#: Execution is verified on the two ISAs whose memory instructions are
#: direction-agnostic. `vortex_rvgpu` declares directional LDG/STG with identical
#: rule, constraint and cost; the selector has no notion of direction, so it
#: resolves the tie by declaration order and picks LDG for stores as well as
#: loads. Its *lowering* is asserted below; its execution is a known gap,
#: recorded in KNOWN_GAPS.md rather than hidden behind a skip with no reason.
EXECUTABLE_ISAS = ("toyisa1", "toyisa2")


class TestSchemas(unittest.TestCase):
    def test_every_schema_loads_and_validates(self) -> None:
        for name in ISAS:
            with self.subTest(isa=name):
                schema = load_builtin(name)
                self.assertEqual(schema.name, name)
                self.assertEqual(validate_schema(schema), [])

    def test_vortex_declares_its_instruction_set(self) -> None:
        schema = load_builtin("vortex_rvgpu")
        expected = {
            "LDG", "STG", "LDS", "STS",
            "TCU_MMA16", "TCU_MMA32",
            "VADD", "VMUL", "VSUB", "VMOD", "VRELU", "VCLAMP",
            "BARRIER",
        }
        self.assertTrue(expected.issubset(set(schema.instructions)))

    def test_barrier_is_not_a_memory_candidate(self) -> None:
        """A fence must not compete with loads and stores for a memory operand.

        BARRIER was declared `kind: memory` with a trivially-true constraint and
        a flat 0.01 cost, so it won every memory selection on minimum cost and
        each load and store lowered to a fence.
        """
        schema = load_builtin("vortex_rvgpu")
        barrier = schema.instructions["BARRIER"]
        self.assertEqual(barrier.kind, "control")
        self.assertNotEqual(barrier.rule, "memory")

    def test_scratchpad_moves_are_not_global_lowerings(self) -> None:
        """LDS/STS read and write scratch; they cannot lower a global access.

        Declared `rule: memory` they undercut LDG/STG on cost and were selected
        for global loads, producing a program that read a scratchpad nothing had
        filled.
        """
        schema = load_builtin("vortex_rvgpu")
        for name in ("LDS", "STS"):
            with self.subTest(instruction=name):
                self.assertNotEqual(schema.instructions[name].rule, "memory")


class TestDirectionalSelection(unittest.TestCase):
    """An ISA that splits loads from stores must have the right one chosen.

    `vortex_rvgpu` declares LDG and STG identically in every field selection
    reads, so before `direction` existed the minimum-cost rule broke the tie by
    declaration order and picked LDG for stores as well as loads -- and the
    emulator executed a store as a load.
    """

    def test_load_and_store_get_different_instructions(self) -> None:
        from triton_toyisa.isa.schema import load_builtin
        from triton_toyisa.lower import assemble_program, launch_env
        from triton_toyisa.ttir.graph import build_def_use
        from triton_toyisa.ttir.to_ir import parse_module

        text = (Path(__file__).resolve().parents[2] / "fixtures" / "t0_vecadd.ttir").read_text()
        module = parse_module(text).module
        graph = build_def_use(module)
        program = assemble_program(
            module, graph, load_builtin("vortex_rvgpu"), launch_env("t0_vecadd"), []
        )
        by_source = {}
        for instr in program.instructions():
            for source in instr.source_ops:
                if source.op_name in ("tt.load", "tt.store"):
                    by_source.setdefault(source.op_name, set()).add(instr.name)
        self.assertEqual(by_source.get("tt.load"), {"LDG"})
        self.assertEqual(by_source.get("tt.store"), {"STG"})

    def test_direction_agnostic_isa_is_unaffected(self) -> None:
        """toyisa1 declares no direction, so one instruction still serves both."""
        from triton_toyisa.isa.schema import load_builtin

        for name, instruction in load_builtin("toyisa1").instructions.items():
            with self.subTest(instruction=name):
                self.assertIsNone(instruction.direction)
                self.assertTrue(instruction.serves("load"))
                self.assertTrue(instruction.serves("store"))

    def test_direction_mismatch_is_recorded_as_a_rejection(self) -> None:
        """Not pre-filtered: the audit trail must say why STG lost a load."""
        from triton_toyisa.isa.schema import load_builtin
        from triton_toyisa.isa.select import enumerate_candidates

        candidates = enumerate_candidates(
            load_builtin("vortex_rvgpu"), "memory", None, None, {}, "load"
        )
        rejected = {
            c.instruction.name: c.rejected_by for c in candidates if not c.admissible
        }
        self.assertIn("STG", rejected)
        self.assertIn("direction", rejected["STG"])


class TestLowering(unittest.TestCase):
    def test_lowerable_tiers_lower_on_every_isa(self) -> None:
        for isa in ISAS:
            for tier in LOWERABLE:
                with self.subTest(isa=isa, tier=tier):
                    ctx = lower_fixture(tier, isa_name=isa)
                    self.assertEqual(ctx.unsupported, [], f"{tier} on {isa} was refused")
                    self.assertTrue(ctx.fully_lowered)
                    self.assertGreater(ctx.emitted_instructions, 0)
                    self.assertGreater(ctx.total_cost, 0.0)

    def test_selection_differs_across_isas(self) -> None:
        """Transfer is a re-selection, not a rename.

        If two ISAs produced the same instruction names at the same cost, the
        schema would not be driving anything -- which is exactly the failure the
        previous `"LDG" if isa_name == "vortex_rvgpu" else "DMA1D"` ternary hid.
        """
        chosen = {}
        for isa in ISAS:
            ctx = lower_fixture("t1_matmul", isa_name=isa)
            chosen[isa] = (
                frozenset(instr.name for instr in ctx.program.instructions()),
                round(ctx.total_cost, 3),
            )
        self.assertEqual(len(set(chosen.values())), len(ISAS), chosen)

    def test_tier3_is_refused_on_every_isa(self) -> None:
        """The negative control. Refused, with the originating name attached."""
        for isa in ISAS:
            with self.subTest(isa=isa):
                ctx = lower_fixture(REFUSED, isa_name=isa)
                self.assertFalse(ctx.fully_lowered)
                self.assertTrue(ctx.unsupported, "the modulo kernel was not refused")
                joined = " ".join(ctx.unsupported)
                self.assertIn("UNSUPPORTED", joined)
                self.assertIn("modulo wraparound", joined)
                self.assertIsNone(ctx.emu_outputs, "a refused program was executed anyway")

    def test_unknown_fixture_is_a_diagnostic_not_a_crash(self) -> None:
        ctx = lower_fixture("does_not_exist", isa_name="toyisa1")
        self.assertTrue(ctx.unsupported)
        self.assertFalse(ctx.fully_lowered)


class TestNumericalParity(unittest.TestCase):
    def test_vecadd_is_exact(self) -> None:
        for isa in EXECUTABLE_ISAS:
            with self.subTest(isa=isa):
                ctx = lower_fixture("t0_vecadd", isa_name=isa)
                self.assertEqual(ctx.parity_max_rel_err, 0.0)

    def test_matmul_is_within_the_derived_tolerance(self) -> None:
        for isa in EXECUTABLE_ISAS:
            with self.subTest(isa=isa):
                ctx = lower_fixture("t1_matmul", isa_name=isa)
                self.assertIsNotNone(ctx.parity_max_rel_err)
                self.assertLessEqual(ctx.parity_max_rel_err, ctx.tolerance)

    def test_tolerance_is_derived_from_the_reduction_length(self) -> None:
        """Not a hardcoded epsilon: K * (2*2^-11 + 2^-24) for the tf32 path."""
        ctx = lower_fixture("t1_matmul", isa_name="toyisa1")
        expected = 64 * (2 * 2.0**-11 + 2.0**-24)
        self.assertAlmostEqual(ctx.tolerance, expected, places=9)

    def test_hardware_stats_come_from_the_emitted_stream(self) -> None:
        ctx = lower_fixture("t0_vecadd", isa_name="toyisa1")
        self.assertIsNotNone(ctx.hardware_stats)
        self.assertEqual(
            ctx.hardware_stats.instructions_executed, ctx.emitted_instructions
        )
        self.assertGreater(ctx.hardware_stats.dram_bytes_transacted, 0)


class TestReluEpilogue(unittest.TestCase):
    def test_relu_output_is_non_negative(self) -> None:
        """Tier 2's epilogue must clamp at zero wherever it is exercised."""
        ctx = lower_fixture("t2_matmul_relu", isa_name="toyisa1")
        self.assertEqual(ctx.unsupported, [])
        self.assertIsNotNone(ctx.emu_outputs)
        self.assertTrue(np.all(ctx.emu_outputs["out"] >= 0.0))


if __name__ == "__main__":
    unittest.main()
