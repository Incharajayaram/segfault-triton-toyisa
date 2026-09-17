"""Contract tests for RawModule (Track A/B seam) using standard library unittest.

Validates the parser / IR-model seam specified in raw-module.md:
types stay strings, attributes stay strings, SSA names stay strings.
"""

from __future__ import annotations

import unittest

from triton_toyisa.ttir.parser import (
    ParseDiagnostic,
    RawBlock,
    RawLoc,
    RawModule,
    RawOp,
    RawRegion,
    parse_raw,
)


class TestRawModuleContract(unittest.TestCase):
    """Contract tests for RawModule structure and parser output invariants."""

    def test_raw_structures_field_types(self) -> None:
        """RawOp must store text-level representations without semantic parsing."""
        loc = RawLoc(name="#loc1", line=10, col=4)
        op = RawOp(
            name="arith.constant",
            results=["%c64_i32"],
            operands=[],
            attrs={"value": "64 : i32"},
            result_types=["i32"],
            operand_types=[],
            regions=[],
            loc=loc,
            line=10,
            col=4,
        )
        block = RawBlock(args=[("%arg0", "!tt.ptr<f32>")], ops=[op], terminator_index=0)
        region = RawRegion(blocks=[block])
        raw_mod = RawModule(
            ops=[op],
            loc_table={"#loc1": loc},
            source_path="<test>",
            triton_version="3.7.1",
            diagnostics=[],
        )

        # Invariants from raw-module.md:
        self.assertEqual(op.name, "arith.constant")
        self.assertIsInstance(op.results[0], str)
        self.assertIsInstance(op.attrs["value"], str)
        self.assertIsInstance(op.result_types[0], str)
        self.assertEqual(len(raw_mod.diagnostics), 0)

    def test_parse_diagnostic_layer_tagging(self) -> None:
        """ParseDiagnostic must identify the faulting layer ('syntax' vs 'ir')."""
        diag = ParseDiagnostic(
            kind="PARSE_UNSUPPORTED",
            line=5,
            col=10,
            expected="operation name",
            found="@#$",
            snippet="unknown token",
            layer="syntax",
        )
        self.assertEqual(diag.layer, "syntax")
        self.assertEqual(diag.kind, "PARSE_UNSUPPORTED")

    def test_parse_raw_produces_raw_module(self) -> None:
        """parse_raw must always return a RawModule with diagnostics rather than raising."""
        # Empty input
        raw = parse_raw("")
        self.assertIsInstance(raw, RawModule)
        self.assertEqual(len(raw.ops), 0)
        self.assertEqual(len(raw.diagnostics), 0)

        # Bad syntax should record diagnostic with layer='syntax'
        bad = parse_raw("module { %0 = }")
        self.assertIsInstance(bad, RawModule)
        self.assertGreater(len(bad.diagnostics), 0)
        self.assertEqual(bad.diagnostics[0].layer, "syntax")


if __name__ == "__main__":
    unittest.main()
