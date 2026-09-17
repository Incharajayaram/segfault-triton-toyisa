"""Unit tests for parser negative/edge cases using standard library unittest."""

from __future__ import annotations

import unittest

from triton_toyisa.ttir.parser import parse_raw


class TestParserSyntaxNegative(unittest.TestCase):
    """Negative syntax tests for parse_raw."""

    def test_empty_input(self) -> None:
        # EC-001
        module = parse_raw("")
        self.assertEqual(len(module.diagnostics), 0)

    def test_comments_only(self) -> None:
        # EC-002
        module = parse_raw("// just a comment\n// another")
        self.assertEqual(len(module.diagnostics), 0)

    def test_unterminated_attribute_dict(self) -> None:
        # EC-011 vs EC-025
        text = "module { \n %0 = arith.constant 64 {value = 64 \n }"
        module = parse_raw(text)
        self.assertGreater(len(module.diagnostics), 0)
        self.assertEqual(module.diagnostics[0].layer, "syntax")

    def test_truncated_module(self) -> None:
        # EC-025
        text = "module { "
        module = parse_raw(text)
        self.assertGreater(len(module.diagnostics), 0)
        self.assertIn("eof", str(module.diagnostics[0].found).lower())

    def test_deep_nesting(self) -> None:
        # EC-027
        text = "module { \n"
        for _i in range(250):
            text += "  scf.if %cond {\n"
        text += "    tt.return\n"
        for _i in range(250):
            text += "  }\n"
        text += "}\n"

        module = parse_raw(text)
        self.assertGreater(len(module.diagnostics), 0)
        self.assertEqual(module.diagnostics[0].layer, "syntax")

    def test_crlf_bom_handling(self) -> None:
        # EC-019, EC-020: the lexer strips a UTF-8 BOM read as text and
        # handles CRLF line endings
        text = "\ufeffmodule {\r\n  tt.return\r\n}\r\n"
        module = parse_raw(text)
        self.assertEqual(len(module.diagnostics), 0)

    def test_minified_single_line(self) -> None:
        # EC-021
        text = "module{%0=arith.constant 64:i32 tt.return}"
        module = parse_raw(text)
        self.assertIsNotNone(module)

    def test_unknown_token(self) -> None:
        text = "module { \n %0 = @#$%^ \n }"
        module = parse_raw(text)
        self.assertGreater(len(module.diagnostics), 0)
        self.assertEqual(module.diagnostics[0].layer, "syntax")


if __name__ == "__main__":
    unittest.main()
