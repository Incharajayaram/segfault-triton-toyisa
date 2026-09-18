"""Contract tests for ttir.parser using standard library unittest.

Validates text-level parsing of raw Triton IR fixtures into RawModule.
Zero dependency on pytest.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from tritonflow.ttir.parser import parse_raw

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures"


class TestTtirParserContract(unittest.TestCase):
    """Contract tests for raw TTIR parsing."""

    def _test_fixture(self, filename: str) -> None:
        filepath = FIXTURES_DIR / filename
        if not filepath.exists():
            self.skipTest(f"Fixture {filename} not found.")

        text = filepath.read_text()
        raw_module = parse_raw(text, source_path=str(filepath))

        # Parser should never crash
        self.assertIsNotNone(raw_module)
        # Should have parsed operations
        self.assertGreater(len(raw_module.ops), 0, f"No ops parsed from {filename}")
        # Should have parsed location table
        self.assertGreater(len(raw_module.loc_table), 0, f"No loc table in {filename}")

    def test_parse_t0_vecadd(self) -> None:
        self._test_fixture("t0_vecadd.ttir")

    def test_parse_t1_matmul(self) -> None:
        self._test_fixture("t1_matmul.ttir")

    def test_parse_t2_matmul_relu(self) -> None:
        self._test_fixture("t2_matmul_relu.ttir")

    def test_parse_t3_modulo(self) -> None:
        self._test_fixture("t3_modulo.ttir")


if __name__ == "__main__":
    unittest.main()
