#!/usr/bin/env python3
"""Standalone test runner for triton_tritonflow.

Runs all tests using Python standard library unittest.
Zero dependency on pytest or third-party test frameworks.

Usage:
    python3 run_tests.py                 # run all tests
    python3 run_tests.py --suite contract # run contract tests
    python3 run_tests.py --suite unit     # run unit tests
    python3 run_tests.py -v              # verbose output
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
TOOLS_DIR = ROOT / "tools"
TESTS_DIR = ROOT / "tests"

# Ensure src and tools are on sys.path
for p in (str(SRC_DIR), str(TOOLS_DIR), str(TESTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)


def build_suite(suite_name: str | None = None, pattern: str = "test_*.py") -> unittest.TestSuite:
    loader = unittest.defaultTestLoader
    suite = unittest.TestSuite()

    if suite_name == "unit":
        start_dir = str(TESTS_DIR / "unit")
    elif suite_name == "contract":
        start_dir = str(TESTS_DIR / "contract")
    elif suite_name == "integration":
        start_dir = str(TESTS_DIR / "integration")
    elif suite_name == "e2e":
        start_dir = str(TESTS_DIR / "e2e")
    else:
        start_dir = str(TESTS_DIR)

    if os.path.exists(start_dir):
        discovered = loader.discover(start_dir=start_dir, pattern=pattern, top_level_dir=str(ROOT))
        suite.addTest(discovered)

    return suite


def main() -> int:
    parser = argparse.ArgumentParser(description="triton_tritonflow test runner (unittest)")
    parser.add_argument("--suite", choices=["all", "unit", "contract", "integration", "e2e"], default="all")
    parser.add_argument("-v", "--verbose", action="store_true", default=True, help="verbose output")
    parser.add_argument("-k", "--pattern", default="test_*.py", help="test file pattern")
    args = parser.parse_args()

    suite_choice = None if args.suite == "all" else args.suite
    suite = build_suite(suite_choice, pattern=args.pattern)

    count = suite.countTestCases()
    print("============================================================")
    print(f"  triton_tritonflow test runner (unittest) — {count} test cases")
    print(f"  Suite: {args.suite} | Pattern: {args.pattern}")
    print("============================================================")

    start_time = time.perf_counter()
    runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
    result = runner.run(suite)
    elapsed = time.perf_counter() - start_time

    print("\n------------------------------------------------------------")
    print(f"Ran {result.testsRun} tests in {elapsed:.3f}s")
    print(f"  Passed:   {result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)}")
    print(f"  Failed:   {len(result.failures)}")
    print(f"  Errors:   {len(result.errors)}")
    print(f"  Skipped:  {len(result.skipped)}")
    print("------------------------------------------------------------")

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
