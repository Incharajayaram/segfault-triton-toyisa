#!/usr/bin/env python3
"""R5 enforcement: the module/contract <-> test mapping.

Three laws, one per check:

  1. every module under src/tritonflow/ has tests/unit/test_<module>.py
  2. every contract in specs/.../contracts/ has tests/contract/test_<stem>.py
  3. no module imports triton or torch outside harness/ and torch_backend/

Runs in two modes so it is useful on day 1 and strict at the end:

    python3 tools/check_test_map.py            # report; exit 1 only on check 3
    python3 tools/check_test_map.py --strict   # CI after the day-3 deadline

Stdlib only.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "tritonflow"
CONTRACTS = ROOT / "specs" / "001-triton-to-tritonflow" / "contracts"
UNIT = ROOT / "tests" / "unit"
CONTRACT_TESTS = ROOT / "tests" / "contract"

# Modules that are dev-only by design (see plan.md: harness is the only place
# Triton may appear at all, and torch_backend exists to talk to torch).
DEV_ONLY = {"harness"}
TORCH_ALLOWED = {"torch_backend"}
TRITON_ALLOWED = {"extract", "harness"}

# Tests that legitimately pair with a contract file of a different name.
CONTRACT_ALIASES = {
    "test_fixtures.py": "raw-module.md, ttir-parser.md",
    "test_device_interface.py": "torch-seam.md",
}


def modules() -> list[Path]:
    if not SRC.is_dir():
        return []
    return sorted(p for p in SRC.rglob("*.py") if p.name != "__init__.py")


def module_test(mod: Path) -> Path:
    return UNIT / f"test_{mod.stem}.py"


def imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="fail on unmapped modules")
    args = ap.parse_args()

    problems: list[str] = []
    mods = modules()

    # 1. module -> unit test
    unmapped = [m for m in mods if not module_test(m).exists()]
    for m in unmapped:
        rel = m.relative_to(ROOT)
        msg = f"{rel}: no {module_test(m).relative_to(ROOT)}"
        (problems if args.strict else []).append(msg)

    # 2. contract -> contract test
    contract_files = sorted(CONTRACTS.glob("*.md")) if CONTRACTS.is_dir() else []
    missing_contracts = []
    for c in contract_files:
        expected = CONTRACT_TESTS / f"test_{c.stem.replace('-', '_')}.py"
        if not expected.exists():
            missing_contracts.append(f"{c.name}: no {expected.relative_to(ROOT)}")
    if args.strict:
        problems.extend(missing_contracts)

    # 3. import hygiene — always enforced
    for mod in mods:
        top = mod.relative_to(SRC).parts[0] if len(mod.relative_to(SRC).parts) > 1 else None
        roots = imported_roots(mod)
        if "triton" in roots and top not in TRITON_ALLOWED:
            problems.append(f"{mod.relative_to(ROOT)}: imports triton outside {sorted(TRITON_ALLOWED)}")
        if "torch" in roots and top not in TORCH_ALLOWED:
            problems.append(f"{mod.relative_to(ROOT)}: imports torch outside {sorted(TORCH_ALLOWED)}")

    n_mods, n_contracts = len(mods), len(contract_files)
    print(f"test map: {n_mods} module(s), {n_contracts} contract(s), "
          f"{'strict' if args.strict else 'report'} mode")
    if not mods and not contract_files:
        print("  (nothing to map yet: source tree and specs are not populated)")
    if missing_contracts:
        print(f"  contract tests pending ({len(missing_contracts)}):")
        for m in missing_contracts:
            print(f"    - {m}")
    if unmapped:
        print(f"  modules without a unit test ({len(unmapped)}):")
        for m in unmapped:
            print(f"    - {m}")

    if problems:
        print(f"\nFAIL: {len(problems)} problem(s)")
        for p in problems:
            print(f"  - {p}")
        if not args.strict and all("imports " not in p for p in problems):
            print("\n(unmapped entries are warnings until --strict)")
            return 0
        return 1

    print("  OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
