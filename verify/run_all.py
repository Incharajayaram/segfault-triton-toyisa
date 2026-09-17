#!/usr/bin/env python3
"""run_all.py — the CI gate. Runs every verify/*.py script, reports pass/fail
per script, exits non-zero if any failed. No framework, no config."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    scripts = sorted(HERE.glob("verify_*.py"))
    if not scripts:
        print("no verify scripts found")
        return 1
    failures: list[str] = []
    print(f"running {len(scripts)} verification scripts\n")
    for script in scripts:
        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True, text=True, timeout=600,
        )
        dt = time.perf_counter() - t0
        status = "PASS" if proc.returncode == 0 else "FAIL"
        print(f"[{status}] {script.name}  ({dt:.2f}s, exit={proc.returncode})")
        if proc.returncode != 0:
            failures.append(script.name)
            tail = (proc.stdout + proc.stderr).strip().splitlines()
            for line in tail[-12:]:
                print(f"    {line}")
    print()
    if failures:
        print(f"FAILED: {len(failures)} script(s): {failures}")
        return 1
    print(f"ALL {len(scripts)} SCRIPTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
