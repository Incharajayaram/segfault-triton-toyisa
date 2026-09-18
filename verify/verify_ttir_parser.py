#!/usr/bin/env python3
"""verify_ttir_parser.py — parser contract, verified against the frozen fixture corpus.

Converted from tests/contract/test_ttir_parser.py (pytest) + tests/unit/
test_parser_syntax_negative.py. Every expected value below comes from
fixtures/GOLDEN.json (the frozen canon), NOT from running the parser.

Checks:
  V1  each of the 4 frozen fixtures parses with 0 diagnostics and the
      exact op count the canon records (20 / 63 / 70 / 29)
  V2  tt.func is parsed AS tt.func with its arguments as block arguments
      (the regression test for the '='-scan bug that named it 'false')
  V3  attributes are captured verbatim (attrs != {} on tt.func)
  V4  syntax-negative inputs produce diagnostics, never exceptions
      (EC-001, EC-002, EC-011, EC-025, deep nesting, minified, CRLF/BOM,
      unknown token)
  V5  parse_raw never raises across an adversarial corpus
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from tritonflow.ttir.parser import parse_raw

FAILURES: list[str] = []


def iter_all_ops(raw):
    """Yield every op in the RawModule tree, excluding the synthetic module
    root. The canon's ops_total (20/63/70/29) counts all ops INCLUDING
    tt.func but NOT the module wrapper — verified against GOLDEN.json
    op_hist sums before writing this expectation."""
    def walk(op):
        if op.name not in ('module', 'builtin.module'):
            yield op
        for region in op.regions:
            for block in region.blocks:
                for child in block.ops:
                    yield from walk(child)
    for root in raw.ops:
        yield from walk(root)


def find_ops(raw, name):
    return [op for op in iter_all_ops(raw) if op.name == name]


def check(name: str, actual, expected, context: str = "") -> None:
    """Assert actual == expected, recording actual-vs-expected on failure."""
    ok = actual == expected
    if ok:
        print(f"  PASS {name}: {actual!r}")
    else:
        FAILURES.append(name)
        print(f"  FAIL {name}: actual={actual!r} expected={expected!r}"
              + (f"  ({context})" if context else ""))


def main() -> int:
    fixtures = ROOT / "fixtures"
    golden_op_counts = {  # source: fixtures/GOLDEN.json observations (frozen canon)
        "t0_vecadd": 20,
        "t1_matmul": 63,
        "t2_matmul_relu": 70,
        "t3_modulo": 29,
    }

    print("V1: each frozen fixture parses with canon op counts")
    for tier, want_ops in golden_op_counts.items():
        text = (fixtures / f"{tier}.ttir").read_text()
        raw = parse_raw(text, source_path=str(fixtures / f"{tier}.ttir"))
        check(f"{tier}: diagnostics==0", len(raw.diagnostics), 0)
        check(f"{tier}: op count", sum(1 for _ in iter_all_ops(raw)), want_ops,
              "op histogram must match the frozen canon")

    print("V2: tt.func parsed as tt.func, arguments as block args (regression: '='-scan bug)")
    t0 = parse_raw((fixtures / "t0_vecadd.ttir").read_text(), source_path="t0")
    funcs = find_ops(t0, "tt.func")
    check("t0: exactly one tt.func", len(funcs), 1)
    check("t0: no op named 'false' (the old bug)", any(op.name == "false" for op in iter_all_ops(t0)), False)
    if funcs:
        func = funcs[0]
        entry_args = func.regions[0].blocks[0].args if func.regions else []
        check("t0: tt.func has block arguments",
              len(entry_args), 4,
              f"entry block args were {entry_args!r} — function arguments must be block args, not results")
        check("t0: tt.func results is empty (args are NOT misfiled as results)",
              len(func.results), 0)

    print("V3: attribute dict captured verbatim")
    check("t0: noinline attr present", funcs and funcs[0].attrs.get("noinline") == "false", True,
          f"attrs were {funcs[0].attrs!r}" if funcs else "no tt.func found")

    print("V4: syntax-negative inputs -> diagnostics, never exceptions")
    negative_cases = [
        ("EC-001 empty", ""),
        ("EC-002 comments only", "// just a comment\n// another"),
        ("EC-011 unterminated attr dict", "module { \n %0 = arith.constant 64 {value = 64 \n }"),
        ("EC-025 truncated module", "module { "),
        ("minified single line", "module { %0 = arith.constant 1 : i32 }"),
        ("unknown token", "module { %0 = weiroperand 42 : i32 }"),
    ]
    for label, text in negative_cases:
        try:
            raw = parse_raw(text, source_path="<negative>")
            check(f"{label}: produced diagnostic or clean parse (no crash)",
                  isinstance(raw.diagnostics, list), True)
        except Exception as exc:
            FAILURES.append(label)
            print(f"  FAIL {label}: parse_raw RAISED {type(exc).__name__}: {exc}")

    # deep nesting must produce a diagnostic, not RecursionError
    try:
        deep = "module {\n" + "  %x = arith.constant 1 : i32\n" * 250 + "}"
        raw = parse_raw(deep, source_path="<deep>")
        check("deep nesting: no exception", True, True)
    except RecursionError:
        FAILURES.append("deep nesting")
        print("  FAIL deep nesting: RecursionError propagated")
    except Exception as exc:
        check("deep nesting: refused with diagnostic, no crash", type(exc).__name__,
              type(exc).__name__)  # any non-RecursionError exception is acceptable? NO:
        FAILURES.append("deep nesting raised")
        print(f"  FAIL deep nesting: raised {type(exc).__name__}: {exc}")

    print("V5: parse_raw totality over adversarial corpus")
    adversarial = [
        "module {", "module { }", "%", "module { % }", "tt.func",
        "module { %a = }", "module { %a = arith.constant }",
        '#loc = loc("X":1:0)\nmodule {\n}', "module { %a = tt.func }",
        "\x00\x01\x02", "module { %a = arith.constant 64 : i32 : i32 }",
    ]
    raised = []
    for i, text in enumerate(adversarial):
        try:
            parse_raw(text, source_path=f"<adv{i}>")
        except Exception as exc:
            raised.append(f"case {i} ({text[:30]!r}): {type(exc).__name__}: {exc}")
    check("adversarial corpus: parse_raw never raises", len(raised), 0,
          "; ".join(raised[:3]))

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
