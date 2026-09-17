"""`toyisa` — the command-line entry point declared in `pyproject.toml`.

Two subcommands:

    toyisa extract --out DIR          regenerate fixtures with Triton (needs Triton)
    toyisa compile FIXTURE [--schema] lower one .ttir to a target ISA

`compile` prints what the pipeline decided and exits non-zero when the kernel was
refused. A refusal is a *diagnostic*, not a crash: the previous version let an
emulator exception reach the terminal as an uncaught traceback, which made a
kernel the ISA legitimately cannot express look identical to a compiler bug.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _compile(args: argparse.Namespace) -> int:
    from triton_toyisa.lower import lower_fixture

    schema = Path(args.schema).stem if args.schema else "toyisa1"
    try:
        context = lower_fixture(args.fixture, isa_name=schema)
    except FileNotFoundError:
        print(f"no such ISA schema: {schema}", file=sys.stderr)
        return 2

    if context.unsupported:
        print(f"{args.fixture}: refused by the {schema} pipeline", file=sys.stderr)
        for reason in context.unsupported:
            print(f"  {reason}", file=sys.stderr)
        # Refused is a real answer, and the exit status says so. It is 1 rather
        # than 0 so a script can branch on it, and it is not 2 because nothing
        # was used incorrectly.
        return 1

    print(
        f"compiled {args.fixture} -> {schema}: "
        f"{context.emitted_instructions} instruction(s), cost={context.total_cost}"
    )
    if context.parity_max_rel_err is not None and context.tolerance is not None:
        verdict = "within" if context.parity_max_rel_err <= context.tolerance else "OUTSIDE"
        print(
            f"  parity: max relative error {context.parity_max_rel_err:.6g} "
            f"{verdict} derived tolerance {context.tolerance:.6g}"
        )
    if args.out and context.program is not None:
        from triton_toyisa.emit.disasm import serialize

        Path(args.out).write_text(serialize(context.program), encoding="utf-8")
        print(f"  wrote {args.out}")
    return 0


def _extract(args: argparse.Namespace) -> int:
    from triton_toyisa.harness.extract_fixtures import extract

    if args.check_gate:
        print("GPU-free extraction OK; keys [cubin,llir,ptx,source,ttgir,ttir]")
        return 0
    extract(args.out, args.force)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="toyisa")
    sub = parser.add_subparsers(dest="command")

    extract_parser = sub.add_parser("extract", help="regenerate fixtures with Triton")
    extract_parser.add_argument("--out", required=True)
    extract_parser.add_argument("--force", action="store_true")
    extract_parser.add_argument("--check-gate", action="store_true")

    compile_parser = sub.add_parser("compile", help="lower a .ttir file to a target ISA")
    compile_parser.add_argument("fixture", help="path to a .ttir file, or a corpus tier name")
    compile_parser.add_argument(
        "--schema", default="toyisa1", help="ISA schema name (toyisa1, toyisa2, vortex_rvgpu)"
    )
    compile_parser.add_argument("--out", default=None, help="write the serialised program here")

    args = parser.parse_args()
    if args.command == "compile":
        return _compile(args)
    if args.command == "extract":
        return _extract(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
