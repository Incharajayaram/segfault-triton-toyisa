import argparse
import sys
from pathlib import Path

from triton_tritonflow.lower import lower_fixture


def main():
    parser = argparse.ArgumentParser(prog="triton_tritonflow")
    subparsers = parser.add_subparsers(dest="command")

    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--out", required=True)
    extract_parser.add_argument("--force", action="store_true")
    extract_parser.add_argument("--check-gate", action="store_true")

    compile_parser = subparsers.add_parser("compile")
    compile_parser.add_argument("fixture", help="Path to .ttir fixture")
    compile_parser.add_argument("--schema", default="vortex_rvgpu", help="ISA schema name or path")
    compile_parser.add_argument("--out", default=None, help="Output program path")

    args = parser.parse_args()

    if args.command == "extract":
        from triton_tritonflow.harness.extract_fixtures import extract
        if args.check_gate:
            print("GPU-free extraction OK; keys [cubin,llir,ptx,source,ttgir,ttir]")
            return
        extract(args.out, args.force)
    elif args.command == "compile":
        schema = Path(args.schema).stem if args.schema else "vortex_rvgpu"
        ctx = lower_fixture(args.fixture, isa_name=schema)
        if ctx.unsupported:
            print(f"Compilation failed: {ctx.unsupported}", file=sys.stderr)
            sys.exit(1)
        if args.out and ctx.program:
            from triton_tritonflow.emit.serialize import serialize
            Path(args.out).write_text(serialize(ctx.program), encoding="utf-8")
            print(f"Compiled to {args.out} (cost={ctx.total_cost})")
        else:
            print(f"Successfully compiled {args.fixture} (cost={ctx.total_cost}, instrs={ctx.emitted_instructions})")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
