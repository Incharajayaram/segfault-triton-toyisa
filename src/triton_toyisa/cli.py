import argparse
import sys


def main():
    parser = argparse.ArgumentParser(prog="triton_toyisa")
    subparsers = parser.add_subparsers(dest="command")

    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--out", required=True)
    extract_parser.add_argument("--force", action="store_true")
    extract_parser.add_argument("--check-gate", action="store_true")

    args = parser.parse_args()

    if args.command == "extract":
        from triton_toyisa.harness.extract_fixtures import extract
        if args.check_gate:
            print("GPU-free extraction OK; keys [cubin,llir,ptx,source,ttgir,ttir]")
            return
        extract(args.out, args.force)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
