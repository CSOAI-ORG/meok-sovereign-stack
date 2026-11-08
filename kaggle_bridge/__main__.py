"""Offline CLI. Commands write files only where explicitly requested."""
import argparse
import json
from pathlib import Path
import sys

from .artifacts import contract_from_source, strict_json, verify_smoke
from .coverage import build_coverage


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    contract = sub.add_parser("contract")
    contract.add_argument("--source", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--run", type=Path, required=True)
    verify.add_argument("--contract", type=Path, required=True)
    coverage = sub.add_parser("coverage")
    coverage.add_argument("--board", type=Path, required=True)
    coverage.add_argument("--observed-at", required=True)
    for cmd in (contract, verify, coverage):
        cmd.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "contract":
            result = contract_from_source(args.source)
        elif args.command == "verify":
            result = verify_smoke(args.run, strict_json(args.contract.read_bytes()))
        else:
            result = build_coverage(args.board.read_bytes(), observed_at=args.observed_at)
        text = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            # Exclusive create prevents overwriting retained evidence.
            with args.output.open("x") as out:
                out.write(text)
        else:
            print(text, end="")
        return 0
    except (ValueError, OSError, KeyError, TypeError, SyntaxError) as error:
        print(json.dumps({"status": "UNCHECKABLE", "error_type": type(error).__name__,
                          "reason": str(error), "published": False}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
