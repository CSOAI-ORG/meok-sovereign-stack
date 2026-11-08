"""Offline CLI. Commands write files only where explicitly requested."""
import argparse
import json
from pathlib import Path
import sys

from .artifacts import contract_from_source, strict_json, verify_smoke
from .coverage import build_coverage
from .reload import contract_from_replay_source, verify_reload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    contract = sub.add_parser("contract")
    contract.add_argument("--source", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--run", type=Path, required=True)
    verify.add_argument("--contract", type=Path, required=True)
    replay_contract = sub.add_parser("reload-contract")
    replay_contract.add_argument("--source", type=Path, required=True)
    reload = sub.add_parser("verify-reload")
    reload.add_argument("--run", type=Path, required=True)
    reload.add_argument("--contract", type=Path, required=True)
    reload.add_argument("--receipt", type=Path, required=True)
    reload.add_argument("--replay-contract", type=Path, required=True)
    coverage = sub.add_parser("coverage")
    coverage.add_argument("--board", type=Path, required=True)
    coverage.add_argument("--observed-at", required=True)
    for cmd in (contract, verify, coverage, replay_contract, reload):
        cmd.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "contract":
            result = contract_from_source(args.source)
        elif args.command == "reload-contract":
            result = contract_from_replay_source(args.source)
        elif args.command == "verify-reload":
            from .artifacts import require, local_file
            training = strict_json(local_file(args.run, "receipt.json", 2 * 1024 * 1024).read_bytes())
            checked = verify_smoke(args.run, strict_json(args.contract.read_bytes()))
            expected = strict_json(args.replay_contract.read_bytes())
            require(expected.get("schema") == "meok.kaggle-reload-contract/1", "unknown replay contract")
            require(expected.get("training_source_sha256") == checked["source_sha256"], "replay training source mismatch")
            receipt = strict_json(local_file(args.receipt.parent, args.receipt.name, 2 * 1024 * 1024).read_bytes())
            result = verify_reload(receipt, checked, training, replay_sha256=expected["replay_sha256"],
                                   prompts=expected["prompts"], generation=expected["generation"])
            require(result["adapter_sha256"] == expected.get("adapter_sha256"), "replay contract adapter mismatch")
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
