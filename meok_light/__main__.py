import argparse
import json
from .core import load_routes, run, save_receipt

parser = argparse.ArgumentParser(description="MEOK SOVOS-light inference; no model downloads")
parser.add_argument("prompt")
parser.add_argument("--specialist", default="general")
parser.add_argument("--config", default="configs/routes.json")
parser.add_argument("--endpoint", default="http://127.0.0.1:11434")
parser.add_argument("--max-tokens", type=int, default=128)
parser.add_argument("--receipts", default="run-receipts")
args = parser.parse_args()
result = run(args.prompt, args.specialist, load_routes(args.config), args.endpoint,
             max_tokens=args.max_tokens)
save_receipt(result["receipt"], args.receipts)
print(json.dumps(result, indent=2))
raise SystemExit(0 if result["receipt"]["status"] == "RUNTIME_OBSERVED" else 1)
