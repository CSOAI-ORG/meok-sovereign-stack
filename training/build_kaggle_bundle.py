#!/usr/bin/env python3
"""Create a private Kaggle script bundle; this command never submits a job."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


def metadata(owner, slug):
    if not re.fullmatch(r"[a-zA-Z0-9_-]{3,50}", owner):
        raise ValueError("Invalid Kaggle owner")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{4,49}", slug):
        raise ValueError("Invalid Kaggle slug")
    return {"id": f"{owner}/{slug}", "title": "MEOK private CUDA adapter smoke",
        "code_file": "kaggle_smoke.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", required=True); parser.add_argument("--slug", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    meta = metadata(args.owner, args.slug)
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).with_name("kaggle_smoke.py")
    shutil.copyfile(source, args.output / "kaggle_smoke.py")
    (args.output / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"bundle": str(args.output), "private": True, "submitted": False,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
