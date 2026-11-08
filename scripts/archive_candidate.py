#!/usr/bin/env python3
"""Validate and retain an experimental adapter in a NEW PRIVATE HF model repo.

This does not deploy a Space, publish a public release, train or promote a model.
Default is a dry run. Reads the user's existing HF credential; never prints it.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
from verify_artifacts import verify_bundle

def validate_source(path, expected):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
        raise ValueError("Current training source differs from the submitted run")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--card", type=Path, required=True)
    parser.add_argument("--upload-private", action="store_true")
    args = parser.parse_args()
    if args.repo_id.count("/") != 1 or not args.repo_id.split("/")[1].startswith("MEOK-"):
        raise SystemExit("Use an authorised owner's MEOK- experimental namespace")
    validation = verify_bundle(args.artifacts, args.expected_source_sha256)
    validate_source(ROOT / "training/kaggle_smoke.py", args.expected_source_sha256)
    files = {"adapter_config.json": args.artifacts / "adapter/adapter_config.json",
             "LICENSE": ROOT / "training/QWEN-BASE-LICENSE",
             "adapter_model.safetensors": args.artifacts / "adapter/adapter_model.safetensors",
             "README.md": args.card,
             "evidence/training-receipt.json": args.artifacts / "receipt.json",
             "evidence/data.json": args.artifacts / "data.json",
             "evidence/kaggle_smoke.py": ROOT / "training/kaggle_smoke.py"}
    if not args.upload_private:
        print(json.dumps({"state":"PREPARED", "private":True, "repo":args.repo_id,
                          "files":list(files), "validation":validation}, indent=2))
        return
    from huggingface_hub import HfApi, CommitOperationAdd
    api = HfApi()
    identity = api.whoami()
    owner = args.repo_id.split("/")[0]
    if owner != identity.get("name") and not (owner == "csoai" and any(o.get("name") == "csoai" and o.get("roleInOrg") in ("admin", "write") for o in identity.get("orgs", []))):
        raise SystemExit("Use the authenticated user's namespace or their writable csoai organisation")
    if api.repo_exists(args.repo_id, repo_type="model"):
        raise SystemExit("Repository exists: inspect it before changing retained evidence")
    api.create_repo(args.repo_id, repo_type="model", private=True, exist_ok=False)
    info = api.model_info(args.repo_id)
    if info.private is not True:
        raise SystemExit("Expected private repository")
    result = api.create_commit(repo_id=args.repo_id, repo_type="model", parent_commit=info.sha,
        commit_message="Retain MEOK experimental LoRA smoke artifact and evidence",
        operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path)) for name,path in files.items()])
    readback = api.model_info(args.repo_id, revision=result.oid, files_metadata=True)
    expected = next(f["sha256"] for f in json.loads((args.artifacts / "receipt.json").read_text())["adapter_manifest"] if f["path"].endswith("adapter_model.safetensors"))
    weights = next(s for s in readback.siblings if s.rfilename == "adapter_model.safetensors")
    actual = weights.lfs.sha256 if weights.lfs else None
    if actual != expected or readback.private is not True:
        raise SystemExit("HF artifact readback did not match the expected hash/privacy")
    print(json.dumps({"state":"PRIVATE_ARTIFACT_RETAINED", "repo":args.repo_id,
                      "revision":result.oid,"adapter_sha256":actual,"private":True,
                      "deployed":False,"capability_improvement_established":False},indent=2))

if __name__ == "__main__": main()
