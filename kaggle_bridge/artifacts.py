"""Verify a CUDA smoke export against a separately retained source contract.

Byte integrity is checked. Execution, model quality and producer identity are
not authenticated by an unsigned notebook receipt. Supply a stable, completed
export in a private directory; this is not a sandbox against concurrent writers.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re


class ArtifactError(ValueError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ArtifactError("duplicate JSON key")
            result[key] = value
        return result
    def bad_number(value):
        raise ArtifactError("nonfinite JSON number")
    def decimal(value):
        number = float(value)
        if not math.isfinite(number):
            raise ArtifactError("nonfinite JSON number")
        return number
    return json.loads(data, object_pairs_hook=pairs, parse_constant=bad_number, parse_float=decimal)


def require(condition, message):
    if not condition:
        raise ArtifactError(message)


def contract_from_source(source: Path):
    """Read literal constants without importing or executing the training code."""
    raw = source.read_bytes()
    values = {}
    needed = {"MODEL_ID", "MODEL_REVISION", "DEPENDENCIES", "TRAIN", "DEVELOPMENT", "DATA_RIGHTS"}
    for node in ast.parse(raw).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in needed:
                    require(target.id not in values, "source constant reassigned")
                    values[target.id] = ast.literal_eval(node.value)
    require(set(values) == needed, "source does not expose the smoke contract")
    require(re.fullmatch(r"[0-9a-f]{40}", values["MODEL_REVISION"]) is not None, "moving model revision")
    return {"schema": "meok.kaggle-source-contract/1", "source_sha256": sha(raw),
            "model": {"id": values["MODEL_ID"], "revision": values["MODEL_REVISION"]},
            "dependencies": values["DEPENDENCIES"], "rights": values["DATA_RIGHTS"],
            "dataset": {name: {"sha256": sha(canonical(values[key])), "n": len(values[key])}
                        for name, key in (("train", "TRAIN"), ("development", "DEVELOPMENT"))}}


def local_file(root, name, max_bytes):
    require(isinstance(name, str) and "\\" not in name, "invalid artifact path")
    parts = PurePosixPath(name)
    require(not parts.is_absolute() and parts.parts and name == parts.as_posix() and
            all(p not in {"", ".", ".."} for p in name.split("/")), "unsafe artifact path")
    path = root
    for part in parts.parts:
        path = path / part
        require(not path.is_symlink(), "symlink artifact forbidden")
    require(path.is_file(), "missing artifact: " + name)
    require(path.stat().st_size <= max_bytes, "artifact exceeds size budget")
    return path


def digest_file(path, expected_bytes):
    h = hashlib.sha256()
    count = 0
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(min(1024 * 1024, expected_bytes - count + 1))
            if not chunk:
                break
            count += len(chunk)
            require(count <= expected_bytes, "artifact grew during verification")
            h.update(chunk)
    require(count == expected_bytes, "artifact size changed during verification")
    return h.hexdigest()


def verify_smoke(root: Path, contract: dict, max_total_bytes=128 * 1024 * 1024):
    require(not root.is_symlink() and root.is_dir(), "run directory missing or symlinked")
    root = root.resolve()
    require(contract.get("schema") == "meok.kaggle-source-contract/1", "unknown source contract")
    receipt_path = local_file(root, "receipt.json", 2 * 1024 * 1024)
    receipt_raw = receipt_path.read_bytes()
    receipt = strict_json(receipt_raw)
    require(receipt.get("schema") == "meok.lora-smoke/1.0", "unknown receipt schema")
    require(receipt.get("status") == "COMPLETED", "run is not reported completed")
    require(receipt.get("source_sha256") == contract["source_sha256"], "training source mismatch")
    require(all(receipt.get("model", {}).get(k) == v for k, v in contract["model"].items()), "model identity mismatch")
    require(receipt.get("published") is False and receipt.get("capability_improvement_established") is False,
            "unsupported publication or improvement claim")
    steps = receipt.get("requested_steps")
    require(type(steps) is int and 10 <= steps <= 30 and receipt.get("completed_steps") == steps,
            "incomplete optimisation steps")
    finite = lambda n: type(n) in (int, float) and math.isfinite(n)
    losses = receipt.get("train_losses")
    require(isinstance(losses, list) and len(losses) == steps and all(finite(n) and n >= 0 for n in losses),
            "missing or nonfinite per-step losses")
    require(all(finite(receipt.get(k)) and receipt[k] >= 0 for k in
                ("base_development_loss", "adapter_development_loss")), "invalid development losses")
    gpu = receipt.get("gpu", {})
    require(gpu.get("cuda_available") is True and isinstance(gpu.get("name"), str) and gpu["name"].strip()
            and gpu.get("cuda_runtime"), "missing reported CUDA device")
    require(all(receipt.get("versions", {}).get(k) == v for k, v in contract["dependencies"].items()),
            "dependency version mismatch")
    data_path = local_file(root, "data.json", 2 * 1024 * 1024)
    data_raw = data_path.read_bytes()
    data = strict_json(data_raw)
    require(data.get("rights") == contract["rights"] == receipt.get("dataset", {}).get("rights"), "rights mismatch")
    prompts, ids = set(), set()
    for name in ("train", "development"):
        rows = data.get(name)
        expected = contract["dataset"][name]
        require(isinstance(rows, list) and len(rows) == expected["n"] and sha(canonical(rows)) == expected["sha256"],
                "data split differs from reviewed source")
        require(receipt["dataset"].get("digests", {}).get(name) == expected["sha256"]
                and receipt["dataset"].get(name + "_n") == expected["n"], "data receipt mismatch")
        for row in rows:
            require(isinstance(row, list) and len(row) == 3 and all(isinstance(v, str) and v.strip() for v in row),
                    "invalid training example")
            norm = " ".join(row[1].split()).casefold()
            require(row[0] not in ids and norm not in prompts, "overlapping data splits")
            ids.add(row[0]); prompts.add(norm)
    manifest = receipt.get("adapter_manifest")
    require(isinstance(manifest, list) and 2 <= len(manifest) <= 16, "missing adapter manifest")
    seen, total = set(), 0
    for entry in manifest:
        require(isinstance(entry, dict), "invalid manifest entry")
        name = entry.get("path")
        require(isinstance(name, str) and name.startswith("adapter/") and name not in seen, "invalid or duplicate manifest path")
        require(type(entry.get("bytes")) is int and entry["bytes"] > 0, "invalid artifact byte count")
        require(isinstance(entry.get("sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]), "invalid artifact digest")
        file = local_file(root, name, max_total_bytes)
        total += file.stat().st_size
        require(total <= max_total_bytes, "adapter exceeds total size budget")
        require(file.stat().st_size == entry["bytes"] and digest_file(file, entry["bytes"]) == entry["sha256"], "artifact digest or size mismatch")
        seen.add(name)
    require({"adapter/adapter_model.safetensors", "adapter/adapter_config.json"} <= seen, "required adapter files missing")
    for file in (root / "adapter").rglob("*"):
        require(not file.is_symlink(), "symlink in adapter directory")
        if file.is_file():
            require(file.relative_to(root).as_posix() in seen, "unmanifested adapter file")
    config = strict_json(local_file(root, "adapter/adapter_config.json", 1024 * 1024).read_bytes())
    require(config.get("base_model_name_or_path") == contract["model"]["id"] and
            config.get("revision") == contract["model"]["revision"] and config.get("peft_type") == "LORA",
            "adapter base/config mismatch")
    return {"schema": "meok.kaggle-artifact-check/1", "status": "ARTIFACTS_VERIFIED",
            "reported_run_status": "COMPLETED", "receipt_sha256": sha(receipt_raw),
            "data_sha256": sha(data_raw), "contract_sha256": sha(canonical(contract)),
            "source_sha256": contract["source_sha256"], "model": contract["model"],
            "adapter_files": len(seen), "adapter_bytes": total, "signature_verified": False,
            "execution_authenticated": False, "model_load_tested": False, "quality_improvement_established": False,
            "gspc_admission": "NOT_ELIGIBLE_SELF_PRODUCT", "published": False,
            "limitations": ["Checks exported bytes against separately retained source and input pins.",
                            "Unsigned receipt fields report execution; they do not authenticate the worker.",
                            "No tensor deserialization, inference, promotion, upload or public Council score."]}
