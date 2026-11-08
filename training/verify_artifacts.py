#!/usr/bin/env python3
"""Verify a downloaded MEOK smoke bundle without torch or loading executable code.

Checks receipt consistency, pinned identity, exact file bytes and a constrained
LoRA safetensors structure. This establishes bundle integrity, not provenance
authenticity, model quality, independent replay or permission to publish.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import struct
import sys

MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
ALLOWED_FILES = {"adapter/adapter_config.json", "adapter/adapter_model.safetensors", "adapter/README.md"}
SHA_RE = re.compile(r"[0-9a-f]{64}")
TENSOR_RE = re.compile(r"base_model\.model\.model\.layers\.(\d+)\.self_attn\.(q_proj|v_proj)\.lora_([AB])\.weight")
MAX_ADAPTER_BYTES = 32 * 1024 * 1024


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def parse_json(raw):
    def reject_constant(_value):
        raise ValueError("Non-finite JSON number")
    return json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject_constant)


def safe_file(root, relative, max_bytes):
    require(isinstance(relative, str) and "\\" not in relative and ":" not in relative, "Invalid artifact path")
    path = PurePosixPath(relative)
    require(not path.is_absolute() and ".." not in path.parts and str(path) == relative, "Path traversal or noncanonical path")
    target = root
    for part in path.parts:
        target = target / part
        require(not target.is_symlink(), "Symlink artifacts are forbidden")
    require(target.is_file() and target.resolve().is_relative_to(root), "Artifact must be a regular file under the run directory")
    size = target.stat().st_size
    require(0 < size <= max_bytes, "Artifact size outside bounds")
    return target.read_bytes()


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_safetensors(raw):
    require(len(raw) >= 10, "Truncated safetensors")
    header_size = struct.unpack("<Q", raw[:8])[0]
    require(2 <= header_size <= min(1024 * 1024, len(raw)-8), "Invalid safetensors header size")
    header = parse_json(raw[8:8+header_size])
    require(isinstance(header, dict), "Safetensors header must be an object")
    data = memoryview(raw)[8+header_size:]
    intervals, pairs = [], {}
    for name, entry in header.items():
        if name == "__metadata__":
            require(isinstance(entry, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in entry.items()), "Invalid tensor metadata")
            continue
        match = TENSOR_RE.fullmatch(name)
        require(match is not None and 0 <= int(match[1]) < 28, "Unexpected non-LoRA tensor")
        require(isinstance(entry, dict), "Invalid tensor entry")
        dtype, shape, offsets = entry.get("dtype"), entry.get("shape"), entry.get("data_offsets")
        widths = {"F32": 4, "F16": 2, "BF16": 2}
        require(dtype in widths, "Unsupported adapter tensor dtype")
        require(isinstance(shape, list) and len(shape) == 2 and all(type(n) is int and 0 < n <= 4096 for n in shape), "Invalid LoRA matrix shape")
        require(shape[0 if match[3] == "A" else 1] == 8, "Unexpected LoRA rank")
        require(isinstance(offsets, list) and len(offsets) == 2 and all(type(n) is int for n in offsets), "Invalid tensor offsets")
        start, end = offsets
        require(0 <= start < end <= len(data) and end-start == math.prod(shape)*widths[dtype], "Tensor byte length mismatch")
        intervals.append((start, end))
        pairs.setdefault((int(match[1]), match[2]), set()).add(match[3])
        # Inspect IEEE exponent bits; NaN and infinity are both rejected. This
        # works for F32, F16 and BF16 without installing numerical libraries.
        fmt, exponent = ("<I", 0x7f800000) if dtype == "F32" else (("<H", 0x7c00) if dtype == "F16" else ("<H", 0x7f80))
        require(all((bits[0] & exponent) != exponent for bits in struct.iter_unpack(fmt, data[start:end])), "Non-finite adapter weight")
    require(len(pairs) == 56 and all(value == {"A", "B"} for value in pairs.values()), "Incomplete 28-layer q/v LoRA adapter")
    cursor = 0
    for start, end in sorted(intervals):
        require(start == cursor, "Overlapping or unreferenced tensor data")
        cursor = end
    require(cursor == len(data), "Trailing unreferenced tensor data")
    return {"tensor_count": len(intervals), "all_weights_finite": True, "data_bytes": len(data)}


def verify_bundle(directory, expected_source_sha256):
    require(SHA_RE.fullmatch(expected_source_sha256 or "") is not None, "Expected source SHA-256 is required")
    given_root = Path(directory)
    require(given_root.is_dir() and not given_root.is_symlink(), "Run directory must not be a symlink")
    root = given_root.resolve()
    receipt_raw = safe_file(root, "receipt.json", 1024 * 1024)
    receipt = parse_json(receipt_raw)
    require(isinstance(receipt, dict) and receipt.get("schema") == "meok.lora-smoke/1.0", "Unknown receipt schema")
    require(receipt.get("status") == "COMPLETED", "Only a COMPLETED run can be validated")
    require(receipt.get("source_sha256") == expected_source_sha256, "Run source identity mismatch")
    require(receipt.get("model") == {"id": MODEL_ID, "revision": MODEL_REVISION, "licence": "apache-2.0"}, "Base model identity mismatch")
    require(receipt.get("published") is False and receipt.get("capability_improvement_established") is False, "Unexpected publication or improvement claim")
    steps = receipt.get("requested_steps")
    require(type(steps) is int and 10 <= steps <= 30 and receipt.get("completed_steps") == steps and type(receipt["completed_steps"]) is int, "Incomplete or unbounded training")
    losses = receipt.get("train_losses")
    require(isinstance(losses, list) and len(losses) == steps and all(finite_number(n) and n >= 0 for n in losses), "Invalid training losses")
    for key in ("base_development_loss", "adapter_development_loss"):
        require(finite_number(receipt.get(key)) and receipt[key] >= 0, "Missing or non-finite development loss")
    gpu = receipt.get("gpu")
    require(isinstance(gpu, dict) and gpu.get("cuda_available") is True and isinstance(gpu.get("name"), str) and gpu["name"].strip()
            and isinstance(gpu.get("cuda_runtime"), str) and gpu["cuda_runtime"].strip()
            and type(gpu.get("count")) is int and gpu["count"] > 0, "No observed CUDA device")
    require(receipt.get("attention") == "PyTorch SDPA", "Unexpected attention backend")
    manifest = receipt.get("adapter_manifest")
    require(isinstance(manifest, list) and 2 <= len(manifest) <= 3, "Invalid adapter manifest")
    seen, contents = set(), {}
    for item in manifest:
        require(isinstance(item, dict), "Invalid manifest entry")
        path = item.get("path")
        require(isinstance(path, str) and path in ALLOWED_FILES and path not in seen, "Unexpected or duplicate adapter path")
        require(SHA_RE.fullmatch(item.get("sha256", "")) is not None and type(item.get("bytes")) is int, "Invalid manifest digest or length")
        raw = safe_file(root, path, MAX_ADAPTER_BYTES)
        require(len(raw) == item["bytes"] and hashlib.sha256(raw).hexdigest() == item["sha256"], "Adapter bytes do not match manifest")
        contents[path] = raw; seen.add(path)
    require({"adapter/adapter_config.json", "adapter/adapter_model.safetensors"}.issubset(seen), "Missing adapter weights or configuration")
    actual = set()
    for path in (root / "adapter").rglob("*"):
        require(not path.is_symlink(), "Unexpected symlink in adapter directory")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    require(actual == seen, "Unmanifested adapter files")
    config = parse_json(contents["adapter/adapter_config.json"])
    require(config.get("base_model_name_or_path") == MODEL_ID and config.get("revision") == MODEL_REVISION, "Adapter base identity mismatch")
    require(config.get("peft_type") == "LORA" and config.get("task_type") == "CAUSAL_LM" and config.get("r") == 8
            and config.get("lora_alpha") == 16 and config.get("bias") == "none" and config.get("lora_dropout") == 0.0
            and set(config.get("target_modules", [])) == {"q_proj", "v_proj"}, "Unexpected adapter configuration")
    tensors = validate_safetensors(contents["adapter/adapter_model.safetensors"])
    data = parse_json(safe_file(root, "data.json", 128 * 1024))
    dataset = receipt.get("dataset", {})
    require(data.get("rights") == dataset.get("rights"), "Fixture rights metadata mismatch")
    prompts, ids = set(), set()
    for key, count in (("train", 16), ("development", 6)):
        rows = data.get(key)
        require(isinstance(rows, list) and len(rows) == count and dataset.get(f"{key}_n") == count, "Fixture split count mismatch")
        encoded = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        require(hashlib.sha256(encoded).hexdigest() == dataset.get("digests", {}).get(key), "Fixture split digest mismatch")
        for row in rows:
            require(isinstance(row, list) and len(row) == 3 and all(isinstance(v, str) and v.strip() for v in row), "Invalid fixture row")
            prompt = re.sub(r"\s+", " ", row[1]).strip().casefold()
            require(row[0] not in ids and prompt not in prompts, "Train/development duplication")
            ids.add(row[0]); prompts.add(prompt)
    return {"schema": "meok.adapter-integrity/1.0", "status": "INTEGRITY_CHECKED",
            "receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(), "source_sha256": expected_source_sha256,
            "adapter_files": sorted(seen), "safetensors": tensors, "model": receipt["model"],
            "completed_steps": steps, "gpu": gpu, "capability_improvement_established": False,
            "limitations": ["Local file integrity is not cryptographic provenance authentication.",
                            "This does not execute or independently evaluate the adapter.",
                            "No release, publication or production promotion is authorised by this result."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--expected-source-sha256", required=True)
    args = parser.parse_args()
    try:
        result = verify_bundle(args.directory, args.expected_source_sha256)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError, OverflowError) as exc:
        print(json.dumps({"status": "REJECTED", "reason": str(exc), "capability_improvement_established": False}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
