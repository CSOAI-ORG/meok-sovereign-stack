#!/usr/bin/env python3
"""Build, but never submit, a private saved-adapter CUDA reload smoke bundle."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re

TRAINING_SOURCE_SHA256 = "b6a573e9594f711e4ae097e460c00e8e8bec058475f314941ab32497b09a91be"
ADAPTER_SHA256 = "0c197a14fa771718028a0ea151d30457fb0dd9d55755d510c45a2862ca6d4f13"
SOURCE_KERNEL = "nicktempleman/meok-private-cuda-adapter-smoke"

SCRIPT = r'''#!/usr/bin/env python3
"""Private CUDA saved-adapter reload smoke. No training or publication."""
import base64, hashlib, importlib.metadata, json, os, signal, sys, time, types
from datetime import datetime, timezone
from pathlib import Path

TRAINING_SOURCE_SHA256 = "__TRAINING_SHA__"
EXPECTED_ADAPTER_SHA256 = "__ADAPTER_SHA__"
SOURCE_KERNEL = "__SOURCE_KERNEL__"
OUT = Path("/kaggle/working/meok-replay")
PROMPTS = [
    {"id": "reload-01", "text": "Explain in one sentence why a file hash helps detect changed bytes."},
    {"id": "reload-02", "text": "Give two concrete checks before loading a saved model adapter."},
    {"id": "reload-03", "text": "What can a successful model reload demonstrate, and what does it leave unknown?"},
]


def embedded(name, encoded, expected_sha):
    source = base64.b64decode(encoded, validate=True)
    if hashlib.sha256(source).hexdigest() != expected_sha:
        raise ValueError("EMBEDDED_SOURCE_IDENTITY_MISMATCH")
    module = types.ModuleType(name)
    module.__file__ = "<embedded-reviewed-" + name + ">"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


helpers = embedded("meok_smoke_helpers", "__SMOKE_B64__", TRAINING_SOURCE_SHA256)
verifier = embedded("meok_integrity_helpers", "__VERIFY_B64__", "__VERIFY_SHA__")


def save(receipt):
    temporary = OUT / "receipt.json.tmp"
    temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    temporary.replace(OUT / "receipt.json")


def find_training_artifacts(root=Path("/kaggle/input")):
    # Only one explicitly mounted source kernel is requested. Do not traverse
    # arbitrary mounts or follow symlinks; fail rather than scan an unbounded tree.
    if not root.is_dir() or root.is_symlink():
        raise ValueError("TRAINING_MOUNT_UNAVAILABLE")
    candidates, entries = [], 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        current = Path(directory)
        relative = current.relative_to(root)
        entries += len(dirs) + len(files)
        if entries > 2000:
            raise ValueError("MOUNT_SCAN_LIMIT")
        dirs[:] = [name for name in sorted(dirs) if not (current / name).is_symlink()]
        if len(relative.parts) >= 7:
            dirs[:] = []
        if "receipt.json" not in files:
            continue
        path = current / "receipt.json"
        if path.is_symlink() or not 0 < path.stat().st_size <= 1024 * 1024:
            continue
        try:
            data = verifier.parse_json(path.read_bytes())
        except (ValueError, OSError):
            continue
        if isinstance(data, dict) and data.get("schema") == "meok.lora-smoke/1.0" and data.get("source_sha256") == TRAINING_SOURCE_SHA256:
            candidates.append(current)
    if len(candidates) != 1:
        raise ValueError("TRAINING_ARTIFACT_NOT_UNIQUE")
    training = candidates[0]
    checked = verifier.verify_bundle(training, TRAINING_SOURCE_SHA256)
    weights = training / "adapter/adapter_model.safetensors"
    if hashlib.sha256(weights.read_bytes()).hexdigest() != EXPECTED_ADAPTER_SHA256:
        raise ValueError("EXPECTED_ADAPTER_IDENTITY_MISMATCH")
    return training, checked


def execute(receipt):
    training, integrity = find_training_artifacts()
    receipt["artifact_integrity"] = integrity
    receipt["artifact_mount"] = training.relative_to(Path("/kaggle/input")).as_posix()
    save(receipt)
    helpers.install_dependencies()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, PeftModel
    if not torch.cuda.is_available() or not torch.version.cuda:
        raise RuntimeError("CUDA_GPU_UNAVAILABLE")
    versions = {key: importlib.metadata.version(key) for key in helpers.DEPENDENCIES}
    if versions != helpers.DEPENDENCIES:
        raise RuntimeError("DEPENDENCY_MISMATCH")
    versions.update({"torch": torch.__version__, "python": sys.version.split()[0]})
    gpu = {"cuda_available": True, "name": torch.cuda.get_device_name(0),
        "count": torch.cuda.device_count(), "used_device": "cuda:0",
        "cuda_runtime": torch.version.cuda, "capability": list(torch.cuda.get_device_capability(0)),
        "memory_bytes": torch.cuda.get_device_properties(0).total_memory}
    receipt.update(gpu=gpu, versions=versions)
    save(receipt)
    torch.manual_seed(73); torch.cuda.manual_seed_all(73)
    torch.backends.cudnn.benchmark = False
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(helpers.MODEL_ID, revision=helpers.MODEL_REVISION, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(helpers.MODEL_ID, revision=helpers.MODEL_REVISION,
        trust_remote_code=False, use_safetensors=True, torch_dtype=dtype,
        attn_implementation="sdpa").to("cuda:0")
    if base.config._commit_hash != helpers.MODEL_REVISION:
        raise RuntimeError("BASE_REVISION_MISMATCH")
    # Explicit local LoRA class prevents adapter metadata selecting another
    # PEFT implementation. No remote adapter code or pickle file is loaded.
    config = LoraConfig.from_pretrained(str(training / "adapter"), local_files_only=True)
    model = PeftModel.from_pretrained(base, str(training / "adapter"), config=config,
        is_trainable=False, local_files_only=True, use_safetensors=True).to("cuda:0")
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("REPLAY_MODEL_NOT_FROZEN")
    receipt.update(adapter_loaded=True, precision=str(dtype), attention="PyTorch SDPA")
    save(receipt)
    for prompt in PROMPTS:
        started = time.monotonic()
        row = {**prompt, "status": "RUNNING"}
        try:
            ids = tokenizer.apply_chat_template([
                {"role": "system", "content": helpers.SYSTEM},
                {"role": "user", "content": prompt["text"]}], tokenize=True,
                add_generation_prompt=True, enable_thinking=False, return_tensors="pt").to("cuda:0")
            with torch.inference_mode():
                answer = model.generate(ids, attention_mask=torch.ones_like(ids),
                    do_sample=False, max_new_tokens=96, use_cache=True,
                    pad_token_id=tokenizer.eos_token_id)
            torch.cuda.synchronize()
            output = tokenizer.decode(answer[0, ids.shape[1]:], skip_special_tokens=True)
            if not output.strip():
                raise RuntimeError("EMPTY_GENERATION")
            row.update(status="RUNTIME_OBSERVED", output=output,
                output_sha256=hashlib.sha256(output.encode()).hexdigest(),
                prompt_sha256=hashlib.sha256(prompt["text"].encode()).hexdigest(),
                generated_tokens=int(answer.shape[1]-ids.shape[1]))
        except Exception as error:
            row.update(status="FAILED", error_type=type(error).__name__)
            raise
        finally:
            row["duration_seconds"] = time.monotonic()-started
            receipt["outputs"].append(row)
            save(receipt)
    receipt.update(status="COMPLETED", successful_generations=len(receipt["outputs"]),
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(0))
    if receipt["successful_generations"] != len(PROMPTS):
        raise RuntimeError("INCOMPLETE_REPLAY")


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError("OUTPUT_NOT_EMPTY")
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ.update({"HF_HUB_DISABLE_TELEMETRY": "1", "WANDB_DISABLED": "true", "TOKENIZERS_PARALLELISM": "false"})
    started = time.monotonic()
    receipt = {"schema": "meok.adapter-reload/1.0", "status": "RUNNING",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "replay_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "training_source_sha256": TRAINING_SOURCE_SHA256, "source_kernel": SOURCE_KERNEL,
        "expected_adapter_sha256": EXPECTED_ADAPTER_SHA256,
        "model": {"id": helpers.MODEL_ID, "revision": helpers.MODEL_REVISION, "licence": "apache-2.0"},
        "adapter_loaded": False, "training_performed": False, "published": False,
        "capability_improvement_established": False, "outputs": [],
        "generation": {"seed": 73, "do_sample": False, "max_new_tokens": 96, "enable_thinking": False},
        "limitations": ["Saved-adapter reload and nonempty generation smoke only.",
            "This is a second process on the same platform, not an independent capability evaluation.",
            "Output correctness, improvement, safety and human acceptance remain unmeasured.",
            "Source-kernel name is a mount locator; adapter and source SHA-256 pins bind selected bytes."]}
    save(receipt)
    def timeout(_signal, _frame):
        raise TimeoutError("REPLAY_TIMEOUT")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(900)
    try:
        execute(receipt)
        code = 0
    except Exception as error:
        receipt.update(status="FAILED", error_type=type(error).__name__)
        print(f"FAILED: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        code = 1
    finally:
        signal.alarm(0)
        receipt["duration_seconds"] = time.monotonic()-started
        save(receipt)
    print(json.dumps({"status": receipt["status"], "receipt": str(OUT / "receipt.json"),
        "successful_generations": receipt.get("successful_generations"), "capability_improvement_established": False}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
'''


def metadata(owner, slug):
    if not re.fullmatch(r"[a-zA-Z0-9_-]{3,50}", owner) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{4,49}", slug):
        raise ValueError("Invalid Kaggle owner or slug")
    return {"id": f"{owner}/{slug}", "title": "MEOK private saved adapter reload",
        "code_file": "replay.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "dataset_sources": [], "competition_sources": [], "kernel_sources": [SOURCE_KERNEL], "model_sources": []}


def render_script():
    parent = Path(__file__).resolve().parent
    smoke = (parent / "kaggle_smoke.py").read_bytes()
    verifier = (parent / "verify_artifacts.py").read_bytes()
    if hashlib.sha256(smoke).hexdigest() != TRAINING_SOURCE_SHA256:
        raise ValueError("Training source differs from the completed pinned run")
    substitutions = {"__TRAINING_SHA__": TRAINING_SOURCE_SHA256, "__ADAPTER_SHA__": ADAPTER_SHA256,
        "__SOURCE_KERNEL__": SOURCE_KERNEL, "__SMOKE_B64__": base64.b64encode(smoke).decode(),
        "__VERIFY_B64__": base64.b64encode(verifier).decode(), "__VERIFY_SHA__": hashlib.sha256(verifier).hexdigest()}
    script = SCRIPT
    for key, value in substitutions.items():
        script = script.replace(key, value)
    compile(script, "replay.py", "exec")
    return script


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    meta, script = metadata(args.owner, args.slug), render_script()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "replay.py").write_text(script)
    (args.output / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"bundle": str(args.output), "private": True, "submitted": False,
        "source_sha256": hashlib.sha256(script.encode()).hexdigest(), "adapter_sha256": ADAPTER_SHA256,
        "kernel_sources": meta["kernel_sources"]}))


if __name__ == "__main__":
    main()
