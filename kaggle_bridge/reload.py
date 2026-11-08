"""Check a producer's reload receipt; never authenticate or execute its run."""
from __future__ import annotations

import math
import re

from .artifacts import ArtifactError, canonical, require, sha


def _digest(value, label, length=64):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{%d}" % length, value),
            label + " must be a lowercase hexadecimal digest")
    return value


def _encoded(value, label):
    try:
        return canonical(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ArtifactError(label + " must contain finite JSON values") from exc


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def verify_reload(receipt: dict, training_check: dict, training_receipt: dict, *,
                  replay_sha256: str, prompts: list[tuple[str, str]],
                  generation: dict) -> dict:
    """Check against separately retained expectations, with no network or loads.

    ``training_check`` must be the local ``verify_smoke`` result for the original
    export, including its raw and canonical receipt hashes. The caller supplies
    reviewed replay-source and prompt/generation pins independently of receipt.
    JSON bytes must be decoded with ``strict_json`` before calling this API.
    Consistent unsigned statements remain producer reports, not proof of a run.
    """
    require(all(isinstance(x, dict) for x in (receipt, training_check, training_receipt, generation)),
            "receipts, training check and generation must be objects")
    _digest(replay_sha256, "expected replay source")
    require(training_check.get("schema") == "meok.kaggle-artifact-check/1"
            and training_check.get("status") == "ARTIFACTS_VERIFIED",
            "requires a successful local training artifact check")
    raw_training_sha = _digest(training_check.get("receipt_sha256"), "training receipt")
    canonical_training_sha = _digest(training_check.get("receipt_canonical_sha256"),
                                     "canonical training receipt")
    require(sha(_encoded(training_receipt, "training receipt")) == canonical_training_sha,
            "training receipt differs from the locally verified receipt")
    source_sha = _digest(training_check.get("source_sha256"), "training source")
    require(training_receipt.get("schema") == "meok.lora-smoke/1.0"
            and training_receipt.get("status") == "COMPLETED"
            and training_receipt.get("source_sha256") == source_sha,
            "training receipt identity or state mismatch")
    model = training_check.get("model")
    require(isinstance(model, dict) and isinstance(model.get("id"), str) and model["id"].strip(),
            "missing checked model identity")
    _digest(model.get("revision"), "model revision", 40)
    training_model = training_receipt.get("model")
    require(isinstance(training_model, dict) and
            all(training_model.get(k) == model[k] for k in ("id", "revision")),
            "training model differs from the checked model")
    manifest = training_receipt.get("adapter_manifest")
    require(isinstance(manifest, list) and all(isinstance(row, dict) for row in manifest),
            "missing verified training adapter manifest")
    entries = [row for row in manifest if row.get("path") == "adapter/adapter_model.safetensors"]
    require(len(entries) == 1, "training receipt must identify one adapter weight file")
    adapter_sha = _digest(entries[0].get("sha256"), "adapter weights")

    require(receipt.get("schema") == "meok.adapter-reload/1.0"
            and receipt.get("status") == "COMPLETED", "reload is not reported completed")
    require(receipt.get("replay_source_sha256") == replay_sha256
            and receipt.get("training_source_sha256") == source_sha
            and receipt.get("expected_adapter_sha256") == adapter_sha,
            "reload source or adapter pin mismatch")
    require(receipt.get("model") == training_model, "reload model identity mismatch")
    require(receipt.get("adapter_loaded") is True, "adapter load was not reported")
    for key in ("training_performed", "published", "capability_improvement_established"):
        require(receipt.get(key) is False, "unsupported reload effect or improvement claim")
    integrity = receipt.get("artifact_integrity")
    require(isinstance(integrity, dict)
            and integrity.get("schema") == "meok.adapter-integrity/1.0"
            and integrity.get("status") == "INTEGRITY_CHECKED"
            and integrity.get("receipt_sha256") == raw_training_sha
            and integrity.get("source_sha256") == source_sha
            and integrity.get("model") == training_model,
            "producer integrity report does not bind the checked training export")

    require(isinstance(prompts, list) and prompts, "expected prompts must be a nonempty list")
    expected = {}
    for pair in prompts:
        require(isinstance(pair, (tuple, list)) and len(pair) == 2
                and all(isinstance(x, str) and x.strip() for x in pair), "invalid expected prompt")
        ident, text = pair
        require(ident not in expected, "duplicate expected prompt ID")
        expected[ident] = text
    require(type(generation.get("seed")) is int
            and type(generation.get("do_sample")) is bool
            and type(generation.get("enable_thinking")) is bool
            and type(generation.get("max_new_tokens")) is int
            and generation["max_new_tokens"] > 0, "invalid expected generation settings")
    require(_encoded(receipt.get("generation"), "reported generation") ==
            _encoded(generation, "expected generation"), "generation settings mismatch")
    outputs = receipt.get("outputs")
    require(isinstance(outputs, list) and len(outputs) == len(expected), "missing or extra generation")
    seen = set()
    for row in outputs:
        require(isinstance(row, dict), "invalid generation row")
        ident = row.get("id")
        require(isinstance(ident, str) and ident in expected and ident not in seen,
                "unknown or duplicate prompt ID")
        seen.add(ident)
        require(row.get("status") == "RUNTIME_OBSERVED", "generation was not reported complete")
        text = expected[ident]
        require(row.get("text") == text and row.get("prompt_sha256") == sha(text.encode("utf-8")),
                "prompt text or digest mismatch")
        output = row.get("output")
        require(isinstance(output, str) and output.strip(), "empty generated output")
        require(row.get("output_sha256") == sha(output.encode("utf-8")), "output digest mismatch")
        tokens = row.get("generated_tokens")
        require(type(tokens) is int and 0 < tokens <= generation["max_new_tokens"],
                "invalid reported token count")
        require(_positive(row.get("duration_seconds")), "invalid generation duration")
    require(type(receipt.get("successful_generations")) is int
            and receipt["successful_generations"] == len(expected), "generation count mismatch")
    require(_positive(receipt.get("duration_seconds")), "invalid reload duration")
    return {"schema": "meok.kaggle-reload-check/1", "status": "RELOAD_RECEIPT_VERIFIED",
            "producer_reported_generation_count": len(seen), "execution_authenticated": False,
            "answer_correctness": "UNMEASURED", "independent_replay_established": False,
            "quality_improvement_established": False,
            "replay_source_sha256": replay_sha256, "training_source_sha256": source_sha,
            "adapter_sha256": adapter_sha, "training_receipt_sha256": raw_training_sha,
            "reload_receipt_canonical_sha256": sha(_encoded(receipt, "reload receipt")),
            "limitations": ["Verified consistency of an unsigned producer receipt against supplied pins.",
                            "No model loading, inference, execution authentication or independent replay.",
                            "Recorded text is ungraded; correctness and capability remain unmeasured."]}


def contract_from_replay_source(path):
    """Extract reviewed literal pins without executing the replay script."""
    import ast
    with path.open("rb") as stream:
        raw = stream.read(2 * 1024 * 1024 + 1)
    require(len(raw) <= 2 * 1024 * 1024, "replay source exceeds budget")
    tree = ast.parse(raw)
    def literal(node):
        for child in ast.walk(node):
            if isinstance(child, ast.Dict):
                require(all(isinstance(key, ast.Constant) and isinstance(key.value, str) for key in child.keys),
                        "dynamic literal dictionary keys or unpacking")
                keys = [key.value for key in child.keys]
                require(len(keys) == len(set(keys)), "duplicate literal dictionary keys")
        return ast.literal_eval(node)
    constants = {}
    wanted = {"PROMPTS", "TRAINING_SOURCE_SHA256", "EXPECTED_ADAPTER_SHA256"}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in wanted:
                    require(target.id not in constants, "replay constant reassigned")
                    constants[target.id] = literal(node.value)
    require(set(constants) == wanted, "replay source does not expose expected pins")
    generations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            candidate = any(isinstance(key, ast.Constant) and key.value == "schema"
                            and isinstance(value, ast.Constant) and value.value == "meok.adapter-reload/1.0"
                            for key, value in zip(node.keys, node.values))
            if candidate:
                require(all(isinstance(key, ast.Constant) and isinstance(key.value, str) for key in node.keys),
                        "dynamic replay receipt keys or unpacking")
                keys = [key.value for key in node.keys]
                require(len(keys) == len(set(keys)), "duplicate replay receipt keys")
                pairs = dict(zip(keys, node.values))
                value = literal(pairs["generation"])
                require(isinstance(value, dict) and set(value) == {"seed", "do_sample", "max_new_tokens", "enable_thinking"},
                        "invalid replay generation contract")
                require(type(value["seed"]) is int and type(value["do_sample"]) is bool
                        and type(value["enable_thinking"]) is bool
                        and type(value["max_new_tokens"]) is int and value["max_new_tokens"] > 0,
                        "invalid replay generation value types")
                generation_keys = [key.value for key in pairs["generation"].keys]
                require(len(generation_keys) == len(set(generation_keys)), "duplicate replay generation keys")
                generations.append(value)
    require(len(generations) == 1, "ambiguous or absent replay generation contract")
    rows = constants["PROMPTS"]
    require(isinstance(rows, list) and rows and all(isinstance(p, dict) for p in rows), "invalid replay prompts")
    prompts = [[row["id"], row["text"]] for row in rows]
    require(all(isinstance(v, str) and v.strip() for pair in prompts for v in pair), "invalid replay prompt text")
    require(len({pair[0] for pair in prompts}) == len(prompts), "duplicate replay prompt ID")
    return {"schema": "meok.kaggle-reload-contract/1", "replay_sha256": sha(raw),
            "training_source_sha256": _digest(constants["TRAINING_SOURCE_SHA256"], "training source"),
            "adapter_sha256": _digest(constants["EXPECTED_ADAPTER_SHA256"], "adapter weights"),
            "prompts": prompts, "generation": generations[0]}
