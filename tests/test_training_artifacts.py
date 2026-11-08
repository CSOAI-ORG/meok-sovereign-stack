import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verify = load("verify_artifacts", "training/verify_artifacts.py")
smoke = load("smoke_artifacts_fixture", "training/kaggle_smoke.py")


def synthetic_adapter(nonfinite=False):
    header, parts, offset = {}, [], 0
    for layer in range(28):
        for target in ("q_proj", "v_proj"):
            for half in ("A", "B"):
                name = f"base_model.model.model.layers.{layer}.self_attn.{target}.lora_{half}.weight"
                part = struct.pack("<8f", *([float("nan")] + [0.0]*7 if nonfinite else [0.0]*8))
                header[name] = {"dtype": "F32", "shape": [8, 1] if half == "A" else [1, 8], "data_offsets": [offset, offset+len(part)]}
                offset += len(part); parts.append(part)
    encoded = json.dumps(header).encode()
    return struct.pack("<Q", len(encoded)) + encoded + b"".join(parts)


class ArtifactValidationTests(unittest.TestCase):
    def make_bundle(self, directory):
        root = Path(directory)
        (root / "adapter").mkdir()
        config = {"base_model_name_or_path": smoke.MODEL_ID, "revision": smoke.MODEL_REVISION,
            "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 8, "lora_alpha": 16,
            "bias": "none", "lora_dropout": 0.0, "target_modules": ["q_proj", "v_proj"]}
        (root / "adapter/adapter_config.json").write_text(json.dumps(config))
        (root / "adapter/adapter_model.safetensors").write_bytes(synthetic_adapter())
        manifest = [{"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size,
                     "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in (root / "adapter").iterdir()]
        receipt = smoke.make_receipt("COMPLETED", source_sha256="a"*64, requested_steps=20, completed_steps=20,
            train_losses=[2.0]*20, base_development_loss=2.0, adapter_development_loss=2.1,
            gpu={"cuda_available": True, "name": "Test fixture GPU", "cuda_runtime": "12.1", "count": 1},
            attention="PyTorch SDPA", adapter_manifest=manifest)
        (root / "receipt.json").write_text(json.dumps(receipt))
        (root / "data.json").write_text(json.dumps({"rights": smoke.DATA_RIGHTS, "train": smoke.TRAIN, "development": smoke.DEVELOPMENT}))
        return root, receipt

    def test_valid_bundle_reports_integrity_only_even_when_loss_worsens(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = self.make_bundle(tmp)
            result = verify.verify_bundle(root, "a"*64)
            self.assertEqual(result["status"], "INTEGRITY_CHECKED")
            self.assertEqual(result["safetensors"]["tensor_count"], 112)
            self.assertFalse(result["capability_improvement_established"])

    def test_tamper_source_partial_and_nonfinite_receipts_fail(self):
        mutations = [("source_sha256", "b"*64), ("completed_steps", 19), ("status", "RUNNING"),
                     ("base_development_loss", float("inf")), ("gpu", {"cuda_available": False})]
        for key, value in mutations:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                root, receipt = self.make_bundle(tmp)
                receipt[key] = value
                (root / "receipt.json").write_text(json.dumps(receipt))
                with self.assertRaises(ValueError): verify.verify_bundle(root, "a"*64)

    def test_weights_hash_and_unmanifested_file_are_rejected(self):
        for tamper in ("weights", "extra"):
            with self.subTest(tamper=tamper), tempfile.TemporaryDirectory() as tmp:
                root, _ = self.make_bundle(tmp)
                if tamper == "weights": (root / "adapter/adapter_model.safetensors").write_bytes(b"changed")
                else: (root / "adapter/execute.py").write_text("print('unrequested')")
                with self.assertRaises(ValueError): verify.verify_bundle(root, "a"*64)

    def test_traversal_and_symlink_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, receipt = self.make_bundle(tmp)
            for path in ("../secret", "/tmp/secret", "adapter\\secret", "adapter//adapter_config.json"):
                with self.subTest(path=path), self.assertRaises(ValueError): verify.safe_file(root, path, 1024)
            target = root / "adapter/adapter_config.json"
            target.unlink(); target.symlink_to(root / "receipt.json")
            with self.assertRaises(ValueError): verify.verify_bundle(root, "a"*64)

    def test_nonfinite_tensor_and_overlapping_offsets_rejected(self):
        with self.assertRaises(ValueError): verify.validate_safetensors(synthetic_adapter(nonfinite=True))
        raw = synthetic_adapter()
        size = struct.unpack("<Q", raw[:8])[0]
        header = json.loads(raw[8:8+size])
        list(header.values())[1]["data_offsets"] = [0, 32]
        encoded = json.dumps(header).encode()
        with self.assertRaises(ValueError): verify.validate_safetensors(struct.pack("<Q", len(encoded))+encoded+raw[8+size:])

    def test_duplicate_json_keys_rejected(self):
        with self.assertRaises(ValueError): verify.parse_json('{"status":"FAILED","status":"COMPLETED"}')


if __name__ == "__main__": unittest.main()
