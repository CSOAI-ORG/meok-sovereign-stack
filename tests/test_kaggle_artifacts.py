import copy
import json
from pathlib import Path
import tempfile
import unittest

from kaggle_bridge.artifacts import ArtifactError, canonical, contract_from_source, sha, strict_json, verify_smoke


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "reviewed.py"
        self.source.write_text("MODEL_ID = 'example/base'\nMODEL_REVISION = '" + "a" * 40 + "'\n"
                               "DEPENDENCIES = {'transformers': '4.57.1'}\n"
                               "TRAIN = [('train-1', 'training prompt', 'answer')]\n"
                               "DEVELOPMENT = [('dev-1', 'development prompt', 'answer')]\n"
                               "DATA_RIGHTS = {'origin': 'synthetic fixture'}\n"
                               "raise RuntimeError('source must never execute')\n")
        self.contract = contract_from_source(self.source)
        self.run = self.root / "run"
        (self.run / "adapter").mkdir(parents=True)
        self.data = {"rights": {"origin": "synthetic fixture"},
                     "train": [["train-1", "training prompt", "answer"]],
                     "development": [["dev-1", "development prompt", "answer"]]}
        (self.run / "data.json").write_bytes(canonical(self.data))
        # Opaque byte fixture: integrity verification does not deserialize weights.
        (self.run / "adapter/adapter_model.safetensors").write_bytes(b"test-only opaque bytes")
        (self.run / "adapter/adapter_config.json").write_bytes(canonical({
            "base_model_name_or_path": "example/base", "revision": "a" * 40, "peft_type": "LORA"}))
        self.receipt = {"schema": "meok.lora-smoke/1.0", "status": "COMPLETED",
                        "source_sha256": self.contract["source_sha256"], "model": self.contract["model"],
                        "published": False, "capability_improvement_established": False,
                        "requested_steps": 10, "completed_steps": 10, "train_losses": [2.0] * 10,
                        "base_development_loss": 2.0, "adapter_development_loss": 1.9,
                        "gpu": {"cuda_available": True, "name": "fixture GPU", "cuda_runtime": "12.0"},
                        "versions": self.contract["dependencies"],
                        "dataset": {"rights": self.data["rights"], "digests": {
                            k: self.contract["dataset"][k]["sha256"] for k in ("train", "development")},
                            "train_n": 1, "development_n": 1}, "adapter_manifest": []}
        self.update_manifest()

    def update_manifest(self):
        self.receipt["adapter_manifest"] = [{"path": p.relative_to(self.run).as_posix(),
                                             "bytes": p.stat().st_size, "sha256": sha(p.read_bytes())}
                                            for p in sorted((self.run / "adapter").iterdir())]

    def verify(self):
        (self.run / "receipt.json").write_bytes(canonical(self.receipt))
        return verify_smoke(self.run, self.contract)

    def test_verified_bytes_do_not_authenticate_training_or_quality(self):
        result = self.verify()
        self.assertEqual(result["status"], "ARTIFACTS_VERIFIED")
        for key in ("signature_verified", "execution_authenticated", "model_load_tested", "quality_improvement_established", "published"):
            self.assertIs(result[key], False)
        self.assertEqual(result["gspc_admission"], "NOT_ELIGIBLE_SELF_PRODUCT")

    def test_source_is_parsed_not_executed(self):
        self.assertEqual(self.contract["source_sha256"], sha(self.source.read_bytes()))

    def test_changed_weights_rejected(self):
        (self.run / "adapter/adapter_model.safetensors").write_bytes(b"changed")
        with self.assertRaises(ArtifactError): self.verify()

    def test_failed_running_or_partial_run_rejected(self):
        for state in ("FAILED", "RUNNING", None):
            with self.subTest(state=state):
                self.receipt["status"] = state
                with self.assertRaises(ArtifactError): self.verify()
        self.receipt["status"] = "COMPLETED"; self.receipt["completed_steps"] = 9
        with self.assertRaises(ArtifactError): self.verify()

    def test_wrong_source_model_dependency_rejected(self):
        for key, value in (("source_sha256", "b"*64), ("model", {"id": "other/base", "revision": "a"*40}),
                           ("versions", {"transformers": "other"})):
            with self.subTest(key=key):
                original = self.receipt[key]; self.receipt[key] = value
                with self.assertRaises(ArtifactError): self.verify()
                self.receipt[key] = original

    def test_train_dev_tamper_even_with_rewritten_receipt_rejected(self):
        self.data["development"] = self.data["train"]
        (self.run / "data.json").write_bytes(canonical(self.data))
        self.receipt["dataset"]["digests"]["development"] = sha(canonical(self.data["development"]))
        with self.assertRaises(ArtifactError): self.verify()

    def test_duplicate_path_and_escape_rejected(self):
        original = copy.deepcopy(self.receipt["adapter_manifest"])
        for path in ("adapter/../../secret", "/etc/passwd", "adapter/./file", "adapter\\file",
                     "adapter//adapter_config.json", "adapter/adapter_config.json/", original[0]["path"]):
            with self.subTest(path=path):
                self.receipt["adapter_manifest"] = copy.deepcopy(original)
                self.receipt["adapter_manifest"][1]["path"] = path
                with self.assertRaises(ArtifactError): self.verify()

    def test_symlink_and_unmanifested_files_rejected(self):
        extra = self.run / "adapter/extra"
        extra.write_text("extra")
        with self.assertRaises(ArtifactError): self.verify()
        extra.unlink(); extra.symlink_to(self.source)
        with self.assertRaises(ArtifactError): self.verify()

    def test_wrong_peft_base_rejected_even_with_matching_hash(self):
        path = self.run / "adapter/adapter_config.json"
        config = json.loads(path.read_text()); config["revision"] = "b"*40
        path.write_bytes(canonical(config)); self.update_manifest()
        with self.assertRaises(ArtifactError): self.verify()

    def test_nonfinite_losses_and_false_gpu_rejected(self):
        self.receipt["train_losses"] = [True] * 10
        with self.assertRaises(ArtifactError): self.verify()
        self.receipt["train_losses"] = [2.0] * 10; self.receipt["gpu"]["cuda_available"] = False
        with self.assertRaises(ArtifactError): self.verify()

    def test_total_byte_budget_enforced(self):
        self.verify()
        with self.assertRaises(ArtifactError): verify_smoke(self.run, self.contract, max_total_bytes=4)

    def test_json_ambiguity_and_nonfinite_rejected(self):
        for value in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}'):
            with self.subTest(value=value):
                with self.assertRaises(ArtifactError): strict_json(value)

    def test_false_improvement_and_missing_file_rejected(self):
        self.receipt["capability_improvement_established"] = True
        with self.assertRaises(ArtifactError): self.verify()
        self.receipt["capability_improvement_established"] = False
        (self.run / "adapter/adapter_config.json").unlink()
        with self.assertRaises(ArtifactError): self.verify()


if __name__ == "__main__": unittest.main()
