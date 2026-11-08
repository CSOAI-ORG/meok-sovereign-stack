import importlib.util
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


smoke = load("kaggle_smoke", "training/kaggle_smoke.py")
builder = load("build_kaggle_bundle", "training/build_kaggle_bundle.py")


class TrainingContractTests(unittest.TestCase):
    def test_fixed_splits_have_distinct_ids_and_content(self):
        digests = smoke.validate_splits()
        self.assertEqual(len(smoke.TRAIN), 16)
        self.assertEqual(len(smoke.DEVELOPMENT), 6)
        self.assertNotEqual(digests["train"], digests["development"])
        self.assertEqual(len(smoke.MODEL_REVISION), 40)

    def test_duplicates_and_case_whitespace_leakage_are_rejected(self):
        row = smoke.TRAIN[0]
        for bad in [row, ("different-id", "  " + row[1].upper() + "  ", "different answer")]:
            with self.assertRaises(ValueError):
                smoke.validate_splits([row], [bad])

    def test_completed_receipt_requires_real_metrics_steps_and_adapter(self):
        good = {"base_development_loss": 2.0, "adapter_development_loss": 2.1,
                "adapter_manifest": [{"path": "adapter/adapter_model.safetensors"}],
                "completed_steps": 20, "requested_steps": 20, "gpu": {"cuda_available": True}}
        self.assertFalse(smoke.make_receipt("COMPLETED", **good)["capability_improvement_established"])
        for key, value in [("base_development_loss", math.nan), ("adapter_development_loss", None),
                           ("adapter_manifest", []), ("completed_steps", 19), ("gpu", {"cuda_available": False})]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                smoke.make_receipt("COMPLETED", **{**good, key: value})

    def test_failure_receipt_never_becomes_success(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            with patch.object(smoke, "train_run", side_effect=RuntimeError("GPU unavailable")):
                code = smoke.main(["--output", str(out), "--skip-install"])
            receipt = json.loads((out / "receipt.json").read_text())
            self.assertEqual(code, 1)
            self.assertEqual(receipt["status"], "FAILED")
            self.assertFalse(receipt["adapter_usable"])
            self.assertIsNone(receipt["completed_steps"])
            self.assertNotIn("GPU unavailable", (out / "receipt.json").read_text())

    def test_reused_output_is_refused_without_rewriting_old_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = Path(tmp) / "receipt.json"
            old.write_text("previous bytes")
            self.assertEqual(smoke.main(["--output", tmp, "--skip-install"]), 2)
            self.assertEqual(old.read_text(), "previous bytes")

    def test_bundle_metadata_is_private_and_never_selects_paid_compute(self):
        meta = builder.metadata("nicktempleman", "meok-private-smoke")
        self.assertIs(meta["is_private"], True)
        self.assertIs(meta["enable_gpu"], True)
        self.assertEqual(meta["dataset_sources"], [])
        self.assertNotIn("machine_shape", meta)
        with self.assertRaises(ValueError):
            builder.metadata("user/foreign", "private-smoke")


if __name__ == "__main__":
    unittest.main()
