import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("replay_builder", ROOT / "training/build_replay_bundle.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class ReplayBundleTests(unittest.TestCase):
    def test_private_metadata_mounts_only_expected_training_kernel(self):
        data = builder.metadata("nicktempleman", "meok-private-adapter-reload")
        self.assertIs(data["is_private"], True)
        self.assertIs(data["enable_gpu"], True)
        self.assertEqual(data["kernel_sources"], ["nicktempleman/meok-private-cuda-adapter-smoke"])
        self.assertEqual(data["dataset_sources"], [])

    def test_rendered_script_imports_without_gpu_actions(self):
        source = builder.render_script()
        namespace = {"__name__": "replay_test", "__file__": "/test/replay.py"}
        exec(compile(source, "replay.py", "exec"), namespace)
        self.assertEqual(namespace["TRAINING_SOURCE_SHA256"], builder.TRAINING_SOURCE_SHA256)
        self.assertEqual(namespace["EXPECTED_ADAPTER_SHA256"], builder.ADAPTER_SHA256)
        self.assertEqual(len(namespace["PROMPTS"]), 3)
        self.assertEqual(namespace["helpers"].MODEL_REVISION, "c1899de289a04d12100db370d81485cdf75e47ca")
        self.assertNotIn("__SMOKE_B64__", source)
        # A nonexistent mount must stop before dependency installation or model loading.
        with tempfile.TemporaryDirectory() as temp, self.assertRaises(ValueError):
            namespace["find_training_artifacts"](Path(temp))

    def test_embedded_source_tamper_is_rejected(self):
        namespace = {"__name__": "replay_test", "__file__": "/test/replay.py"}
        exec(compile(builder.render_script(), "replay.py", "exec"), namespace)
        with self.assertRaises(ValueError):
            namespace["embedded"]("bad", "cHJpbnQoJ2JhZCcp", "0"*64)


if __name__ == "__main__": unittest.main()
