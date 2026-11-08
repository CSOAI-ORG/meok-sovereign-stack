import importlib.metadata
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from kaggle_bridge.nvidia_probe import probe


class NvidiaProbeTests(unittest.TestCase):
    @patch("kaggle_bridge.nvidia_probe.importlib.metadata.version", side_effect=importlib.metadata.PackageNotFoundError)
    @patch("kaggle_bridge.nvidia_probe.subprocess.run", side_effect=FileNotFoundError)
    def test_no_gpu_is_not_reported_as_working(self, run, version):
        result = probe()
        self.assertEqual(result["hardware"]["status"], "UNCHECKABLE")
        self.assertEqual(result["torch_cuda"]["status"], "NOT_RUN")
        self.assertTrue(all(v is None for v in result["packages"].values()))
        self.assertEqual(run.call_count, 1)

    @patch("kaggle_bridge.nvidia_probe.importlib.metadata.version", return_value="fixture")
    @patch("kaggle_bridge.nvidia_probe.subprocess.run")
    def test_installed_package_does_not_make_failed_probe_work(self, run, version):
        run.side_effect = [SimpleNamespace(returncode=0, stdout="T4, 1, 16000, 7.5"),
                           SimpleNamespace(returncode=1, stdout=""), SimpleNamespace(returncode=1, stdout="")]
        result = probe(exercise=True)
        self.assertEqual(result["hardware"]["status"], "RUNTIME_OBSERVED")
        self.assertEqual(result["torch_cuda"]["status"], "UNCHECKABLE")
        self.assertEqual(result["rapids_cudf"]["status"], "UNCHECKABLE")
        self.assertEqual(result["model_inference"], "NOT_RUN")


if __name__ == "__main__": unittest.main()
