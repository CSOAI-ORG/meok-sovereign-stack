"""Contract extraction and CLI binding tests; no replay or model is executed."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kaggle_bridge.__main__ import main
from kaggle_bridge.artifacts import canonical, sha
from kaggle_bridge.reload import contract_from_replay_source, verify_reload


GENERATION = {"seed": 73, "do_sample": False, "max_new_tokens": 96, "enable_thinking": False}


class ReplaySourceContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / "reviewed_replay.py"

    def source(self, receipt_literal=None, extra=""):
        literal = receipt_literal or repr({"schema": "meok.adapter-reload/1.0", "generation": GENERATION})
        raw = ("TRAINING_SOURCE_SHA256 = '" + "b" * 64 + "'\n"
               "EXPECTED_ADAPTER_SHA256 = '" + "c" * 64 + "'\n"
               "PROMPTS = [{'id': 'p1', 'text': 'Explain café bytes.'}]\n"
               "def main():\n    receipt = " + literal + "\n" + extra)
        self.path.write_text(raw, encoding="utf-8")
        return raw.encode()

    def test_literal_pins_extracted_without_executing_top_level_code(self):
        marker = self.root / "must-not-exist"
        raw = self.source(extra="from pathlib import Path\nPath(" + repr(str(marker)) +
                          ").write_text('executed')\nraise RuntimeError('must never execute')\n")
        result = contract_from_replay_source(self.path)
        self.assertFalse(marker.exists())
        self.assertEqual(result["replay_sha256"], sha(raw))
        self.assertEqual(result["training_source_sha256"], "b" * 64)
        self.assertEqual(result["adapter_sha256"], "c" * 64)
        self.assertEqual(result["prompts"], [["p1", "Explain café bytes."]])
        self.assertEqual(result["generation"], GENERATION)

    def test_nonliteral_pin_expression_not_evaluated(self):
        raw = self.source().decode().replace("PROMPTS = [", "PROMPTS = make_prompts([").replace(
            "'Explain café bytes.'}]", "'Explain café bytes.'}])")
        self.path.write_text(raw, encoding="utf-8")
        with self.assertRaises(ValueError): contract_from_replay_source(self.path)

    def test_reassigned_literal_pin_rejected(self):
        self.source(extra="EXPECTED_ADAPTER_SHA256 = '" + "d" * 64 + "'\n")
        with self.assertRaises(ValueError): contract_from_replay_source(self.path)

    def test_two_receipt_generation_contracts_are_ambiguous(self):
        self.source(extra="other = " + repr({"schema": "meok.adapter-reload/1.0", "generation": GENERATION}) + "\n")
        with self.assertRaises(ValueError): contract_from_replay_source(self.path)

    def test_no_receipt_generation_contract_rejected(self):
        self.source(receipt_literal="{'schema': 'unrelated', 'generation': {}}")
        with self.assertRaises(ValueError): contract_from_replay_source(self.path)

    def test_duplicate_generation_field_rejected_even_if_values_equal(self):
        literal = "{'schema': 'meok.adapter-reload/1.0', 'generation': " + repr(GENERATION)
        self.source(receipt_literal=literal + ", 'generation': " + repr(GENERATION) + "}")
        with self.assertRaises(ValueError): contract_from_replay_source(self.path)

    def test_dict_unpacking_cannot_override_generation_contract(self):
        literal = "{'schema': 'meok.adapter-reload/1.0', 'generation': " + repr(GENERATION) + ", **overrides}"
        self.source(receipt_literal=literal)
        with self.assertRaises(ValueError): contract_from_replay_source(self.path)

    def test_duplicate_nested_generation_setting_rejected(self):
        generation = "{'seed': 73, 'do_sample': False, 'max_new_tokens': 96, 'max_new_tokens': 96, 'enable_thinking': False}"
        self.source(receipt_literal="{'schema': 'meok.adapter-reload/1.0', 'generation': " + generation + "}")
        with self.assertRaises(ValueError): contract_from_replay_source(self.path)

    def test_invalid_generation_contract_types_rejected_at_extraction(self):
        for generation in ([], {**GENERATION, "do_sample": 0}, {**GENERATION, "seed": True},
                           {**GENERATION, "max_new_tokens": 0}, {**GENERATION, "enable_thinking": 1}):
            with self.subTest(generation=generation):
                self.source(receipt_literal=repr({"schema": "meok.adapter-reload/1.0", "generation": generation}))
                with self.assertRaises(ValueError): contract_from_replay_source(self.path)

    def test_duplicate_prompt_id_rejected(self):
        raw = self.source().decode().replace(
            "PROMPTS = [{'id': 'p1', 'text': 'Explain café bytes.'}]",
            "PROMPTS = [{'id': 'p1', 'text': 'First'}, {'id': 'p1', 'text': 'Second'}]")
        self.path.write_text(raw, encoding="utf-8")
        with self.assertRaises(ValueError): contract_from_replay_source(self.path)

    def test_nonhex_expected_pin_rejected(self):
        raw = self.source().decode().replace("'" + "c" * 64 + "'", "'moving-main'")
        self.path.write_text(raw, encoding="utf-8")
        with self.assertRaises(ValueError): contract_from_replay_source(self.path)


class ReloadCliBindingTests(unittest.TestCase):
    """Isolate the already-tested training verifier; exercise real reload and CLI."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = self.root / "training"
        self.run.mkdir()
        self.model = {"id": "example/base", "revision": "d" * 40, "licence": "apache-2.0"}
        self.training = {"schema": "meok.lora-smoke/1.0", "status": "COMPLETED", "source_sha256": "b" * 64,
                         "model": self.model, "adapter_manifest": [{"path": "adapter/adapter_model.safetensors",
                                                                     "sha256": "c" * 64, "bytes": 32}]}
        training_raw = canonical(self.training)
        (self.run / "receipt.json").write_bytes(training_raw)
        self.checked = {"schema": "meok.kaggle-artifact-check/1", "status": "ARTIFACTS_VERIFIED",
                        "source_sha256": "b" * 64, "receipt_sha256": sha(training_raw),
                        "receipt_canonical_sha256": sha(training_raw),
                        "model": {k: self.model[k] for k in ("id", "revision")}}
        self.expected = {"schema": "meok.kaggle-reload-contract/1", "replay_sha256": "a" * 64,
                         "training_source_sha256": "b" * 64, "adapter_sha256": "c" * 64,
                         "prompts": [["p1", "Prompt"]], "generation": copy.deepcopy(GENERATION)}
        self.receipt = {"schema": "meok.adapter-reload/1.0", "status": "COMPLETED",
                        "replay_source_sha256": "a" * 64, "training_source_sha256": "b" * 64,
                        "expected_adapter_sha256": "c" * 64, "model": self.model,
                        "adapter_loaded": True, "training_performed": False, "published": False,
                        "capability_improvement_established": False, "successful_generations": 1,
                        "duration_seconds": 1.5, "generation": copy.deepcopy(GENERATION),
                        "artifact_integrity": {"schema": "meok.adapter-integrity/1.0", "status": "INTEGRITY_CHECKED",
                            "receipt_sha256": sha(training_raw), "source_sha256": "b" * 64, "model": self.model},
                        "outputs": [{"id": "p1", "text": "Prompt", "status": "RUNTIME_OBSERVED",
                            "prompt_sha256": sha(b"Prompt"), "output": "Answer", "output_sha256": sha(b"Answer"),
                            "generated_tokens": 1, "duration_seconds": 0.5}]}
        self.contract_path = self.root / "training-contract.json"
        self.contract_path.write_text('{}')
        self.expected_path = self.root / "replay-contract.json"
        self.receipt_path = self.root / "reload-receipt.json"
        self.receipt_path.write_bytes(canonical(self.receipt))
        self.output = self.root / "checked.json"
        # Establish this receipt passes the real verifier before testing CLI pin binding.
        self.assertEqual(verify_reload(self.receipt, self.checked, self.training,
                         replay_sha256="a" * 64, prompts=self.expected["prompts"], generation=GENERATION)["status"],
                         "RELOAD_RECEIPT_VERIFIED")

    def run_cli(self, expected_raw=None):
        self.expected_path.write_bytes(canonical(self.expected) if expected_raw is None else expected_raw)
        out, err = io.StringIO(), io.StringIO()
        with patch("kaggle_bridge.__main__.verify_smoke", return_value=self.checked) as check:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                status = main(["verify-reload", "--run", str(self.run), "--contract", str(self.contract_path),
                               "--receipt", str(self.receipt_path), "--replay-contract", str(self.expected_path),
                               "--output", str(self.output)])
            check.assert_called_once_with(self.run, {})
        return status, out.getvalue(), err.getvalue()

    def test_matching_contract_emits_only_report_verification(self):
        code, _, err = self.run_cli()
        self.assertEqual((code, err), (0, ""))
        result = json.loads(self.output.read_text())
        self.assertEqual(result["status"], "RELOAD_RECEIPT_VERIFIED")
        self.assertIs(result["execution_authenticated"], False)

    def test_valid_reload_rejected_when_expected_adapter_is_different(self):
        self.expected["adapter_sha256"] = "f" * 64
        code, out, err = self.run_cli()
        self.assertEqual((code, out), (2, ""))
        self.assertFalse(self.output.exists())
        self.assertIn("adapter mismatch", json.loads(err)["reason"])

    def test_wrong_expected_training_source_rejected_before_output(self):
        self.expected["training_source_sha256"] = "f" * 64
        code, out, err = self.run_cli()
        self.assertEqual((code, out), (2, ""))
        self.assertFalse(self.output.exists())
        self.assertIn("source mismatch", json.loads(err)["reason"])

    def test_duplicate_expected_json_pin_rejected_without_success_file(self):
        encoded = canonical(self.expected)
        raw = encoded[:-1] + b',"adapter_sha256":"' + b"c" * 64 + b'"}'
        code, out, err = self.run_cli(expected_raw=raw)
        self.assertEqual((code, out), (2, ""))
        self.assertFalse(self.output.exists())
        self.assertIn("duplicate JSON key", json.loads(err)["reason"])


if __name__ == "__main__":
    unittest.main()
