import copy
import unittest

from kaggle_bridge.artifacts import ArtifactError, canonical, sha
from kaggle_bridge.reload import verify_reload


class ReloadTests(unittest.TestCase):
    def setUp(self):
        self.replay_sha = "a" * 64
        self.source_sha = "b" * 64
        self.adapter_sha = "c" * 64
        self.prompts = [("p1", "Explain café bytes."), ("p2", "Name a limitation.")]
        self.generation = {"seed": 73, "do_sample": False, "max_new_tokens": 96, "enable_thinking": False}
        self.model = {"id": "example/model", "revision": "d" * 40, "licence": "apache-2.0"}
        self.training = {"schema": "meok.lora-smoke/1.0", "status": "COMPLETED",
                         "source_sha256": self.source_sha, "model": copy.deepcopy(self.model),
                         "adapter_manifest": [{"path": "adapter/adapter_model.safetensors",
                                               "sha256": self.adapter_sha, "bytes": 32}]}
        self.check = {"schema": "meok.kaggle-artifact-check/1", "status": "ARTIFACTS_VERIFIED",
                      "receipt_sha256": "e" * 64, "receipt_canonical_sha256": sha(canonical(self.training)),
                      "source_sha256": self.source_sha,
                      "model": {k: self.model[k] for k in ("id", "revision")}}
        self.receipt = {"schema": "meok.adapter-reload/1.0", "status": "COMPLETED",
                        "replay_source_sha256": self.replay_sha, "training_source_sha256": self.source_sha,
                        "expected_adapter_sha256": self.adapter_sha, "model": copy.deepcopy(self.model),
                        "adapter_loaded": True, "training_performed": False, "published": False,
                        "capability_improvement_established": False, "successful_generations": 2,
                        "duration_seconds": 2.5, "generation": copy.deepcopy(self.generation),
                        "artifact_integrity": {"schema": "meok.adapter-integrity/1.0",
                            "status": "INTEGRITY_CHECKED", "receipt_sha256": "e" * 64,
                            "source_sha256": self.source_sha, "model": copy.deepcopy(self.model)},
                        "outputs": [{"id": ident, "text": prompt, "status": "RUNTIME_OBSERVED",
                                     "prompt_sha256": sha(prompt.encode()), "output": "Reported answer.",
                                     "output_sha256": sha(b"Reported answer."), "generated_tokens": 4,
                                     "duration_seconds": 0.5} for ident, prompt in self.prompts]}

    def verify(self, **kwargs):
        args = {"replay_sha256": self.replay_sha, "prompts": self.prompts, "generation": self.generation}
        args.update(kwargs)
        return verify_reload(self.receipt, self.check, self.training, **args)

    def test_consistent_report_never_authenticates_execution_or_grades_answers(self):
        result = self.verify()
        self.assertEqual(result["status"], "RELOAD_RECEIPT_VERIFIED")
        self.assertEqual(result["producer_reported_generation_count"], 2)
        self.assertEqual(result["answer_correctness"], "UNMEASURED")
        for key in ("execution_authenticated", "independent_replay_established", "quality_improvement_established"):
            self.assertIs(result[key], False)

    def test_changed_training_receipt_cannot_rebind_adapter(self):
        self.training["adapter_manifest"][0]["sha256"] = "f" * 64
        self.receipt["expected_adapter_sha256"] = "f" * 64
        with self.assertRaises(ArtifactError): self.verify()

    def test_missing_or_failed_training_check_rejected(self):
        for key, value in (("status", "REJECTED"), ("receipt_canonical_sha256", None), ("receipt_sha256", "bad")):
            with self.subTest(key=key):
                old = self.check[key]; self.check[key] = value
                with self.assertRaises(ArtifactError): self.verify()
                self.check[key] = old

    def test_source_model_and_adapter_pin_changes_rejected(self):
        for key in ("replay_source_sha256", "training_source_sha256", "expected_adapter_sha256"):
            with self.subTest(key=key):
                old = self.receipt[key]; self.receipt[key] = "f" * 64
                with self.assertRaises(ArtifactError): self.verify()
                self.receipt[key] = old
        self.receipt["model"]["revision"] = "e" * 40
        with self.assertRaises(ArtifactError): self.verify()

    def test_expected_digests_must_be_hex(self):
        for value in ("moving-main", "z" * 64, "a" * 63, None):
            with self.subTest(value=value), self.assertRaises(ArtifactError):
                self.verify(replay_sha256=value)
        self.check["source_sha256"] = "g" * 64
        self.training["source_sha256"] = "g" * 64
        self.check["receipt_canonical_sha256"] = sha(canonical(self.training))
        with self.assertRaises(ArtifactError): self.verify()

    def test_failed_reload_and_false_effect_flags_rejected(self):
        for key, value in (("status", "RUNNING"), ("adapter_loaded", 1), ("training_performed", True),
                           ("published", 0), ("capability_improvement_established", True)):
            with self.subTest(key=key):
                old = self.receipt[key]; self.receipt[key] = value
                with self.assertRaises(ArtifactError): self.verify()
                self.receipt[key] = old

    def test_producer_integrity_report_must_bind_checked_receipt(self):
        self.receipt["artifact_integrity"]["receipt_sha256"] = "f" * 64
        with self.assertRaises(ArtifactError): self.verify()

    def test_missing_extra_or_duplicate_generations_rejected(self):
        original = copy.deepcopy(self.receipt["outputs"])
        for outputs in (original[:1], original + [original[0]], [original[0], original[0]]):
            with self.subTest(outputs=outputs):
                self.receipt["outputs"] = outputs
                with self.assertRaises(ArtifactError): self.verify()

    def test_untrusted_prompt_text_cannot_change_expected_prompt(self):
        row = self.receipt["outputs"][0]
        row["text"] = "Changed prompt"; row["prompt_sha256"] = sha(row["text"].encode())
        with self.assertRaises(ArtifactError): self.verify()

    def test_duplicate_expected_ids_rejected(self):
        with self.assertRaises(ArtifactError): self.verify(prompts=[self.prompts[0], self.prompts[0]])

    def test_wrong_hash_or_empty_output_rejected(self):
        row = self.receipt["outputs"][0]
        row["output"] = "Changed output"
        with self.assertRaises(ArtifactError): self.verify()
        row["output"] = " \n"; row["output_sha256"] = sha(row["output"].encode())
        with self.assertRaises(ArtifactError): self.verify()

    def test_generation_settings_exact_and_types_preserved(self):
        for key, value in (("seed", 73.0), ("do_sample", 0), ("max_new_tokens", 95), ("enable_thinking", True)):
            with self.subTest(key=key):
                old = self.receipt["generation"][key]; self.receipt["generation"][key] = value
                with self.assertRaises(ArtifactError): self.verify()
                self.receipt["generation"][key] = old

    def test_tokens_and_timing_are_finite_positive_and_bounded(self):
        row = self.receipt["outputs"][0]
        for value in (0, -1, 97, True, 4.0):
            with self.subTest(tokens=value):
                row["generated_tokens"] = value
                with self.assertRaises(ArtifactError): self.verify()
        row["generated_tokens"] = 4
        for value in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(duration=value):
                row["duration_seconds"] = value
                with self.assertRaises(ArtifactError): self.verify()

    def test_overall_count_duration_and_per_row_state_checked(self):
        for key, value in (("successful_generations", 2.0), ("successful_generations", 1),
                           ("duration_seconds", float("inf"))):
            with self.subTest(key=key, value=value):
                old = self.receipt[key]; self.receipt[key] = value
                with self.assertRaises(ArtifactError): self.verify()
                self.receipt[key] = old
        self.receipt["outputs"][0]["status"] = "FAILED"
        with self.assertRaises(ArtifactError): self.verify()


if __name__ == "__main__":
    unittest.main()
