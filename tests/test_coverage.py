import hashlib
import json
import unittest

from kaggle_bridge.coverage import build_coverage


STAMP = "2026-11-08T02:00:00Z"


def capture(axes):
    return json.dumps({"schema": "csoai.gspc-axes/0.5", "axes": axes,
                       "totals": {"axis_count": 999}}).encode()


def axis(name="governance", kind="model-comparison", family="gspc"):
    return {"axis": name, "kind": kind, "family": family, "status": "MEASURED",
            "accuracy": 0.99, "n": 200, "leader": "old-board-model"}


class CoverageTests(unittest.TestCase):
    def test_all_axes_dynamic_no_score_promotion(self):
        payload = capture([axis(str(i)) for i in range(35)])
        result = build_coverage(payload, observed_at=STAMP)
        self.assertEqual(result["axis_count"], 35)
        self.assertEqual(result["source_sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(result["observed_at"], STAMP)
        for row in result["axes"]:
            self.assertEqual(row["board_status"], "MEASURED")
            self.assertEqual(row["this_run"]["execution_status"], "NOT_RUN")
            self.assertEqual(row["this_run"]["measurement_status"], "UNMEASURED")
            for field in ("accuracy", "n", "leader"):
                self.assertNotIn(field, row)

    def test_declared_kind_beats_name_or_family_guess(self):
        result = build_coverage(capture([
            axis("safety", "deterministic-facts", "gspc"),
            axis("new-subject", "measured-subject", "subject"),
            axis("governance", "future-kind", "gspc")]), observed_at=STAMP)
        self.assertEqual([r["execution_family"] for r in result["axes"]],
                         ["deterministic-facts", "measured-subject", "UNSUPPORTED"])
        self.assertEqual(result["axes"][2]["declared_kind"], "future-kind")

    def test_arbitrary_reproduced_flag_is_report_only(self):
        result = build_coverage(capture([axis()]), observed_at=STAMP,
            execution_evidence={"governance": {"status": "REPRODUCED", "receipt": {"x": 1}}})
        self.assertEqual(result["axes"][0]["this_run"]["execution_status"], "REPORTED_ONLY")

    def test_validated_replay_is_never_measurement_admission(self):
        calls = []
        def validator(axis_id, evidence):
            calls.append((axis_id, evidence["receipt"]["run"]))
            return True
        result = build_coverage(capture([axis()]), observed_at=STAMP,
            execution_evidence={"governance": {"status": "REPRODUCED", "receipt": {"run": "r1"}}},
            validate_receipt=validator)
        run = result["axes"][0]["this_run"]
        self.assertEqual(calls, [("governance", "r1")])
        self.assertEqual(run["execution_status"], "REPRODUCED")
        self.assertEqual(run["measurement_status"], "UNMEASURED")

    def test_receipt_and_exact_true_required(self):
        for receipt, verdict in ((None, True), ({}, True), ({"run": "r"}, 1),
                                 ({"run": "r"}, "VALID"), ({"run": "r"}, False)):
            with self.subTest(receipt=receipt, verdict=verdict):
                result = build_coverage(capture([axis()]), observed_at=STAMP,
                    execution_evidence={"governance": {"status": "REPRODUCED", "receipt": receipt}},
                    validate_receipt=lambda *_: verdict)
                self.assertEqual(result["axes"][0]["this_run"]["execution_status"], "REPORTED_ONLY")

    def test_validator_failure_is_uncheckable(self):
        def unavailable(*_):
            raise OSError("not reachable")
        result = build_coverage(capture([axis()]), observed_at=STAMP,
            execution_evidence={"governance": {"status": "RUNTIME_OBSERVED", "receipt": {"run": "r"}}},
            validate_receipt=unavailable)
        self.assertEqual(result["axes"][0]["this_run"]["receipt_validation"], "UNCHECKABLE")

    def test_unsupported_kind_or_admission_status_not_promoted(self):
        for kind, status in (("new-kind", "REPRODUCED"), ("model-comparison", "MEASURED")):
            result = build_coverage(capture([axis(kind=kind)]), observed_at=STAMP,
                execution_evidence={"governance": {"status": status, "receipt": {"run": "r"}}},
                validate_receipt=lambda *_: True)
            self.assertEqual(result["axes"][0]["this_run"]["execution_status"], "REPORTED_ONLY")

    def test_duplicate_or_missing_axis_ids_rejected(self):
        for axes in ([axis(), axis()], [{}], ["governance"]):
            with self.subTest(axes=axes), self.assertRaises(ValueError):
                build_coverage(capture(axes), observed_at=STAMP)

    def test_unknown_evidence_ids_are_visible(self):
        result = build_coverage(capture([axis()]), observed_at=STAMP,
                                execution_evidence={"goverance": {"status": "REPRODUCED"}})
        self.assertEqual(result["unused_evidence_axes"], ["goverance"])

    def test_raw_digest_changes_with_capture_bytes(self):
        payload = capture([axis()])
        one = build_coverage(payload, observed_at=STAMP)
        two = build_coverage(payload + b"\n", observed_at=STAMP)
        self.assertNotEqual(one["source_sha256"], two["source_sha256"])
        self.assertEqual(one["axes"], two["axes"])

    def test_schema_timestamp_and_source_fail_closed(self):
        for payload, stamp, url in ((b'{"schema":"future","axes":[]}', STAMP, "https://example.test"),
                                   (capture([]), "2026-11-08", "https://example.test"),
                                   (capture([]), STAMP, "https://user:secret@example.test")):
            with self.subTest(payload=payload, stamp=stamp), self.assertRaises(ValueError):
                build_coverage(payload, observed_at=stamp, source_url=url)

    def test_duplicate_json_keys_rejected_including_nested_axis(self):
        for payload in (
            b'{"schema":"csoai.gspc-axes/0.5","axes":[],"axes":[]}',
            b'{"schema":"csoai.gspc-axes/0.5","axes":[{"axis":"a","axis":"b"}]}',
        ):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                build_coverage(payload, observed_at=STAMP)

    def test_nonfinite_json_numbers_rejected_even_in_unused_fields(self):
        for value in (b"NaN", b"Infinity", b"-Infinity", b"1e999", b"-1e999"):
            payload = b'{"schema":"csoai.gspc-axes/0.5","axes":[],"unused":' + value + b'}'
            with self.subTest(value=value), self.assertRaises(ValueError):
                build_coverage(payload, observed_at=STAMP)


if __name__ == "__main__":
    unittest.main()
