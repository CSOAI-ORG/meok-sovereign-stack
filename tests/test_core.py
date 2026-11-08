import copy
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from meok_light.core import endpoint_url, load_routes, run, save_receipt

CONFIG = load_routes(Path(__file__).resolve().parents[1] / "configs/routes.json")

class CoreTests(unittest.TestCase):
    def transport(self, url, payload=None, timeout=90):
        if url.endswith("tags"):
            return {"models": [{"name": CONFIG["routes"]["general"]["model"],
                                "digest": CONFIG["routes"]["general"]["digest"]}]}
        return {"model": payload["model"], "done": True, "message": {"content": "4"}, "eval_count": 1}

    def test_actual_response_required_and_receipt_excludes_prompt(self):
        result = run("private test question", "general", CONFIG, transport=self.transport)
        self.assertEqual(result["text"], "4")
        self.assertEqual(result["receipt"]["status"], "RUNTIME_OBSERVED")
        self.assertEqual(result["receipt"]["quality"], "UNMEASURED")
        self.assertNotIn("private test question", json.dumps(result["receipt"]))

    def test_identity_drift_fails_before_generation(self):
        calls = []
        def wrong(url, **kwargs):
            calls.append(url)
            return {"models": [{"name": "qwen3:0.6b", "digest": "0" * 64}]}
        result = run("hello", "general", CONFIG, transport=wrong)
        self.assertEqual(result["receipt"]["status"], "UNCHECKABLE")
        self.assertIsNone(result["text"])
        self.assertEqual(len(calls), 1)

    def test_empty_incomplete_wrong_model_never_success(self):
        for response in ({"done": False}, {"done": True, "model": "other"},
                         {"done": True, "model": "qwen3:0.6b", "message": {"content": " "}}):
            with self.subTest(response=response):
                def transport(url, payload=None, timeout=90):
                    return self.transport(url) if url.endswith("tags") else response
                self.assertEqual(run("hello", "general", CONFIG, transport=transport)["receipt"]["status"], "UNCHECKABLE")

    def test_network_failure_is_unreachable(self):
        def broken(*args, **kwargs):
            raise urllib.error.URLError("credential must not appear")
        result = run("hello", "general", CONFIG, transport=broken)
        self.assertEqual(result["receipt"]["status"], "UNREACHABLE")
        self.assertNotIn("credential", json.dumps(result))

    def test_endpoint_and_budget_boundaries(self):
        for url in ("http://example.com", "https://x@y.test", "https://example.com/x", "file:///etc/passwd"):
            with self.assertRaises(ValueError): endpoint_url(url)
        self.assertEqual(endpoint_url("http://127.0.0.1:11434/"), "http://127.0.0.1:11434")
        for tokens in (0, 513, True):
            with self.assertRaises(ValueError): run("hi", "general", CONFIG, max_tokens=tokens)
        with self.assertRaises(ValueError): run("hi", "nonexistent", CONFIG)

    def test_receipts_do_not_overwrite(self):
        receipt = run("hi", "general", CONFIG, transport=self.transport)["receipt"]
        with tempfile.TemporaryDirectory() as folder:
            path = save_receipt(receipt, folder)
            self.assertEqual(json.loads(path.read_text()), receipt)
            with self.assertRaises(FileExistsError): save_receipt(receipt, folder)

    def test_no_unpinned_runnable_route(self):
        config = copy.deepcopy(CONFIG)
        config["routes"]["general"]["digest"] = "latest"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.json"
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError): load_routes(path)

if __name__ == "__main__": unittest.main()
