import contextlib
import io
import json
import math
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from m31eval.cli import parser
from m31eval.quick import capture, compare, evaluate
from m31eval.api import Server


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.reply({"data": [{"id": x} for x in ["reference", "same", "quant", "bad-template", "omits-ids", "zero-tail"]]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        model = body["model"]
        if self.path == "/tokenize":
            self.reply({"tokens": [91, 93 if model == "bad-template" else 92]})
        elif self.path == "/detokenize":
            self.reply({"prompt": "".join(str(i) for i in body["tokens"])})
        elif self.path == "/v1/completions":
            n = body["max_tokens"]
            ids = [1] * n
            probabilities = [.7, .2, .1] if model in ["reference", "same"] else [.65, .25, .1]
            if model == "zero-tail":
                probabilities = [.8, .2, 0]
            requested = body.get("logprob_token_ids")
            keep = list(range(1, 4)) if requested is None else requested
            if model == "omits-ids":
                keep = keep[:1]
            distributions = [{f"token_id:{i}": math.log(probabilities[i-1]) if probabilities[i-1] else -math.inf
                              for i in keep} for _ in ids]
            self.reply({"choices": [{"token_ids": ids, "text": "test", "finish_reason": "length",
                "logprobs": {"tokens": ["token_id:1"] * n, "top_logprobs": distributions}}]})
        else:
            self.reply({"choices": [{"message": {"content": "Beijing", "tool_calls": []}, "finish_reason": "stop"}]})


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}/v1"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def pipeline(self, metric="full", candidate="quant"):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        dataset = root / "data.jsonl"
        dataset.write_text('\n'.join(json.dumps({"id": f"x{i}", "task": f"task{i}", "prompt": "question"}) for i in range(2)))
        args = parser().parse_args(["reference", "--dataset", str(dataset), "--base-url", self.url,
            "--model", "reference", "--reference-kind", "bf16", "--metric", metric, "--tokens", "4",
            "--positions", "0", "1", "2", "3", "--output", str(root / "ref")])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(capture(args), 0)
            for name in ["same", candidate]:
                ev = parser().parse_args(["evaluate", "--reference", str(root / "ref"), "--base-url", self.url,
                     "--model", name, "--output", str(root / name)])
                status = evaluate(ev)
        return root, status

    def test_defaults_use_full_vocabulary_and_all_32_positions(self):
        args = parser().parse_args(["reference", "--dataset", "data.jsonl", "--base-url", self.url,
                                   "--reference-kind", "mxfp8", "--output", "ref"])
        self.assertEqual(args.metric, "full")
        self.assertEqual(args.positions, list(range(32)))

    def test_old_reference_metric_is_rejected(self):
        root, _ = self.pipeline()
        path = root / "ref/run.json"
        metadata = json.loads(path.read_text())
        metadata["metric"] = "legacy_partition"
        path.write_text(json.dumps(metadata))
        args = parser().parse_args(["evaluate", "--reference", str(root / "ref"),
                                   "--base-url", self.url, "--model", "quant", "--output", str(root / "new")])
        with self.assertRaisesRegex(ValueError, "recapture a full-vocabulary reference"):
            evaluate(args)

    def test_native_full_kl_and_paired_comparison(self):
        root, status = self.pipeline("full")
        self.assertEqual(status, 0)
        summary = json.loads((root / "quant/summary.json").read_text())
        self.assertAlmostEqual(summary["prompt_mean_kl"], .007246869, places=8)
        rows = [json.loads(line) for line in (root / "quant/samples.jsonl").read_text().splitlines()]
        for row in rows:
            for position in row["scored_positions"]:
                self.assertEqual(position["candidate_vocab_size"], 3)
                self.assertNotIn("candidate_logprobs", position)
                self.assertEqual(len(position["candidate_distribution_sha256"]), 64)
        with contextlib.redirect_stdout(io.StringIO()):
            compare(root / "same", root / "quant", root / "comparison.json")
        report = json.loads((root / "comparison.json").read_text())
        self.assertEqual(report["conclusion"], "higher measured divergence")
        with contextlib.redirect_stdout(io.StringIO()):
            compare(root / "same", root / "quant", root / "noise_comparison.json", noise_floor=.01)
        self.assertEqual(json.loads((root / "noise_comparison.json").read_text())["conclusion"], "inconclusive")

    def test_missing_requested_ids_is_not_silently_approximated(self):
        root, status = self.pipeline("full", "omits-ids")
        self.assertEqual(status, 2)
        summary = json.loads((root / "omits-ids/summary.json").read_text())
        self.assertEqual(summary["status"], "incomplete")
        self.assertIsNone(summary["prompt_mean_kl"])

    def test_zero_candidate_probability_still_fails_full_kl(self):
        root, status = self.pipeline("full", "zero-tail")
        self.assertEqual(status, 2)
        self.assertEqual(json.loads((root / "zero-tail/summary.json").read_text())["status"], "incomplete")

    def test_interrupted_samples_cannot_produce_improvement_claim(self):
        root, _ = self.pipeline("full")
        path = root / "same/samples.jsonl"
        path.write_text(path.read_text().splitlines()[0] + "\n")
        with self.assertRaisesRegex(ValueError, "complete evaluation runs"):
            compare(root / "same", root / "quant", root / "comparison.json")

    def test_template_mismatch_rejected(self):
        root, status = self.pipeline("agreement", "bad-template")
        self.assertEqual(status, 2)

    def test_resume_retries_only_failed_sample_in_new_round(self):
        root, _ = self.pipeline("full")
        source = root / "quant"
        samples = [json.loads(line) for line in (source / "samples.jsonl").read_text().splitlines()]
        samples[0]["error"] = "transient request failure"
        (source / "samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in samples))
        summary = json.loads((source / "summary.json").read_text())
        summary.update(status="incomplete", completed=1, errors=1)
        (source / "summary.json").write_text(json.dumps(summary))
        original_bytes = (source / "samples.jsonl").read_bytes()
        args = parser().parse_args(["evaluate", "--reference", str(root / "ref"), "--base-url", self.url,
            "--model", "quant", "--output", str(root / "recovered"), "--resume-from", str(source)])
        with patch.object(Server, "tokenize", autospec=True, side_effect=Server.tokenize) as tokenize:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(evaluate(args), 0)
            self.assertEqual(tokenize.call_count, 1)
        self.assertEqual((source / "samples.jsonl").read_bytes(), original_bytes)
        recovered = {row["id"]: row for row in map(json.loads, (root / "recovered/samples.jsonl").read_text().splitlines())}
        self.assertEqual(recovered[samples[1]["id"]], samples[1])
        summary2 = json.loads((root / "recovered/summary.json").read_text())
        self.assertEqual((summary2["completed"], summary2["errors"], summary2["reused_samples"], summary2["retried_samples"]), (2, 0, 1, 1))
        self.assertAlmostEqual(summary2["elapsed_seconds"], summary["elapsed_seconds"] + summary2["resume_elapsed_seconds"])

    def test_resume_rejects_changed_candidate_and_duplicate_samples(self):
        root, _ = self.pipeline("full")
        args = parser().parse_args(["evaluate", "--reference", str(root / "ref"), "--base-url", self.url,
            "--model", "same", "--output", str(root / "recovered"), "--resume-from", str(root / "quant")])
        with self.assertRaisesRegex(ValueError, "different candidate_model"):
            evaluate(args)
        args.model = "quant"
        path = root / "quant/samples.jsonl"
        path.write_text(path.read_text().splitlines()[0] + "\n" + path.read_text().splitlines()[0] + "\n")
        with self.assertRaisesRegex(ValueError, "unique expected samples"):
            evaluate(args)

    def test_chat_only_mode_is_explicitly_labeled(self):
        root, status = self.pipeline("chat")
        summary = json.loads((root / "quant/summary.json").read_text())
        self.assertEqual(status, 0)
        self.assertIn("not token agreement or KL", summary["metric"])
        self.assertNotIn("prompt_mean_kl", summary)


if __name__ == "__main__":
    unittest.main()
