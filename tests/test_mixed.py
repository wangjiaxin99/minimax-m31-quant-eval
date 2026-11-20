import collections
import copy
import math
import unittest
from pathlib import Path

from m31eval.common import read_jsonl
from m31eval.metrics import cluster_bootstrap, domain_bootstrap
from m31eval.mixed import balanced_pick, set_splits, strip_gutenberg, validate_inputs


class MixedTests(unittest.TestCase):
    def test_stratified_selection_does_not_depend_on_rewards(self):
        rows = [{"id": f"{task}-{stage}", "task": str(task), "group": task % 3,
                 "reward": task % 2} for task in range(12) for stage in range(2)]
        a = balanced_pick(rows, 9, 31, lambda r: r["group"], lambda r: r["task"])
        for row in rows:
            row["reward"] = 1 - row["reward"]
        b = balanced_pick(rows, 9, 31, lambda r: r["group"], lambda r: r["task"])
        self.assertEqual([r["id"] for r in a], [r["id"] for r in b])
        self.assertEqual(len({r["task"] for r in a}), 9)
        self.assertEqual(collections.Counter(r["group"] for r in a), {0: 3, 1: 3, 2: 3})

    def test_domain_macro_weights_domains_not_window_counts(self):
        rows = [{"id": "a", "task": "a", "domain": "coding", "v": 0},
                {"id": "b", "task": "b", "domain": "coding", "v": 0},
                {"id": "c0", "task": "c", "domain": "long", "v": 1},
                {"id": "c1", "task": "c", "domain": "long", "v": 1}]
        self.assertAlmostEqual(cluster_bootstrap(rows, "v")["mean"], 1/3)
        self.assertEqual(domain_bootstrap(rows, "v")["mean"], .5)

    def test_source_task_windows_stay_in_one_split(self):
        rows = [{"id": f"{i}-{j}", "task": str(i), "domain": "long",
                 "messages": [{"role": "user", "content": f"{i}-{j}"}]}
                for i in range(10) for j in range(3)]
        set_splits(rows, {"1", "2"}, 31)
        validate_inputs(rows, expected=30)
        self.assertTrue(all(r["split"] == "development" for r in rows if r["task"] in {"1", "2"}))
        broken = copy.deepcopy(rows)
        broken[0]["split"] = "holdout" if broken[1]["split"] == "development" else "development"
        with self.assertRaisesRegex(ValueError, "leaks"):
            validate_inputs(broken, expected=30)

    def test_gutenberg_license_wrapper_not_inserted_into_model_prompt(self):
        text = "license\n*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\nbody\n*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\nterms"
        self.assertEqual(strip_gutenberg(text), "body")
        with self.assertRaises(ValueError):
            strip_gutenberg("unrecognized wrapper")

    def test_frozen_mixed_dataset_has_real_sources_and_balanced_coverage(self):
        path = Path(__file__).resolve().parents[1] / "data/mixed/quick300.jsonl"
        rows = read_jsonl(path)
        validate_inputs(rows)
        self.assertEqual(collections.Counter(r["domain"] for r in rows),
            {"deepswe": 60, "terminal_bench": 60, "math": 60, "non_latin": 60, "long_document": 60})
        self.assertEqual(len({r["task"] for r in rows}), 260)
        multilingual = [r for r in rows if r["domain"] == "non_latin"]
        self.assertEqual(collections.Counter(r["language"] for r in multilingual),
            {lang: 10 for lang in ["ar", "bn", "hi", "ja", "ko", "zh"]})
        self.assertEqual(len({r["source_sample_id"] for r in multilingual}), 60)
        for row in rows:
            self.assertTrue(row["source"])
            self.assertTrue(row["license"])
            self.assertNotIn("expected_answer", row)
            self.assertNotIn("generated_solution", row)
            self.assertNotIn("source_reward", row)
