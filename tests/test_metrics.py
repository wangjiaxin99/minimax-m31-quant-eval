import math
import unittest

from m31eval.metrics import agreement, cluster_bootstrap, kl
from m31eval.common import json_compatible


def lp(values):
    return {i: math.log(p) for i, p in enumerate(values)}


class MetricsTests(unittest.TestCase):
    def test_full_kl_keeps_near_deterministic_reference_positions(self):
        p = {0: math.log(1 - 2e-12), 1: math.log(1e-12), 2: math.log(1e-12)}
        q = {0: math.log(.9), 1: math.log(.05), 2: math.log(.05)}
        self.assertAlmostEqual(kl(p, q)["kl"], -math.log(.9), places=9)

    def test_known_kl_values(self):
        p = lp([.7, .2, .1])
        self.assertAlmostEqual(kl(p, lp([.65, .25, .1]))["kl"], .007246869, places=8)
        self.assertAlmostEqual(kl(p, lp([.5, .3, .2]))["kl"], .085122826, places=8)

    def test_missing_candidate_probability_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "omitted"):
            kl(lp([.7, .2]), {0: math.log(.65)})

    def test_partial_mass_cannot_be_called_full(self):
        with self.assertRaisesRegex(ValueError, "complete vocabulary"):
            kl(lp([.7, .2]), lp([.65, .25]))

    def test_probability_mass_not_renormalized(self):
        with self.assertRaises(ValueError):
            kl(lp([.7, .7]), lp([.5, .5]))

    def test_explicit_zero_probability_is_valid_and_serializable(self):
        p = {0: 0.0, 1: -math.inf}
        q = {0: math.log(.9), 1: math.log(.1)}
        self.assertAlmostEqual(kl(p, q)["kl"], -math.log(.9))
        encoded = json_compatible(p)
        self.assertEqual(encoded[1], "-Infinity")
        self.assertEqual(float(encoded[1]), -math.inf)

    def test_eos_denominators(self):
        short = agreement([1, 2], [1, 2])
        self.assertTrue(short["sequence_exact"])
        self.assertTrue(short["eos_before_32"])
        self.assertFalse(short["prefix_at_32"])
        long = agreement(list(range(32)), list(range(31)) + [99])
        self.assertEqual(long["prefix_tokens"], 31)
        self.assertFalse(long["prefix_at_32"])

    def test_resample_tasks_not_individual_windows(self):
        rows = [{"id": "a0", "task": "a", "v": 1}, {"id": "a1", "task": "a", "v": 1},
                {"id": "b0", "task": "b", "v": 3}]
        measured = cluster_bootstrap(rows, "v", iterations=200)
        self.assertEqual(measured["clusters"], 2)
        self.assertEqual(measured["mean"], 2)


if __name__ == "__main__":
    unittest.main()
