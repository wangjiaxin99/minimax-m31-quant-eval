import math
import unittest

from m31eval.metrics import agreement, cluster_bootstrap, coarse_partition, kl
from m31eval.common import json_compatible


def lp(values):
    return {i: math.log(p) for i, p in enumerate(values)}


class MetricsTests(unittest.TestCase):
    def test_known_kl_values(self):
        p = lp([.7, .2, .1])
        self.assertAlmostEqual(kl(p, lp([.65, .25, .1]), full=True)["kl"], .007246869, places=8)
        self.assertAlmostEqual(kl(p, lp([.5, .3, .2]), full=True)["kl"], .085122826, places=8)

    def test_coarse_is_lower_bound_with_real_other_mass(self):
        p, q = lp([.7, .2, .1]), lp([.65, .25, .1])
        coarse = kl({0: p[0]}, {0: q[0]})
        self.assertGreater(coarse["kl"], 0)
        self.assertLess(coarse["kl"], kl(p, q, full=True)["kl"])
        self.assertAlmostEqual(coarse["reference_coverage"], .7)

    def test_missing_candidate_probability_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "omitted"):
            kl(lp([.7, .2]), {0: math.log(.65)})

    def test_partial_mass_cannot_be_called_full(self):
        with self.assertRaisesRegex(ValueError, "complete vocabulary"):
            kl(lp([.7, .2]), lp([.65, .25]), full=True)

    def test_probability_mass_not_renormalized(self):
        with self.assertRaises(ValueError):
            kl(lp([.7, .7]), lp([.5, .5]))

    def test_reference_partition_avoids_unresolvable_tail(self):
        rounded = {0: math.log(.9999999), 1: math.log(.0000002)}
        self.assertEqual(coarse_partition(rounded), {})
        selected = coarse_partition(lp([.7, .2, .1]))
        self.assertEqual(set(selected), {0, 1})
        self.assertAlmostEqual(sum(math.exp(v) for v in selected.values()), .9)
        self.assertEqual(kl(selected, selected)["kl"], 0)

    def test_explicit_zero_probability_is_valid_and_serializable(self):
        p = {0: 0.0, 1: -math.inf}
        q = {0: math.log(.9), 1: math.log(.1)}
        self.assertAlmostEqual(kl(p, q, full=True)["kl"], -math.log(.9))
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
