import math
import unittest

from m31eval.sparse_experiment import compress, reference_partition, sparse_kl


def lp(values):
    return {i: math.log(v) if v else -math.inf for i, v in enumerate(values)}


class SparseExperimentTests(unittest.TestCase):
    def test_high_confidence_keeps_top1_and_detects_degradation(self):
        p, q = lp([1-1e-12, 5e-13, 5e-13]), lp([.9, .05, .05])
        ids = reference_partition(p, 1)
        self.assertEqual(ids, (0,))
        a, b = compress(p, ids), compress(q, ids)
        self.assertTrue(math.isfinite(a.tail_logprob))
        self.assertAlmostEqual(sparse_kl(a, b), -math.log(.9), places=9)

    def test_exact_zero_reference_tail_still_scores_position(self):
        p, q = lp([1, 0, 0]), lp([.9, .05, .05])
        ids = reference_partition(p, 1)
        self.assertEqual(compress(p, ids).tail_logprob, -math.inf)
        self.assertAlmostEqual(sparse_kl(compress(p, ids), compress(q, ids)), -math.log(.9))

    def test_zero_candidate_support_is_infinite_not_skipped(self):
        p, q = lp([.9, .05, .05]), lp([1, 0, 0])
        ids = reference_partition(p, 1)
        self.assertEqual(sparse_kl(compress(p, ids), compress(q, ids)), math.inf)

    def test_tail_redistribution_can_be_hidden(self):
        p, q = lp([.8, .15, .05]), lp([.8, .05, .15])
        a = sparse_kl(compress(p, (0,)), compress(q, (0,)))
        b = sparse_kl(compress(p, (0, 1, 2)), compress(q, (0, 1, 2)))
        self.assertAlmostEqual(a, 0)
        self.assertGreater(b, .1)

    def test_candidate_must_use_reference_partition(self):
        p, q = lp([.7, .2, .1]), lp([.1, .2, .7])
        with self.assertRaisesRegex(ValueError, 'same partition'):
            sparse_kl(compress(p, reference_partition(p, 1)), compress(q, reference_partition(q, 1)))

    def test_large_mass_error_is_rejected_small_correction_is_recorded(self):
        with self.assertRaisesRegex(ValueError, 'mass'):
            compress(lp([.9, .9]), (0,))
        p = lp([.7*(1+1e-7), .3*(1+1e-7)])
        record = compress(p, (0,))
        self.assertAlmostEqual(record.normalization_log_mass, math.log1p(1e-7), places=14)

    def test_invalid_complete_vector_or_ids_rejected(self):
        with self.assertRaises(ValueError):
            compress({0: 0, 2: -math.inf}, (0,))
        with self.assertRaises(ValueError):
            compress(lp([.7, .3]), (0, 0))
        with self.assertRaises(ValueError):
            compress({0: math.nan}, (0,))

    def test_refinement_is_monotone_and_full_partition_is_full_kl(self):
        p, q = lp([.5, .3, .15, .05]), lp([.35, .2, .25, .2])
        results = []
        for k in (1, 2, 3, 4):
            ids=reference_partition(p,k)
            results.append(sparse_kl(compress(p,ids),compress(q,ids)))
        self.assertTrue(all(a <= b+1e-12 for a,b in zip(results,results[1:])))
        expected=sum(math.exp(p[i])*(p[i]-q[i]) for i in p)
        self.assertAlmostEqual(results[-1], expected)


if __name__ == '__main__':
    unittest.main()
