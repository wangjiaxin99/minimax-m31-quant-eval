import math
import contextlib
import io
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np

from m31eval.sparse_live import measure, top_ids, prepare, score_json, main


class SparseLiveTests(unittest.TestCase):
    def test_reference_tie_selection_is_deterministic(self):
        lp=np.array([-3.,-1.,-1.,-1.,-2.,-3.])
        self.assertEqual(top_ids(lp,2).tolist(),[1,2])
        self.assertEqual(top_ids(lp,4).tolist(),[1,2,3,4])
        self.assertEqual(top_ids(lp,6).tolist(),[1,2,3,4,0,5])

    def test_tail_redistribution_lost_only_by_coarsening(self):
        p=np.log([.99999,.0000099,.0000001])
        q=np.log([.99999,.0000001,.0000099])
        self.assertAlmostEqual(measure(p[:1],math.log(.00001),q[:1],math.log(.00001)),0.)
        self.assertAlmostEqual(measure(p,-math.inf,q,-math.inf),.0000098*math.log(99),places=12)

    def test_high_confidence_position_still_detects_top1_drift(self):
        score=measure(np.log([.999999]),math.log(.000001),np.log([.1]),math.log(.9))
        self.assertGreater(score,2.3)

    def test_infinite_kl_serializes_without_dropping_position(self):
        result = {"positions": [{"kl": math.inf}, {"kl": 0.1}]}
        encoded = json.dumps(score_json(result), allow_nan=False)
        self.assertEqual(json.loads(encoded)["positions"][0]["kl"], "Infinity")
        self.assertEqual(len(json.loads(encoded)["positions"]), 2)

    def test_cli_defaults_to_fixed_2000_without_coverage_audit(self):
        with patch("sys.argv", ["sparse_live", "prepare", "--reference", "ref", "--output", "out"]), patch("m31eval.sparse_live.prepare") as action:
            main()
            args = action.call_args.args[0]
            self.assertEqual(args.top_k, 2000)
            self.assertFalse(hasattr(args, "coverage"))
            self.assertFalse(hasattr(args, "k_audit"))

    def test_prepare_keeps_2000_ids_at_every_high_confidence_position(self):
        # The old minimum-tail rule could erase Top-1 for this reference.
        # Fixed K must keep all 2000 IDs on all 32 positions even for tiny tails.
        with tempfile.TemporaryDirectory() as tmp:
            ref = Path(tmp)/"ref"
            ref.mkdir()
            (ref/"run.json").write_text(json.dumps({"signature":"test-reference", "dataset":[{"id":"sample", "domain":"test"}]}))
            a = np.memmap(ref/"full.f64", dtype="<f8", mode="w+", shape=(32,200064))
            a[:] = math.log(1e-7/200063)
            a[:,0] = math.log1p(-1e-7)
            a.flush()
            del a
            cache = {"generation":{"token_ids":[0]*32}, "probability_arrays":{"shape":[32,200064], "logprobs":"full.f64"}}
            out = Path(tmp)/"prepared"
            with patch("m31eval.sparse_live.load_sample", return_value=cache), contextlib.redirect_stdout(io.StringIO()):
                prepare(SimpleNamespace(reference=ref, output=out, top_k=2000))
            meta = json.loads((out/"run.json").read_text())
            self.assertEqual(meta["selection"], "fixed_top_k")
            self.assertEqual(meta["records"][0]["k"], [2000]*32)
            with np.load(out/meta["records"][0]["file"]) as data:
                self.assertTrue(np.all(data["tail"] < math.log(1e-5)))
                for pos in range(32):
                    lo, hi = data["offsets"][pos:pos+2]
                    self.assertEqual(data["token_ids"][lo:hi].tolist(), list(range(2000)))
                self.assertEqual(data["top1"].tolist(), [0]*32)

    def test_true_zero_candidate_is_infinite_not_dropped(self):
        self.assertEqual(measure(np.log([.9]),math.log(.1),np.array([-math.inf]),0.),math.inf)


if __name__=='__main__':
    unittest.main()
