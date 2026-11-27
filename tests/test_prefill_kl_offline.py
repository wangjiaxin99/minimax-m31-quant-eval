"""Offline parts of prefill-kl: analyze, render, backend checks. NumPy only; no GPU or vLLM."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from m31eval.prefill_kl import store
from m31eval.prefill_kl.analyze import analyze, kendall_tau_b, prompt_weights, report
from m31eval.prefill_kl.render import normalize, render_rows
from m31eval.prefill_kl.run import check_backends


def make_runs(root: Path, scales: dict[str, float], seed: int = 0):
    """answers.jsonl with 2 domains x 6 prompts x 2 answers, and one score run per model with KL = scale x base."""
    rng = np.random.default_rng(seed)
    answers, base = [], []
    for d in ('math', 'chat'):
        for p in range(6):
            for draw in range(2):
                n = int(rng.integers(5, 40))
                answers.append({'key': f'{d}{p}#{draw}', 'id': f'{d}{p}', 'domain': d, 'draw': draw,
                                'finish_reason': 'length' if n > 35 else 'stop', 'prompt_len': 3,
                                'prompt_token_ids': [1, 2, 3], 'answer_token_ids': list(range(n))})
                base.append(rng.gamma(2.0, 0.01, n))
    path = root / 'answers.jsonl'
    path.write_text(''.join(json.dumps(a) + '\n' for a in answers))
    digest = store.answers_digest(answers)
    runs = {}
    for m, scale in scales.items():
        run = root / m
        for i, b in enumerate(base):
            kl = b * scale * rng.uniform(0.9, 1.1, b.shape)
            store.save_seq(run, i, {'kl_20': kl * 0.9, 'kl_128': kl, 'flip': kl > 0.03,
                                    'tgt': np.zeros(len(b), np.float32)})
        store.write_manifest(run, {'ks': [20, 128], 'answers_sha256': digest})
        runs[m] = run
    return path, answers, base, runs


class AnalyzeTest(unittest.TestCase):
    def test_kl_table_matches_direct_means(self):
        with tempfile.TemporaryDirectory() as t:
            path, answers, base, runs = make_runs(Path(t), {'a': 1.0, 'b': 3.0})
            meta = [{'id': a['id'], 'domain': a['domain'], 'n': len(a['answer_token_ids']),
                     'finish': a['finish_reason']} for a in answers]
            res = analyze(meta, runs, lens=(8, 16, 1000), boot=200, prompt_counts=(4,), subset_boot=100)
            self.assertEqual(res['ks'], [20, 128])
            self.assertEqual(res['lens'][-1], max(m['n'] for m in meta))  # full length is the longest answer
            for m, run in runs.items():
                seqs = [store.load_seq(run, i) for i in range(len(answers))]
                want_answer = np.mean([s['kl_128'].mean() for s in seqs])
                want_token = np.concatenate([s['kl_128'] for s in seqs]).mean()
                want_l8 = np.mean([s['kl_128'][:8].mean() for s in seqs])
                self.assertAlmostEqual(res['kl']['answer'][m][1][-1], want_answer, places=12)
                self.assertAlmostEqual(res['kl']['token'][m][1][-1], want_token, places=12)
                self.assertAlmostEqual(res['kl']['answer'][m][1][0], want_l8, places=12)
                lo, hi = res['kl_ci'][m]
                self.assertLess(lo, want_answer)
                self.assertGreater(hi, want_answer)
            pair = res['pairs'][f"answer|L{res['lens'][-1]}|K128"]['a|b']
            self.assertTrue(pair['sig'])
            self.assertAlmostEqual(pair['rel_pct'], 100 * (res['kl']['answer']['a'][1][-1]
                                                           / res['kl']['answer']['b'][1][-1] - 1), places=9)
            self.assertIn(4, res['threshold_vs_prompts'])
            self.assertIn('## KL', report(res))

    def test_accuracy_agreement_with_reference_anchor(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            path, answers, base, runs = make_runs(root, {'a': 1.0, 'b': 3.0})
            meta = [{'id': a['id'], 'domain': a['domain'], 'n': len(a['answer_token_ids']),
                     'finish': a['finish_reason']} for a in answers]
            acc = {}
            for m, p in (('ref', 0.9), ('a', 0.85), ('b', 0.5)):
                f = root / f'{m}.jsonl'
                f.write_text(''.join(json.dumps({'question': q, 'score': float(s < p)}) + '\n'
                                     for q in range(40) for s in np.linspace(0, 1, 8, endpoint=False)))
                acc[m] = f
            res = analyze(meta, runs, lens=(1000,), boot=200, prompt_counts=(), accuracy=acc)
            a = res['accuracy']
            self.assertEqual(a['questions'], 40)
            self.assertEqual(a['pairs']['ref|b']['dKL'], -res['kl'][res['main']]['b'][1][-1])
            self.assertEqual(res['main'], 'token')
            self.assertTrue(a['pairs']['ref|b']['concordant'])
            self.assertAlmostEqual(a['tau_by_len'][res['lens'][-1]]['tau'], 1.0)

    def test_prompt_weights_keep_domain_sizes_and_pair_answers(self):
        answers = [{'id': f'{d}{p}', 'domain': d} for d in 'xy' for p in range(5) for _ in range(2)]
        W = prompt_weights(answers, 50, np.random.default_rng(0))
        self.assertTrue(np.all(W[:, :10].sum(1) == 10) and np.all(W[:, 10:].sum(1) == 10))
        self.assertTrue(np.array_equal(W[:, 0::2], W[:, 1::2]))  # both answers of a prompt move together

    def test_kendall_tau_b(self):
        self.assertAlmostEqual(kendall_tau_b([1, 2, 3], [10, 20, 30])[0], 1.0)
        self.assertAlmostEqual(kendall_tau_b([1, 2, 3], [30, 20, 10])[0], -1.0)


class RenderTest(unittest.TestCase):
    def test_tool_arguments_are_parsed_and_tools_passed(self):
        msgs = [{'role': 'assistant', 'content': '', 'tool_calls': [
            {'function': {'name': 'bash', 'arguments': '{"cmd": "ls"}'}}]}]
        self.assertEqual(normalize(msgs)[0]['tool_calls'][0]['function']['arguments'], {'cmd': 'ls'})
        self.assertEqual(msgs[0]['tool_calls'][0]['function']['arguments'], '{"cmd": "ls"}')  # input unchanged

        class Tok:
            def apply_chat_template(self, messages, add_generation_prompt, tokenize, **kw):
                self.kw = kw
                return {'input_ids': [len(messages), 7]}
        tok = Tok()
        rows = render_rows(tok, [{'id': 'i', 'domain': 'd', 'messages': msgs, 'tools': [{'x': 1}]}],
                           {'enable_thinking': False})
        self.assertEqual(rows, [{'id': 'i', 'domain': 'd', 'n_prompt_tokens': 2, 'prompt_token_ids': [1, 7]}])
        self.assertEqual(tok.kw, {'enable_thinking': False, 'tools': [{'x': 1}]})


class BackendCheckTest(unittest.TestCase):
    def test_emulation_requests_must_bind_emulation(self):
        ok = {'moe': ['OCP_MXQuantizationEmulationTritonExperts'], 'linear_kernels': ['EmulationMxfp4LinearKernel']}
        check_backends(ok, {'moe_backend': 'emulation', 'linear_backend': 'emulation'})
        check_backends({'linear_kernels': []}, {'moe_backend': 'triton', 'linear_backend': 'emulation'})
        with self.assertRaises(SystemExit):
            check_backends({'linear_kernels': ['AiterMxfp4LinearKernel']},
                           {'moe_backend': 'triton', 'linear_backend': 'emulation'})
        with self.assertRaises(SystemExit):
            check_backends({'moe': ['AiterExperts']}, {'moe_backend': 'emulation', 'linear_backend': 'auto'})


if __name__ == '__main__':
    unittest.main()
