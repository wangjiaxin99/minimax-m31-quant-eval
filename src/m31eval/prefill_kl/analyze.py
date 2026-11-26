"""Offline analysis of ``score`` runs: KL tables, resolution, length / K / prompt-count studies, accuracy agreement.

Reads only ``answers.jsonl`` and the runs' ``seqs/*.npz`` (row j = answer token j), so any prefix length can be
studied without running a model again. NumPy only.

Aggregations of the per-position KL:
  answer  mean over an answer's first L positions, then mean over answers (each answer weighs the same; main metric)
  token   all positions pooled (long answers dominate)
  domain  answer mean within each domain, then mean over domains

Uncertainty: paired bootstrap over prompts. Prompts are resampled with replacement within each domain and the
answers of one prompt move together, so every model is scored on the same resample. For two models the 95%
interval of ln KL_a - ln KL_b gives the detection threshold: 1.96 x its bootstrap standard error, as a relative
difference. It answers "does the difference hold on another sample of similar prompts"; it does not include
calibration noise (rebuilding a recipe on other calibration data).

Accuracy (optional): per-sample JSONL rows ``{"question": ..., "score": 0..1}`` per model. Questions are
bootstrapped independently of the KL prompts. A model with accuracy but no run is the reference (KL 0).
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from m31eval.prefill_kl import store

LENS = (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
BUCKETS = (0, 32, 128, 512, 2048, 8192)
MODES = ('answer', 'token', 'domain')


# ----------------------------------------------------------------------------- loading
def load_answers(path) -> list[dict]:
    """Answer metadata only (the token ids are not kept)."""
    out = []
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    for r in rows:
        out.append({'id': r['id'], 'domain': r['domain'], 'n': len(r['answer_token_ids']),
                    'finish': r.get('finish_reason')})
    return out


def load_run(root, answers: list[dict], ks, lens) -> dict:
    """Per-answer KL sums over the first L positions for each K, position counts, bucket sums and top-1 flips."""
    n_ans, kmax = len(answers), max(ks)
    S = np.zeros((len(ks), len(lens), n_ans))
    C = np.zeros((len(lens), n_ans))
    B = np.zeros((len(BUCKETS) - 1, n_ans))
    BC = np.zeros((len(BUCKETS) - 1, n_ans))
    flips, total = np.zeros(n_ans), np.zeros(n_ans)
    for i, a in enumerate(answers):
        if a['n'] == 0:
            continue
        path = store.seq_file(root, i)
        if not path.exists():
            raise SystemExit(f'{root}: answer {i} ({a["id"]}) has no scores; finish the run first')
        with np.load(path) as z:
            n = z['tgt'].shape[0]
            for ki, k in enumerate(ks):
                cs = np.cumsum(z[f'kl_{k}'])
                for li, L in enumerate(lens):
                    S[ki, li, i] = cs[min(L, n) - 1]
            kl = z[f'kl_{kmax}']
            for bi in range(len(BUCKETS) - 1):
                lo, hi = BUCKETS[bi], min(BUCKETS[bi + 1], n)
                if hi > lo:
                    B[bi, i], BC[bi, i] = kl[lo:hi].sum(), hi - lo
            flips[i], total[i] = z['flip'].sum(), n
        for li, L in enumerate(lens):
            C[li, i] = min(L, n)
    return {'S': S, 'C': C, 'B': B, 'BC': BC, 'flips': flips, 'total': total}


def load_accuracy(path) -> dict[str, list[float]]:
    """question -> list of per-sample scores."""
    per_q: dict[str, list[float]] = {}
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    for r in rows:
        per_q.setdefault(str(r['question']), []).append(float(r['score']))
    return per_q


# ----------------------------------------------------------------------------- bootstrap machinery
def prompt_weights(answers: list[dict], boot: int, rng, frac: float = 1.0) -> np.ndarray:
    """(boot, n_answers) weights: prompts drawn with replacement within each domain, round(frac x n) per domain."""
    pid: dict[str, int] = {}
    for a in answers:
        pid.setdefault(a['id'], len(pid))
    p_of = np.array([pid[a['id']] for a in answers])
    dom_of_prompt = {pid[a['id']]: a['domain'] for a in answers}
    counts = np.zeros((boot, len(pid)))
    for d in sorted(set(dom_of_prompt.values())):
        idx = np.array([p for p, dd in dom_of_prompt.items() if dd == d])
        size = max(1, int(round(len(idx) * frac)))
        draws = rng.integers(0, len(idx), size=(boot, size))
        for b in range(boot):
            counts[b, idx] = np.bincount(draws[b], minlength=len(idx))
    return counts[:, p_of]


def aggregate(W: np.ndarray, data: dict, answers: list[dict], mode: str) -> np.ndarray:
    """KL (rows of W, nK, nL) under ``mode`` from per-answer sums ``data['S']`` and counts ``data['C']``."""
    S, C = data['S'], data['C']
    nk, nl = S.shape[:2]
    valid = (C[-1] > 0).astype(float)
    W = W * valid[None]
    with np.errstate(invalid='ignore', divide='ignore'):
        A = np.nan_to_num(S / C[None]).reshape(nk * nl, -1)
    if mode == 'answer':
        return ((W @ A.T) / W.sum(1, keepdims=True)).reshape(-1, nk, nl)
    if mode == 'token':
        num = W @ S.reshape(nk * nl, -1).T
        return num.reshape(-1, nk, nl) / (W @ C.T)[:, None, :]
    if mode == 'domain':
        out = []
        for d in sorted({a['domain'] for a in answers}):
            Wd = W * np.array([a['domain'] == d for a in answers], dtype=float)[None]
            out.append((Wd @ A.T) / Wd.sum(1, keepdims=True))
        return np.mean(out, axis=0).reshape(-1, nk, nl)
    raise ValueError(mode)


def ci(x, q=(2.5, 97.5)) -> list[float]:
    return [float(v) for v in np.percentile(x, q)]


def kendall_tau_b(x, y) -> np.ndarray:
    """tau_b between the rows of x and y (each (boot, M) or a vector of M); ties handled."""
    x, y = np.atleast_2d(x), np.atleast_2d(y)
    iu = np.triu_indices(x.shape[1], 1)
    sx = np.sign(x[:, iu[0]] - x[:, iu[1]])
    sy = np.sign(y[:, iu[0]] - y[:, iu[1]])
    den = np.sqrt((sx ** 2).sum(1) * (sy ** 2).sum(1))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, (sx * sy).sum(1) / den, np.nan)


def pair_stats(point_a: float, point_b: float, boot_a: np.ndarray, boot_b: np.ndarray) -> dict:
    d = np.log(boot_a) - np.log(boot_b)
    p0 = float(np.log(point_a) - np.log(point_b))
    lo, hi = ci(d)
    return {'dln': p0, 'rel_pct': float(100 * (np.exp(p0) - 1)), 'se': float(d.std()), 'ci': [lo, hi],
            'sig': bool(lo > 0 or hi < 0), 'repro': float(max((d > 0).mean(), (d < 0).mean()))}


def threshold(ses) -> dict:
    """Detection threshold (relative %) from the pairs' bootstrap standard errors of ln KL ratio."""
    se = np.asarray(ses)
    return {'median': float(100 * (np.exp(1.96 * np.median(se)) - 1)),
            'min': float(100 * (np.exp(1.96 * se.min()) - 1)), 'max': float(100 * (np.exp(1.96 * se.max()) - 1))}


# ----------------------------------------------------------------------------- analysis
def analyze(answers: list[dict], runs: dict[str, Path], ks=None, lens=LENS, boot: int = 4000,
            prompt_counts=(48, 96, 150), subset_boot: int = 2000, accuracy: dict[str, Path] | None = None) -> dict:
    names = list(runs)
    if ks is None:
        ks = sorted(set.intersection(*(set(store.read_manifest(p)['ks']) for p in runs.values())))
    ks, kmax = sorted(ks), max(ks)
    longest = max(a['n'] for a in answers)
    lens = sorted({L for L in lens if L < longest} | {longest})  # the last length is always the full answer
    ki, full = ks.index(kmax), len(lens) - 1
    data = {m: load_run(p, answers, ks, lens) for m, p in runs.items()}
    W = prompt_weights(answers, boot, np.random.default_rng(0))
    W1 = np.ones((1, len(answers)))
    point = {mode: {m: aggregate(W1, data[m], answers, mode)[0] for m in names} for mode in MODES}
    boots = {mode: {m: aggregate(W, data[m], answers, mode) for m in names} for mode in MODES}
    res = {'answers': len(answers), 'prompts': len({a['id'] for a in answers}),
           'positions': int(sum(a['n'] for a in answers)), 'models': names, 'ks': ks, 'lens': lens, 'boot': boot,
           'runs': {m: str(p) for m, p in runs.items()}}

    # KL table
    res['kl'] = {mode: {m: point[mode][m].tolist() for m in names} for mode in MODES}
    res['kl_ci'] = {m: ci(boots['answer'][m][:, ki, full]) for m in names}
    res['top1_flip_rate'] = {m: float(data[m]['flips'].sum() / data[m]['total'].sum()) for m in names}

    # pairs and thresholds by length / K / aggregation
    positive = [m for m in names if point['answer'][m][ki, full] > 0]
    res['pairs'], res['threshold'] = {}, {}
    for mode in MODES:
        for li, L in enumerate(lens):
            for kj, k in enumerate(ks):
                if mode != 'answer' and (k != kmax or li != full):
                    continue
                ps = {f'{a}|{b}': pair_stats(point[mode][a][kj, li], point[mode][b][kj, li],
                                             boots[mode][a][:, kj, li], boots[mode][b][:, kj, li])
                      for a, b in itertools.combinations(positive, 2)}
                if not ps:
                    continue
                key = f'{mode}|L{L}|K{k}'
                res['pairs'][key] = ps
                res['threshold'][key] = {**threshold([v['se'] for v in ps.values()]),
                                         'n_sig': int(sum(v['sig'] for v in ps.values())), 'n_pairs': len(ps)}

    # threshold vs number of prompts (stratified subsamples, with replacement)
    n_prompts = res['prompts']
    res['threshold_vs_prompts'] = {}
    if len(positive) >= 2:
        for n in [c for c in prompt_counts if c < n_prompts] + [n_prompts]:
            Wf = prompt_weights(answers, subset_boot, np.random.default_rng(7), frac=n / n_prompts)
            bf = {m: aggregate(Wf, data[m], answers, 'answer') for m in positive}
            res['threshold_vs_prompts'][n] = {
                L: threshold([np.std(np.log(bf[a][:, ki, li]) - np.log(bf[b][:, ki, li]))
                              for a, b in itertools.combinations(positive, 2)])['median']
                for li, L in enumerate(lens)}

    # per domain, position profile, share of KL inside the first L positions
    doms = sorted({a['domain'] for a in answers})
    res['domain_info'] = {d: {'answers': int(sum(a['domain'] == d for a in answers)),
                              'median_len': int(np.median([a['n'] for a in answers if a['domain'] == d])),
                              'at_length_limit': int(sum(a['finish'] == 'length' for a in answers
                                                         if a['domain'] == d))} for d in doms}
    res['domain_kl'] = {}
    for m in names:
        S, C = data[m]['S'], data[m]['C']
        res['domain_kl'][m] = {}
        for d in doms:
            idx = np.array([a['domain'] == d and a['n'] > 0 for a in answers])
            res['domain_kl'][m][d] = float((S[ki, full, idx] / C[full, idx]).mean())
    res['buckets'] = list(BUCKETS)
    res['profile'] = {}
    for m in names:
        d = data[m]
        tot = d['S'][ki, full].sum()
        with np.errstate(invalid='ignore', divide='ignore'):
            res['profile'][m] = {
                'bucket_mean_kl': (d['B'].sum(1) / d['BC'].sum(1)).tolist(),
                'bucket_positions': d['BC'].sum(1).tolist(),
                'share_of_kl_first_L': [float(d['S'][ki, li].sum() / tot) if tot > 0 else None
                                        for li in range(len(lens))],
                'share_of_positions_first_L': [float(d['C'][li].sum() / d['C'][full].sum())
                                               for li in range(len(lens))]}

    # ranking stability over length: Kendall tau against the full-length order
    if len(positive) >= 2:
        ref_order = np.array([point['answer'][m][ki, full] for m in positive])
        res['rank_tau_vs_full'] = {L: float(kendall_tau_b(np.array([point['answer'][m][ki, li] for m in positive]),
                                                          ref_order)[0]) for li, L in enumerate(lens)}

    if accuracy:
        res['accuracy'] = agreement(accuracy, point, boots, names, ki, lens, boot)
    return res


def agreement(accuracy: dict[str, Path], point, boots, names, ki, lens, boot) -> dict:
    """Accuracy per model with question-bootstrap intervals, accuracy pairs, and Kendall tau(-KL, accuracy)."""
    acc = {m: load_accuracy(p) for m, p in accuracy.items()}
    models = list(acc)  # a model with accuracy but no run is the reference (KL 0)
    qs = sorted(set.intersection(*(set(acc[m]) for m in models)))
    A = {m: np.array([np.mean(acc[m][q]) for q in qs]) for m in models}
    rng = np.random.default_rng(1)
    Wq = np.stack([np.bincount(rng.integers(0, len(qs), len(qs)), minlength=len(qs)) for _ in range(boot)])
    acc_b = {m: (Wq @ A[m]) / len(qs) for m in models}
    out = {'questions': len(qs), 'samples': {m: sum(len(v) for v in acc[m].values()) for m in models},
           'models': models, 'accuracy': {m: float(100 * A[m].mean()) for m in models},
           'accuracy_ci': {m: [100 * v for v in ci(acc_b[m])] for m in models}, 'pairs': {}}

    def kl_of(m, li, boot_rows=False):
        if m not in names:
            return np.zeros(boot) if boot_rows else 0.0
        return boots['answer'][m][:, ki, li] if boot_rows else point['answer'][m][ki, li]

    full = len(lens) - 1
    for a, b in itertools.combinations(models, 2):
        d = acc_b[a] - acc_b[b]
        lo, hi = ci(d)
        dacc = float(100 * (A[a].mean() - A[b].mean()))
        dkl = kl_of(a, full) - kl_of(b, full)
        out['pairs'][f'{a}|{b}'] = {'diff_pts': dacc, 'ci_pts': [100 * lo, 100 * hi], 'acc_sig': bool(lo > 0 or hi < 0),
                                    'dKL': float(dkl), 'concordant': bool(dkl * dacc < 0) if dkl and dacc else None}
    if len(models) >= 3:
        accp = np.array([A[m].mean() for m in models])
        out['tau_by_len'] = {}
        for li, L in enumerate(lens):
            klp = -np.array([kl_of(m, li) for m in models])
            tb = kendall_tau_b(-np.stack([kl_of(m, li, True) for m in models], 1),
                               np.stack([acc_b[m] for m in models], 1))
            out['tau_by_len'][L] = {'tau': float(kendall_tau_b(klp, accp)[0]), 'ci': ci(tb[~np.isnan(tb)])}
        if len(models) <= 8:  # exact permutation test, one-sided
            perms = np.array(list(itertools.permutations(range(len(models)))))
            klp = -np.array([kl_of(m, full) for m in models])
            t_obs = float(kendall_tau_b(klp, accp)[0])
            t_all = kendall_tau_b(np.repeat(klp[None], len(perms), 0), accp[perms])
            out['perm_p_full'] = float((t_all >= t_obs - 1e-12).mean())
    return out


# ----------------------------------------------------------------------------- report
def _pct(x) -> str:
    return f'{x:+.1f}%'


def report(res: dict) -> str:
    """Markdown summary of ``analyze``'s result."""
    ks, lens, names = res['ks'], res['lens'], res['models']
    ki, full, kmax = len(ks) - 1, len(lens) - 1, ks[-1]
    L = lens[full]
    base = next((m for m in names if res['kl']['answer'][m][ki][full] > 0), None)
    out = [f"# prefill-kl analysis\n\n{res['answers']} answers of {res['prompts']} prompts, {res['positions']:,} "
           f"positions; bootstrap {res['boot']} over prompts, stratified by domain.\n",
           f'## KL (first {L} positions, K = {kmax}, answer mean)\n',
           '| Model | KL | 95% interval | KL (token mean) | Top-1 flip | vs ' + str(base) + ' |', '|---|---|---|---|---|---|']
    for m in names:
        v = res['kl']['answer'][m][ki][full]
        lo, hi = res['kl_ci'][m]
        rel = _pct(100 * (v / res['kl']['answer'][base][ki][full] - 1)) if base and v > 0 and m != base else '-'
        out.append(f"| {m} | {v:.5f} | [{lo:.5f}, {hi:.5f}] | {res['kl']['token'][m][ki][full]:.5f} | "
                   f"{100 * res['top1_flip_rate'][m]:.2f}% | {rel} |")
    key = f'answer|L{L}|K{kmax}'
    if key in res['pairs']:
        out += ['\n## Pairs (relative KL difference, 95% interval of ln ratio)\n',
                '| A / B | A vs B | ln ratio 95% | significant | order reproduced |', '|---|---|---|---|---|']
        for p, v in res['pairs'][key].items():
            out.append(f"| {p.replace('|', ' / ')} | {_pct(v['rel_pct'])} | [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}] | "
                       f"{'yes' if v['sig'] else 'no'} | {100 * v['repro']:.1f}% |")
        out += ['\n## Detection threshold by length (answer mean)\n',
                '| L | ' + ' | '.join(f'K={k}' for k in ks) + ' | pairs separated (K=' + str(kmax) + ') |',
                '|---' * (len(ks) + 2) + '|']
        for L2 in lens:
            cells = [f"{res['threshold'][f'answer|L{L2}|K{k}']['median']:.2f}%" for k in ks]
            t = res['threshold'][f'answer|L{L2}|K{kmax}']
            out.append(f"| {L2} | " + ' | '.join(cells) + f" | {t['n_sig']}/{t['n_pairs']} |")
        out += ['\n## Detection threshold by aggregation (full length)\n', '| Aggregation | threshold | separated |',
                '|---|---|---|']
        for mode in MODES:
            t = res['threshold'][f'{mode}|L{L}|K{kmax}']
            out.append(f"| {mode} | {t['median']:.2f}% | {t['n_sig']}/{t['n_pairs']} |")
    if res.get('threshold_vs_prompts'):
        out += ['\n## Detection threshold by number of prompts (answer mean, K=' + str(kmax) + ')\n',
                '| prompts | ' + ' | '.join(f'L={x}' for x in lens) + ' |', '|---' * (len(lens) + 1) + '|']
        for n, row in res['threshold_vs_prompts'].items():
            out.append(f'| {n} | ' + ' | '.join(f'{row[x]:.1f}%' for x in lens) + ' |')
    doms = list(res['domain_info'])
    out += ['\n## Domains (full length, K=' + str(kmax) + ')\n',
            '| Domain | answers | median length | at limit | ' + ' | '.join(names) + ' |', '|---' * (len(names) + 4) + '|']
    for d in doms:
        i = res['domain_info'][d]
        out.append(f"| {d} | {i['answers']} | {i['median_len']} | {i['at_length_limit']} | "
                   + ' | '.join(f"{res['domain_kl'][m][d]:.4f}" for m in names) + ' |')
    if res.get('rank_tau_vs_full'):
        out += ['\n## Order against the full-length order (Kendall tau)\n',
                '| ' + ' | '.join(f'L={x}' for x in lens) + ' |', '|---' * len(lens) + '|',
                '| ' + ' | '.join(f'{v:+.2f}' for v in res['rank_tau_vs_full'].values()) + ' |']
    acc = res.get('accuracy')
    if acc:
        out += [f"\n## Accuracy ({acc['questions']} questions)\n", '| Model | accuracy | 95% interval | samples |',
                '|---|---|---|---|']
        for m in acc['models']:
            lo, hi = acc['accuracy_ci'][m]
            out.append(f"| {m} | {acc['accuracy'][m]:.2f} | [{lo:.2f}, {hi:.2f}] | {acc['samples'][m]} |")
        out += ['\n| A / B | accuracy A - B | 95% interval | significant | KL A - B | concordant |',
                '|---|---|---|---|---|---|']
        for p, v in acc['pairs'].items():
            out.append(f"| {p.replace('|', ' / ')} | {v['diff_pts']:+.2f} | [{v['ci_pts'][0]:+.2f}, {v['ci_pts'][1]:+.2f}] | "
                       f"{'yes' if v['acc_sig'] else 'no'} | {v['dKL']:+.5f} | {v['concordant']} |")
        if 'tau_by_len' in acc:
            t = acc['tau_by_len'][lens[full]]
            out.append(f"\nKendall tau(-KL, accuracy) at full length: {t['tau']:+.2f}, bootstrap 95% "
                       f"[{t['ci'][0]:+.2f}, {t['ci'][1]:+.2f}]"
                       + (f", exact permutation p = {acc['perm_p_full']:.4f}" if 'perm_p_full' in acc else '') + '.')
    return '\n'.join(out) + '\n'


def _named(items) -> dict[str, Path]:
    out = {}
    for it in items or []:
        name, _, path = it.partition('=')
        if not path:
            raise SystemExit(f'{it!r}: use NAME=PATH')
        out[name] = Path(path)
    return out


def cmd_analyze(args) -> None:
    answers = load_answers(args.answers)
    runs = _named(args.run)
    digest = store.answers_digest(store.read_answers(args.answers))
    for m, p in runs.items():
        if store.read_manifest(p).get('answers_sha256') != digest:
            raise SystemExit(f'{m}: {p} was scored on different answers')
    res = analyze(answers, runs, ks=[int(k) for k in args.ks.split(',')] if args.ks else None,
                  lens=[int(x) for x in args.lens.split(',')], boot=args.boot,
                  prompt_counts=[int(x) for x in args.prompt_counts.split(',') if x],
                  accuracy=_named(args.accuracy))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(out) + '.json').write_text(json.dumps(res, indent=1, default=float))
    Path(str(out) + '.md').write_text(report(res))
    print(f'wrote {out}.json and {out}.md')
