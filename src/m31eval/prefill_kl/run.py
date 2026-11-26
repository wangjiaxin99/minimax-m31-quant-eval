"""``m31-eval prefill-kl``: generate reference answers, capture the reference Top-K, score candidates, compare runs.

All model steps run an offline ``vllm.LLM`` on one GPU (set ``HIP_VISIBLE_DEVICES`` / ``VLLM_ROCM_USE_AITER_MOE`` in
the environment). Prefix caching is off, sequences are read one at a time, and torch inductor runs with
``deterministic=True`` so that a sequence's prefill does not depend on other requests or on autotuning.
``--moe-backend`` and ``--linear-backend`` choose the MoE and quantized dense GEMM kernels (``emulation`` for
MXFP4); every manifest records the kernels the engine actually bound and the ``VLLM_*`` environment, and a step
stops if a requested emulation backend was not bound.

  render  dataset.jsonl -> prompts.jsonl  (chat template -> prompt token ids; no GPU)
  gen     prompts.jsonl -> answers.jsonl  (the reference model answers each prompt --answers times, seed + d)
  ref     answers.jsonl -> reference/     (reference model prefill, Top-K per answer token)
  score   answers.jsonl + reference/ -> run/   (candidate prefill, KL for each K per answer token)
  compare run_a run_b                      (bitwise equality of two score runs)
  analyze answers.jsonl + runs -> report    (offline: KL tables, resolution, length / K / prompt studies)
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# numpy, torch and vLLM are imported inside the steps, so that registering this command keeps the main CLI
# standard-library only.

#: Qwen3.6 model-card sampling, thinking mode for general tasks.
CARD_SAMPLING = {'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'min_p': 0.0,
                 'presence_penalty': 1.5, 'repetition_penalty': 1.0}
VALIDATE_TOL = 1e-3


def open_engine(args):
    from vllm import LLM

    from m31eval.prefill_kl.hooks import EXTENSION
    kwargs = dict(model=args.model, tensor_parallel_size=1, dtype='auto', kv_cache_dtype='auto',
                  max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization,
                  enable_prefix_caching=False, seed=0, language_model_only=True, moe_backend=args.moe_backend,
                  linear_backend=args.linear_backend, worker_extension_cls=EXTENSION,
                  compilation_config={'inductor_compile_config': {'deterministic': True}})
    llm = LLM(**kwargs)
    llm.collective_rpc('prefill_kl_install')
    return llm, {k: v for k, v in kwargs.items() if k != 'worker_extension_cls'}


def engine_record(llm, kwargs):
    """vLLM version, engine arguments, bound kernels and kernel-relevant environment; checks emulation requests."""
    import vllm
    kernels = llm.collective_rpc('prefill_kl_kernels')[0]
    check_backends(kernels, kwargs)
    env = {k: v for k, v in sorted(os.environ.items())
           if k.startswith('VLLM_') or k in ('HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES')}
    return {'vllm': vllm.__version__, 'engine_args': kwargs, 'kernels': kernels, 'env': env}


def check_backends(kernels: dict, kwargs: dict) -> None:
    """Stop when an emulation backend was requested but another kernel was bound."""
    if kwargs['moe_backend'] == 'emulation':
        other = [k for k in kernels.get('moe', []) if 'Emulation' not in k]
        if other:
            raise SystemExit(f'--moe-backend emulation requested, but the MoE layers bound {other}')
    if kwargs['linear_backend'] == 'emulation':
        other = [k for k in kernels.get('linear_kernels', []) if not k.startswith('Emulation')]
        if other:
            raise SystemExit(f'--linear-backend emulation requested, but dense layers bound {other}')


def cmd_gen(args):
    from vllm import SamplingParams
    rows = [json.loads(line) for line in open(args.prompts)]
    llm, kwargs = open_engine(args)
    sampling = dict(CARD_SAMPLING, **json.loads(args.sampling or '{}'))
    prompts, params, meta = [], [], []
    for r in rows:
        for d in range(args.answers):
            prompts.append({'prompt_token_ids': r['prompt_token_ids']})
            params.append(SamplingParams(max_tokens=args.max_tokens, seed=args.seed + d, **sampling))
            meta.append((r, d))
    started = time.time()
    outs = llm.generate(prompts, params)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w') as f:
        for (r, d), o in zip(meta, outs):
            c = o.outputs[0]
            f.write(json.dumps({'key': f"{r['id']}#{d}", 'id': r['id'], 'domain': r['domain'], 'draw': d,
                                'seed': args.seed + d, 'prompt_len': len(r['prompt_token_ids']),
                                'finish_reason': c.finish_reason, 'prompt_token_ids': r['prompt_token_ids'],
                                'answer_token_ids': list(c.token_ids)}) + '\n')
    rec = {'step': 'gen', 'model': args.model, 'answers': args.answers, 'seed': args.seed,
           'max_tokens': args.max_tokens, 'sampling': sampling, 'seconds': time.time() - started,
           **engine_record(llm, kwargs)}
    Path(str(out_path) + '.manifest.json').write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ('step', 'seconds', 'kernels')}))


def _teacher_force(llm, mode, a, extra):
    """One sequence through prefill; returns the hook's arrays and the engine's own target logprobs."""
    import numpy as np
    from vllm import SamplingParams

    from m31eval.prefill_kl.hooks import TAG, unpack
    tokens = a['prompt_token_ids'] + a['answer_token_ids']
    start = a['prompt_len']
    llm.collective_rpc('prefill_kl_begin', args=(mode, {a['key']: (start, tokens)}), kwargs=extra)
    sp = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0, extra_args={TAG: {'key': a['key']}})
    out = llm.generate([{'prompt_token_ids': tokens}], sp, use_tqdm=False)[0]
    got = unpack(llm.collective_rpc('prefill_kl_end')[0][a['key']])
    if 'missing' in got:
        raise RuntimeError(f"{a['key']}: {int(got['missing'])} positions not seen by the hook")
    engine_tgt = np.array([out.prompt_logprobs[start + j][tok].logprob
                           for j, tok in enumerate(a['answer_token_ids'])], dtype=np.float64)
    return got, engine_tgt


def _loop(args, mode, answers, root, extra_for):
    import numpy as np

    from m31eval.prefill_kl import store
    llm, kwargs = open_engine(args)
    started, worst = time.time(), 0.0
    for i, a in enumerate(answers):
        if not a['answer_token_ids']:
            continue
        if store.seq_file(root, i).exists() and not args.overwrite:
            continue
        got, engine_tgt = _teacher_force(llm, mode, a, extra_for(i, a))
        diff = float(np.nanmax(np.abs(got['tgt'].astype(np.float64) - engine_tgt)))
        if not diff <= VALIDATE_TOL:
            raise RuntimeError(f"{a['key']}: hook target logprobs differ from the engine's by {diff}")
        worst = max(worst, diff)
        store.save_seq(root, i, got)
        if i % 20 == 0:
            print(f'{mode} {i + 1}/{len(answers)} {a["key"]} n={len(a["answer_token_ids"])} '
                  f'{time.time() - started:.0f}s', flush=True)
    return llm, kwargs, time.time() - started, worst


def cmd_ref(args):
    from m31eval.prefill_kl import store
    answers = store.read_answers(args.answers)
    root = Path(args.out)
    llm, kwargs, secs, worst = _loop(args, 'ref', answers, root, lambda i, a: {'k': args.topk})
    store.write_manifest(root, {'step': 'ref', 'model': args.model, 'topk': args.topk, 'answers': str(args.answers),
                                'answers_sha256': store.answers_digest(answers), 'seconds': secs,
                                'validate_max_abs_diff': worst, **engine_record(llm, kwargs)})
    print('ref done', secs, worst)


def cmd_score(args):
    from m31eval.prefill_kl import store
    answers = store.read_answers(args.answers)
    ref = Path(args.reference)
    man = store.read_manifest(ref)
    if man['answers_sha256'] != store.answers_digest(answers):
        raise SystemExit('reference was built on different answers')
    ks = tuple(int(k) for k in args.ks.split(','))
    root = Path(args.out)
    llm, kwargs, secs, worst = _loop(args, 'score', answers, root,
                                     lambda i, a: {'ks': ks, 'ref_files': {a['key']: str(store.seq_file(ref, i))}})
    summary = summarize(answers, root, ks)
    store.write_manifest(root, {'step': 'score', 'model': args.model, 'reference': str(ref), 'ks': ks,
                                'answers_sha256': man['answers_sha256'], 'seconds': secs,
                                'validate_max_abs_diff': worst, 'summary': summary, **engine_record(llm, kwargs)})
    print(json.dumps(summary, indent=1))


def summarize(answers, root, ks):
    import numpy as np

    from m31eval.prefill_kl import store
    per = {k: [] for k in ks}
    tok = {k: [0.0, 0] for k in ks}
    flips = [0, 0]
    for i, a in enumerate(answers):
        if not store.seq_file(root, i).exists():
            continue
        s = store.load_seq(root, i)
        for k in ks:
            v = s[f'kl_{k}']
            per[k].append(float(v.mean()))
            tok[k][0] += float(v.sum())
            tok[k][1] += v.size
        flips[0] += int(s['flip'].sum())
        flips[1] += s['flip'].size
    return {'answers': len(per[ks[0]]), 'positions': flips[1],
            'kl_answer_mean': {k: float(np.mean(per[k])) for k in ks},
            'kl_token_mean': {k: tok[k][0] / tok[k][1] for k in ks},
            'top1_flip_rate': flips[0] / flips[1] if flips[1] else None}


def cmd_compare(args):
    import numpy as np
    a, b = Path(args.run_a), Path(args.run_b)
    files = sorted(p.name for p in (a / 'seqs').glob('*.npz'))
    same = pos_same = pos_total = 0
    diffs = []
    for name in files:
        za, zb = np.load(a / 'seqs' / name), np.load(b / 'seqs' / name)
        eq = all(np.array_equal(za[k], zb[k], equal_nan=True) for k in za.files)
        same += eq
        rows = np.ones(za['tgt'].shape[0], dtype=bool)
        for k in za.files:
            x, y = za[k], zb[k]
            eqk = (x == y) | (np.isnan(x) & np.isnan(y)) if x.dtype.kind == 'f' else x == y
            rows &= eqk.reshape(eqk.shape[0], -1).all(axis=1)
        pos_same += int(rows.sum())
        pos_total += rows.size
        if not eq:
            diffs.append(name)
    res = {'sequences': len(files), 'identical_sequences': same, 'positions': pos_total,
           'identical_positions': pos_same, 'differing_files': diffs[:50]}
    print(json.dumps(res, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


def cmd_render(args):
    from m31eval.prefill_kl.render import cmd_render as run
    run(args)


def cmd_analyze(args):
    from m31eval.prefill_kl.analyze import cmd_analyze as run
    run(args)


def add_parser(commands):
    p = commands.add_parser('prefill-kl', help='Long teacher-forced KL scored inside vLLM (offline engine)')
    sub = p.add_subparsers(dest='action', required=True)

    def engine_opts(q):
        q.add_argument('--model', required=True)
        q.add_argument('--moe-backend', default='triton', help='triton for BF16, emulation for MXFP4')
        q.add_argument('--linear-backend', default='auto',
                       help='quantized dense GEMM backend (vLLM --linear-backend); emulation for MXFP4 dense layers')
        q.add_argument('--max-model-len', type=int, default=65536)
        q.add_argument('--gpu-memory-utilization', type=float, default=0.80)

    rd = sub.add_parser('render')
    rd.add_argument('--src', required=True, help='JSONL rows {id, domain, messages, tools?}')
    rd.add_argument('--model', required=True, help='model directory with the tokenizer and chat template')
    rd.add_argument('--chat-template-kwargs', help='JSON passed to apply_chat_template, e.g. {"enable_thinking": false}')
    rd.add_argument('--out', required=True)
    rd.set_defaults(func=cmd_render)
    g = sub.add_parser('gen')
    engine_opts(g)
    g.add_argument('--prompts', required=True, help='JSONL rows {id, domain, prompt_token_ids}')
    g.add_argument('--answers', type=int, default=2)
    g.add_argument('--seed', type=int, default=17)
    g.add_argument('--max-tokens', type=int, default=8192)
    g.add_argument('--sampling', help='JSON overriding the model-card sampling fields')
    g.add_argument('--out', required=True)
    g.set_defaults(func=cmd_gen)
    r = sub.add_parser('ref')
    engine_opts(r)
    r.add_argument('--answers', required=True)
    r.add_argument('--topk', type=int, default=2000)
    r.add_argument('--out', required=True)
    r.add_argument('--overwrite', action='store_true')
    r.set_defaults(func=cmd_ref)
    s = sub.add_parser('score')
    engine_opts(s)
    s.add_argument('--answers', required=True)
    s.add_argument('--reference', required=True)
    s.add_argument('--ks', default='20,128,2000')
    s.add_argument('--out', required=True)
    s.add_argument('--overwrite', action='store_true')
    s.set_defaults(func=cmd_score)
    c = sub.add_parser('compare')
    c.add_argument('run_a')
    c.add_argument('run_b')
    c.add_argument('--out')
    c.set_defaults(func=cmd_compare)
    an = sub.add_parser('analyze')
    an.add_argument('--answers', required=True)
    an.add_argument('--run', action='append', required=True, help='NAME=score_dir, once per model')
    an.add_argument('--accuracy', action='append', help='NAME=jsonl of {question, score} rows; a NAME without '
                                                       'a run is the reference model (KL 0)')
    an.add_argument('--ks', help='K values to analyse (default: those every run has)')
    an.add_argument('--lens', default='32,64,128,256,512,1024,2048,4096,8192')
    an.add_argument('--prompt-counts', default='48,96,150')
    an.add_argument('--boot', type=int, default=4000)
    an.add_argument('--out', required=True, help='output prefix; writes .json and .md')
    an.set_defaults(func=cmd_analyze)
    return p


if __name__ == '__main__':
    import sys
    ap = argparse.ArgumentParser(prog='python -m m31eval.prefill_kl.run')
    add_parser(ap.add_subparsers(dest='command', required=True))
    a = ap.parse_args(['prefill-kl'] + sys.argv[1:])
    a.func(a)
