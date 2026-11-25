"""vLLM worker extension: reduce teacher-forced logits on the GPU.

Adapted from QDA's ``qda_kl/vllm_hooks.py`` (https://github.com/humanfia/qda, commit dc39a02, Apache-2.0),
restricted to vLLM's V2 GPU model runner (the only GPU runner in vLLM 0.34) and to one tensor-parallel rank.

A tagged request (``SamplingParams.extra_args["prefill_kl"] = {"key": ...}``) carries prompt + reference answer as
its prompt; prompt log-probabilities make the runner call ``prompt_logprobs_worker.compute_prompt_logprobs`` with
the step's hidden states. The wrapper finds the tagged request's rows and calls the model's ``compute_logits`` on
the engine's own prompt-logprob blocks (``CHUNK_SIZE`` rows counted from row 0 of the step, the last block cut at
the step's row count), so a quantized lm-head sees the same shapes as the engine's own call. Then:

* ``ref`` mode: per answer position the Top-K log-probabilities, their ids, the log tail mass, the target logprob;
* ``score`` mode: per answer position KL for each K in ``ks`` against the stored reference, top-1 flip, target logprob.

Position p of the sequence predicts token p + 1; answer token j sits at index start + j, predicted by row
start + j - 1, so the outputs have one row per answer token.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from m31eval.prefill_kl.topk import TopK, kl_multi, reference_topk

EXTENSION = 'm31eval.prefill_kl.hooks.KLWorker'
TAG = 'prefill_kl'


def engine_chunk() -> int:
    from vllm.v1.worker.gpu.sample.prompt_logprob import CHUNK_SIZE
    return CHUNK_SIZE


@dataclass
class _Track:
    start: int
    tokens: torch.Tensor  # [n_tok] int64 on the card
    anchor: TopK | None = None
    out: dict[str, torch.Tensor] = field(default_factory=dict)
    seen: torch.Tensor | None = None


@dataclass
class _State:
    mode: str = ''
    k: int = 0
    ks: tuple[int, ...] = ()
    tracks: dict[str, _Track] = field(default_factory=dict)
    tags: dict[str, dict] = field(default_factory=dict)
    installed: bool = False


class KLWorker:
    """Methods called through ``LLM.collective_rpc``; vLLM sets ``self.model_runner``."""

    model_runner: Any

    def _st(self) -> _State:
        st = getattr(self, '_prefill_kl_state', None)
        if st is None:
            st = self._prefill_kl_state = _State()
        return st

    def prefill_kl_install(self) -> str:
        st, runner = self._st(), self.model_runner
        if st.installed:
            return 'v2'
        owner = getattr(runner, 'prompt_logprobs_worker', None)
        if owner is None:
            raise RuntimeError('this vLLM runner has no prompt_logprobs_worker (V2 runner required)')
        add, remove, original = owner.add_request, owner.remove_request, owner.compute_prompt_logprobs

        def add_request(req_id, req_idx, sampling_params):
            tag = (sampling_params.extra_args or {}).get(TAG)
            if tag is not None:
                st.tags[req_id] = tag
            return add(req_id, req_idx, sampling_params)

        def remove_request(req_id):
            st.tags.pop(req_id, None)
            return remove(req_id)

        def entry(*args, **kwargs):
            if st.tracks:
                hidden = kwargs.get('hidden_states', args[1] if len(args) > 1 else None)
                batch = kwargs.get('input_batch', args[2] if len(args) > 2 else None)
                q, done, n_rows = batch.query_start_loc_np, batch.num_computed_prefill_tokens_np, int(batch.num_tokens)
                for i, req_id in enumerate(batch.req_ids):
                    tag = st.tags.get(req_id)
                    if tag is not None and tag['key'] in st.tracks:
                        self._take(st, st.tracks[tag['key']], hidden, int(q[i]), int(q[i + 1]), int(done[i]), n_rows)
            return original(*args, **kwargs)

        owner.add_request, owner.remove_request, owner.compute_prompt_logprobs = add_request, remove_request, entry
        st.installed = True
        return 'v2'

    def prefill_kl_kernels(self) -> dict[str, list[str]]:
        got: dict[str, set] = {}
        for _, m in self.model_runner.model.named_modules():
            qm = getattr(m, 'quant_method', None)
            if qm is None:
                continue
            kind = 'moe' if (hasattr(m, 'w13_weight') or 'MoE' in type(m).__name__) else 'dense'
            mk = getattr(qm, 'moe_kernel', None)
            experts = getattr(mk, 'fused_experts', None) or getattr(qm, 'fused_experts', None)
            got.setdefault(kind, set()).add(type(experts or qm).__name__)
        return {k: sorted(v) for k, v in got.items()}

    def prefill_kl_begin(self, mode: str, seqs: dict[str, tuple[int, list[int]]], k: int = 0,
                         ks: tuple[int, ...] = (), ref_files: dict[str, str] | None = None) -> bool:
        """Track ``seqs`` ({key: (start, tokens)}). ``ref`` keeps Top-``k``; ``score`` reads ``ref_files[key]``."""
        if mode not in ('ref', 'score'):
            raise ValueError(mode)
        st = self._st()
        st.mode, st.k, st.ks, st.tracks = mode, k, tuple(ks), {}
        dev = self.model_runner.device
        for key, (start, tokens) in seqs.items():
            n = len(tokens) - start
            t = _Track(start=start, tokens=torch.tensor(tokens, dtype=torch.long, device=dev))
            t.seen = torch.zeros(n, dtype=torch.bool, device=dev)
            t.out['tgt'] = torch.full((n,), float('nan'), device=dev)
            if mode == 'ref':
                t.out.update(topv=torch.empty(n, k, device=dev), topi=torch.empty(n, k, dtype=torch.int32, device=dev),
                             log_rest=torch.empty(n, device=dev))
            else:
                with np.load(ref_files[key]) as z:
                    t.anchor = TopK(torch.from_numpy(z['topv']).to(dev), torch.from_numpy(z['topi']).to(dev),
                                    torch.from_numpy(z['log_rest']).to(dev))
                if t.anchor.topv.shape[0] != n:
                    raise ValueError(f'{key}: reference has {t.anchor.topv.shape[0]} rows, sequence has {n}')
                for kk in st.ks:
                    t.out[f'kl_{kk}'] = torch.full((n,), float('nan'), dtype=torch.float64, device=dev)
                t.out['flip'] = torch.zeros(n, dtype=torch.bool, device=dev)
            st.tracks[key] = t
        return True

    def _take(self, st: _State, t: _Track, hidden: torch.Tensor, lo: int, hi: int, pos0: int, n_rows: int) -> None:
        n_tok = t.tokens.shape[0]
        # scored positions start-1 .. n_tok-2 that fall in this step (positions pos0 .. pos0 + hi - lo - 1)
        a, b = max(pos0, t.start - 1), min(pos0 + hi - lo, n_tok - 1)
        if b <= a:
            return
        first, last = lo + a - pos0, lo + b - pos0
        block = engine_chunk()
        compute_logits = self.model_runner.model.compute_logits
        for s0 in range(first - first % block, last, block):
            e0 = min(s0 + block, n_rows)
            s, e = max(a, pos0 + s0 - lo), min(b, pos0 + e0 - lo)
            if e <= s:
                continue
            logits = compute_logits(hidden[s0:e0])
            lp = logits.float().log_softmax(-1)[s - (pos0 + s0 - lo): e - (pos0 + s0 - lo)]
            r = slice(s - (t.start - 1), e - (t.start - 1))
            tgt = t.tokens[s + 1: e + 1]
            t.out['tgt'][r] = lp.gather(1, tgt[:, None])[:, 0]
            if st.mode == 'ref':
                top = reference_topk(lp, st.k)
                t.out['topv'][r], t.out['topi'][r], t.out['log_rest'][r] = top
            else:
                anchor = TopK(*(x[r] for x in t.anchor))
                for kk, v in kl_multi(anchor, lp, st.ks).items():
                    t.out[f'kl_{kk}'][r] = v
                t.out['flip'][r] = anchor.topi[:, 0].long() != lp.argmax(-1)
            t.seen[r] = True
            del logits, lp

    def prefill_kl_end(self) -> dict[str, dict[str, tuple[str, list[int], bytes]]]:
        st = self._st()
        tracks, st.tracks = st.tracks, {}
        out = {}
        for key, t in tracks.items():
            missing = int((~t.seen).sum())
            got = {'missing': np.array(missing)} if missing else {n: v.cpu().numpy() for n, v in t.out.items()}
            out[key] = {n: (arr.dtype.str, list(arr.shape), arr.tobytes()) for n, arr in got.items()}
        return out


def unpack(packed: dict[str, tuple[str, list[int], bytes]]) -> dict[str, np.ndarray]:
    return {n: np.frombuffer(b, dtype=np.dtype(d)).reshape(s).copy() for n, (d, s, b) in packed.items()}
