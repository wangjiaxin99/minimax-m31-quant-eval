"""Top-K + tail KL estimators on GPU tensors (positions on the first axis).

At one position the reference has log-probabilities ``a`` and the candidate ``c`` over V tokens. The reference
keeps its K most likely tokens (``topv`` descending, ``topi``) and ``log_rest``, the log of its mass outside them.
For any K' <= K the estimate is exact over the first K' tokens and merges the rest into one bucket::

    KL_K' = sum_{i < K'} p_a(i) (a_i - c_i)  +  m_a (log m_a - log m_c)

``m`` is each side's mass outside the reference's first K' ids. Every tail is a log-sum-exp, never ``1 - sum``:
the candidate's tail over the K ids is a masked log-sum-exp of its row, and a smaller K' adds the reference's
dropped ids back with ``logaddexp``. Merging outcomes never increases KL, so ``KL_K' <= KL`` (equal at K' = V).
The reductions after the log-softmax run in float64.
"""
from __future__ import annotations

from typing import NamedTuple

import torch


class TopK(NamedTuple):
    topv: torch.Tensor  # [T, K] float32, descending
    topi: torch.Tensor  # [T, K] int32
    log_rest: torch.Tensor  # [T] float32, -inf when K covers the vocabulary


def log_outside(logp: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Per row, log of the mass of ``logp`` outside the columns ``ids`` (-inf when nothing remains)."""
    return torch.logsumexp(logp.scatter(1, ids.long(), float('-inf')), dim=-1)


def reference_topk(logp: torch.Tensor, k: int) -> TopK:
    v, i = logp.topk(min(k, logp.shape[-1]), dim=-1)
    return TopK(v, i.to(torch.int32), log_outside(logp, i))


def _bucket(log_m_a: torch.Tensor, log_m_c: torch.Tensor) -> torch.Tensor:
    term = log_m_a.exp() * (log_m_a - log_m_c)
    return torch.where(torch.isneginf(log_m_a), torch.zeros_like(term), term)


def kl_multi(ref: TopK, cand_logp: torch.Tensor, ks: tuple[int, ...]) -> dict[int, torch.Tensor]:
    """Per-position KL_K'(reference || candidate) for each K' in ``ks`` (each <= the stored K), float64."""
    K = ref.topi.shape[1]
    a = ref.topv.double()
    c = cand_logp.gather(1, ref.topi.long()).double()
    rest_a = ref.log_rest.double()
    rest_c = log_outside(cand_logp, ref.topi).double()
    contrib = a.exp() * (a - c)
    out = {}
    for k in ks:
        if k > K:
            raise ValueError(f'K={k} exceeds the stored reference K={K}')
        if k == K:
            ra, rc = rest_a, rest_c
        else:
            ra = torch.logaddexp(rest_a, torch.logsumexp(a[:, k:], dim=-1))
            rc = torch.logaddexp(rest_c, torch.logsumexp(c[:, k:], dim=-1))
        out[k] = contrib[:, :k].sum(-1) + _bucket(ra, rc)
    return out


def kl_exact(ref_logp: torch.Tensor, cand_logp: torch.Tensor) -> torch.Tensor:
    """Full-vocabulary KL(reference || candidate) per row: the slow check path, float64."""
    a, c = ref_logp.double(), cand_logp.double()
    p = a.exp()
    return torch.where(p > 0, p * (a - c), torch.zeros_like(p)).sum(-1)
