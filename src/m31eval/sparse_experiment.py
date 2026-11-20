"""Experimental fixed-partition KL. Not used by the full-vocabulary CLI.

Complete input log probabilities are normalized only after validating their
mass. The correction is explicit in every exported record. Tail mass is
summed directly in log space; no minimum tail, token removal, or position drop.
"""
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SparseDistribution:
    token_ids: tuple
    logprobs: tuple
    tail_logprob: float
    vocab_size: int
    normalization_log_mass: float


def logsumexp(values):
    values = tuple(values)
    if not values:
        return -math.inf
    m = max(values)
    if m == -math.inf:
        return m
    return m + math.log(math.fsum(math.exp(v-m) for v in values))


def _validated_mass(full, tolerance):
    if not full or set(full) != set(range(len(full))):
        raise ValueError('Complete contiguous vocabulary is required')
    if any(math.isnan(v) or v == math.inf or v > 1e-5 for v in full.values()):
        raise ValueError('Malformed log probabilities')
    z = logsumexp(full.values())
    if not math.isfinite(z) or abs(math.expm1(z)) > tolerance:
        raise ValueError('Full probability mass is outside tolerance')
    return z


def reference_partition(full, k):
    _validated_mass(full, 2e-5)
    if not isinstance(k, int) or k < 1 or k > len(full):
        raise ValueError('K must be between 1 and vocabulary size')
    return tuple(sorted(full, key=lambda i: (-full[i], i))[:k])


def compress(full, token_ids, mass_tolerance=2e-5):
    z = _validated_mass(full, mass_tolerance)
    ids = tuple(token_ids)
    selected = set(ids)
    if not ids or len(selected) != len(ids) or not selected <= full.keys():
        raise ValueError('Invalid fixed partition')
    tail = logsumexp(v for i, v in full.items() if i not in selected)
    return SparseDistribution(ids, tuple(full[i]-z for i in ids), tail-z,
                              len(full), z)


def compress_gpu(full, token_ids, mass_tolerance=2e-5):
    """Reduce a complete 1D GPU logprob tensor before copying sparse values.

    PyTorch is an optional experiment dependency. Model execution, API export,
    and TP integration are deliberately outside this standalone helper.
    """
    import torch
    if full.ndim != 1 or not full.is_cuda:
        raise ValueError('Expected a one-dimensional GPU logprob tensor')
    values = full.to(torch.float64)
    if bool((torch.isnan(values) | torch.isposinf(values) | (values > 1e-5)).any().item()):
        raise ValueError('Malformed log probabilities')
    z = torch.logsumexp(values, dim=0)
    correction = z.item()
    if not math.isfinite(correction) or abs(math.expm1(correction)) > mass_tolerance:
        raise ValueError('Full probability mass is outside tolerance')
    ids = tuple(token_ids)
    if not ids or len(set(ids)) != len(ids) or min(ids) < 0 or max(ids) >= values.numel():
        raise ValueError('Invalid fixed partition')
    index = torch.tensor(ids, dtype=torch.long, device=values.device)
    mask = torch.ones(values.numel(), dtype=torch.bool, device=values.device)
    mask[index] = False
    tail = torch.logsumexp(values[mask], dim=0)-z
    # Exactly K+1 numeric log probabilities cross to CPU here.
    exported = torch.cat((values[index]-z, tail.reshape(1))).cpu().tolist()
    return SparseDistribution(ids, tuple(exported[:-1]), exported[-1],
                              values.numel(), correction)


def sparse_kl(p, q):
    if p.token_ids != q.token_ids or p.vocab_size != q.vocab_size:
        raise ValueError('Reference and candidate must use the same partition')
    for dist in (p, q):
        if len(dist.logprobs) != len(dist.token_ids):
            raise ValueError('Malformed sparse record')
        values = (*dist.logprobs, dist.tail_logprob)
        if any(math.isnan(v) or v == math.inf for v in values):
            raise ValueError('Malformed sparse record')
        if abs(math.expm1(logsumexp(values))) > 1e-10:
            raise ValueError('Sparse record is not normalized')
    terms = []
    for lp, lq in zip((*p.logprobs, p.tail_logprob), (*q.logprobs, q.tail_logprob)):
        if lp == -math.inf:
            continue  # Exact zero P contributes zero; the position is retained.
        if lq == -math.inf:
            return math.inf  # True zero Q at positive P, not a dropped position.
        terms.append(math.exp(lp)*(lp-lq))
    value = math.fsum(terms)
    if value < -1e-12:
        raise ValueError('Negative KL beyond numerical tolerance')
    return max(0.0, value)
