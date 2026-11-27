import pytest

torch = pytest.importorskip('torch')

from m31eval.prefill_kl.topk import kl_exact, kl_multi, reference_topk  # noqa: E402


def merged_kl(a, c, ids):
    """KL after merging every token outside ``ids`` into one bucket, computed explicitly per row in float64."""
    out = []
    for r in range(a.shape[0]):
        keep = ids[r].long()
        mask = torch.ones(a.shape[1], dtype=torch.bool)
        mask[keep] = False
        pa = torch.cat([a[r, keep].double().exp(), a[r, mask].double().exp().sum()[None]])
        pc = torch.cat([c[r, keep].double().exp(), c[r, mask].double().exp().sum()[None]])
        out.append(float((pa * (pa.log() - pc.log())).sum()))
    return torch.tensor(out, dtype=torch.float64)


def test_multi_k_matches_explicit_merge_and_bounds():
    g = torch.Generator().manual_seed(0)
    V, T = 3000, 7
    a = (torch.randn(T, V, generator=g) * 3).log_softmax(-1)
    c = (a + 0.3 * torch.randn(T, V, generator=g)).log_softmax(-1)
    ref = reference_topk(a, 500)
    got = kl_multi(ref, c, (20, 128, 500))
    full = kl_exact(a, c)
    for k, v in got.items():
        want = merged_kl(a, c, ref.topi[:, :k])
        assert torch.allclose(v, want, rtol=1e-6, atol=1e-9), k
        assert torch.all(v <= full + 1e-9)
    assert torch.all(got[20] <= got[128] + 1e-9) and torch.all(got[128] <= got[500] + 1e-9)


def test_full_vocabulary_k_equals_exact_and_self_is_zero():
    g = torch.Generator().manual_seed(1)
    a = torch.randn(4, 300, generator=g).log_softmax(-1)
    c = torch.randn(4, 300, generator=g).log_softmax(-1)
    ref = reference_topk(a, 300)
    assert torch.allclose(kl_multi(ref, c, (300,))[300], kl_exact(a, c), rtol=1e-6)
    assert torch.allclose(kl_multi(ref, a, (20, 300))[20], torch.zeros(4, dtype=torch.float64), atol=1e-12)
