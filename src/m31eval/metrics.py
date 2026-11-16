"""KL operates on raw next-token distributions at identical token prefixes."""
import math
import random
import statistics


def _validate_logprobs(lp):
    if not lp or any(math.isnan(v) or v > 1e-5 for v in lp.values()):
        raise ValueError("Invalid log probabilities")


def coarse_partition(logprobs, min_tail_mass=1e-5):
    """Freeze a coarsening using P alone, away from float32 tail cancellation.

    Remove the smallest returned reference probabilities until the other mass
    is resolvable. An almost deterministic P can leave no informative partition;
    retain its argmax check, but do not invent tail probabilities or a KL score.
    """
    _validate_logprobs(logprobs)
    selected = dict(logprobs)
    if math.fsum(math.exp(v) for v in selected.values()) > 1.00002:
        raise ValueError("Reference probability mass exceeds one")
    for token in sorted(selected, key=selected.get):
        if math.fsum(math.exp(v) for v in selected.values()) <= 1 - min_tail_mass:
            break
        del selected[token]
    return selected


def kl(p, q, full=False, mass_tolerance=0.00002):
    """Full KL, or exact KL on the fixed partition {each p key, other}.

    Coarse KL is a lower bound on full KL, not a renormalized top-k KL.
    q MUST contain all reference keys, including their actual probabilities.
    """
    _validate_logprobs(p)
    _validate_logprobs(q)
    missing = p.keys() - q.keys()
    if missing:
        raise ValueError(f"Candidate omitted {len(missing)} requested reference token IDs")
    pm = math.fsum(math.exp(v) for v in p.values())
    qm = math.fsum(math.exp(q[k]) for k in p)
    if pm > 1 + mass_tolerance or qm > 1 + mass_tolerance:
        raise ValueError("Probability mass exceeds one; logprobs may be processed/malformed")
    if full and (p.keys() != q.keys() or abs(pm - 1) > mass_tolerance or abs(qm - 1) > mass_tolerance):
        raise ValueError("Full KL requires the same complete vocabulary and normalized mass")
    value = 0.0
    for k, lp in p.items():
        pv = math.exp(lp)
        if pv:
            if not math.isfinite(q[k]):
                raise ValueError("Candidate assigns zero probability to positive reference mass")
            value += pv * (lp - q[k])
    if not full:
        pt, qt = max(0.0, 1 - pm), max(0.0, 1 - qm)
        if pt:
            if not qt:
                raise ValueError("Candidate other bucket has zero probability")
            value += pt * math.log(pt / qt)
    if value < -mass_tolerance:
        raise ValueError("Negative KL indicates inconsistent probability data")
    return {"kl": max(0.0, value), "reference_coverage": min(1.0, pm),
            "candidate_coverage": min(1.0, qm)}


def agreement(reference, candidate):
    prefix = 0
    for a, b in zip(reference, candidate):
        if a != b:
            break
        prefix += 1
    return {"prefix_tokens": prefix, "reference_tokens": len(reference),
            "candidate_tokens": len(candidate),
            "sequence_exact": reference == candidate,
            "prefix_at_8": len(reference) >= 8 and prefix >= 8,
            "prefix_at_32": len(reference) >= 32 and prefix >= 32,
            "eos_before_32": len(reference) < 32}


def cluster_bootstrap(rows, field, iterations=2000, seed=31):
    groups = {}
    for r in rows:
        if r.get(field) is not None:
            groups.setdefault(r.get("task", r["id"]), []).append(float(r[field]))
    values = [statistics.mean(v) for v in groups.values()]
    if not values:
        return None
    rng = random.Random(seed)
    sims = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(iterations))
    return {"mean": statistics.mean(values), "ci95": [sims[int(.025 * iterations)],
            sims[min(iterations - 1, int(.975 * iterations))]], "clusters": len(values)}


def domain_bootstrap(rows, field, iterations=2000, seed=31):
    """Equal domain weight, resampling original tasks within each domain."""
    groups = {}
    for row in rows:
        if row.get(field) is not None:
            groups.setdefault(row.get("domain", "unknown"), {}).setdefault(row["task"], []).append(float(row[field]))
    values = [[statistics.mean(v) for v in tasks.values()] for tasks in groups.values()]
    if not values:
        return None
    rng = random.Random(seed)
    sims = sorted(statistics.mean(statistics.mean(rng.choices(v, k=len(v))) for v in values)
                  for _ in range(iterations))
    return {"mean": statistics.mean(statistics.mean(v) for v in values),
            "ci95": [sims[int(.025 * iterations)], sims[min(iterations - 1, int(.975 * iterations))]],
            "domains": len(values), "clusters": sum(len(v) for v in values)}


def summarize(rows, metric):
    ok = [r for r in rows if not r.get("error")]
    values = sorted(r["mean_kl"] for r in ok if r.get("mean_kl") is not None)
    out = {"metric": metric, "expected": len(rows), "completed": len(ok),
           "errors": len(rows) - len(ok), "status": "complete" if len(ok) == len(rows) else "incomplete",
           "task_cluster_kl": cluster_bootstrap(ok, "mean_kl"),
           "prompt_mean_kl": statistics.mean(values) if values else None,
           "prompt_p95_kl": values[min(len(values) - 1, int(.95 * len(values)))] if values else None}
    out["same_context_top1_agreement"] = cluster_bootstrap(ok, "same_context_top1_agreement")
    out["domain_macro_kl"] = domain_bootstrap(ok, "mean_kl")
    out["kl_scored_prompts"] = len(values)
    out["kl_scored_positions"] = sum(r.get("kl_positions", 0) for r in ok)
    out["kl_unavailable_positions"] = sum(r.get("kl_unavailable_positions", 0) for r in ok)
    for field in ["sequence_exact", "prefix_at_8", "prefix_at_32"]:
        eligible = [r for r in ok if field != "prefix_at_32" or not r.get("eos_before_32")]
        if field == "prefix_at_8":
            eligible = [r for r in eligible if r.get("reference_tokens", 0) >= 8]
        out[field] = {"rate": statistics.mean(float(r[field]) for r in eligible) if eligible else None,
                      "n": len(eligible)}
    out["by_domain"] = {}
    for domain in sorted({r.get("domain", "unknown") for r in ok}):
        rs = [r for r in ok if r.get("domain", "unknown") == domain]
        out["by_domain"][domain] = {"n": len(rs), "mean_kl": cluster_bootstrap(rs, "mean_kl")}
    return out
