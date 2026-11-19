import concurrent.futures
import gzip
import json
import math
import statistics
import time
from pathlib import Path

from .api import CapabilityError, Server
from .common import digest, json_compatible, new_output, public_url, read_jsonl, write_json, write_jsonl
from .metrics import agreement, cluster_bootstrap, domain_bootstrap, kl, summarize


def selected_rows(path, limit=None, split=None):
    rows = read_jsonl(path)
    if split:
        rows = [r for r in rows if r.get("split") == split]
    if limit is not None:
        rows = rows[:limit]
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Dataset must be nonempty and have unique IDs")
    return rows


def _distribution(step):
    if step is None:
        raise CapabilityError("Server omitted logprobs")
    return {int(k): float(v) for k, v in step.items()}


def _cache_path(directory, sample_id):
    return Path(directory) / "samples" / (digest(sample_id) + ".json.gz")


def _store_sample(directory, row):
    path = _cache_path(directory, row["id"])
    path.parent.mkdir(exist_ok=True)
    with gzip.open(str(path) + ".part", "wt", encoding="utf-8") as f:
        json.dump(json_compatible(row), f, ensure_ascii=False, allow_nan=False)
    Path(str(path) + ".part").replace(path)


def _load_sample(directory, sample_id):
    with gzip.open(_cache_path(directory, sample_id), "rt", encoding="utf-8") as f:
        return json.load(f)


def _normalize_chat(response):
    msg = response["message"]
    return {"content": msg.get("content"), "reasoning_content": msg.get("reasoning_content"),
            "tool_calls": [t.get("function") for t in (msg.get("tool_calls") or [])]}


def _parallel(rows, worker, workers):
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker, r): r for r in rows}
        for future in concurrent.futures.as_completed(futures):
            source = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {"id": source["id"], "task": source.get("task", source["id"]),
                       "domain": source.get("domain", "unknown"),
                       "split": source.get("split", "unspecified"), "error": str(exc)}
            yield row


def capture(args):
    rows = selected_rows(args.dataset, args.limit, args.split)
    server = Server(args.base_url, args.model, args.api_key_env, args.timeout)
    server.discover()
    server.deadline = time.monotonic() + args.time_budget
    positions = sorted(set(args.positions))
    if not positions or positions[0] < 0 or positions[-1] >= args.tokens:
        raise ValueError("Scored positions must lie inside --tokens")
    metadata = {"schema": 2, "dataset": rows, "dataset_sha256": digest(rows),
        "reference_kind": args.reference_kind, "reference_model": server.model,
        "reference_url": public_url(server.base), "metric": args.metric, "tokens": args.tokens,
        "positions": positions, "max_prompt_tokens": args.max_prompt_tokens,
        "calibration_disjoint": "unverified", "logprobs_semantics": "raw, temperature-1 next-token distribution"}
    if args.calibration_manifest:
        cal = read_jsonl(args.calibration_manifest)
        cal_ids = {r.get("task") for r in cal if r.get("task")}
        cal_hashes = {r.get("input_sha256") or digest({k: r[k] for k in ["messages", "prompt", "tools"] if k in r}) for r in cal}
        overlap = [r["id"] for r in rows if r.get("task") in cal_ids or r.get("input_sha256") in cal_hashes]
        if overlap:
            raise ValueError(f"Calibration/evaluation overlap: {overlap[:10]}")
        metadata["calibration_disjoint"] = "provided-manifest-exact-task/hash-check; near-duplicates not checked"
    signature = digest(metadata)
    directory = Path(args.output)
    if directory.exists():
        old = json.loads((directory / "run.json").read_text())
        if not args.resume or old.get("signature") != signature:
            raise ValueError("Existing reference cache: use --resume with identical configuration")
    else:
        new_output(directory, {**metadata, "signature": signature})
    start = time.monotonic()

    def worker(row):
        saved = _cache_path(directory, row["id"])
        if args.resume and saved.exists():
            prior = _load_sample(directory, row["id"])
            if not prior.get("error"):
                return prior
        result = {"id": row["id"], "task": row.get("task", row["id"]),
                  "domain": row.get("domain", "unknown"), "input_sha256": digest(row)}
        if args.metric in ("chat", "responses"):
            result["chat"] = getattr(server, args.metric)(row, args.tokens)
        else:
            tokens = server.tokenize(row)
            if len(tokens) > args.max_prompt_tokens:
                raise ValueError(f"Prompt has {len(tokens)} tokens, exceeds budget {args.max_prompt_tokens}; not truncated")
            generation = server.generate(tokens, args.tokens)
            result.update(prompt_tokens=tokens, generation={k: v for k, v in generation.items() if k != "distributions"})
            result["continuation_text"] = server.detokenize(generation["token_ids"])
            result["scores"] = []
            for position in positions:
                if position >= len(generation["token_ids"]):
                    continue
                if args.metric == "full":
                    scored = server.generate(tokens + generation["token_ids"][:position], 1, -1, score=True)
                    lp = _distribution(scored["distributions"][0])
                    kl(lp, lp)
                else:
                    continue
                top1 = max(lp, key=lp.get)
                result["scores"].append({"position": position, "logprobs": lp,
                                         "top1": top1, "reference_returned_keys": len(lp),
                                         "kl_available": True})
        _store_sample(directory, result)
        return result

    completed = []
    for row in _parallel(rows, worker, args.concurrency):
        if row.get("error"):
            _store_sample(directory, row)
        completed.append({k: row[k] for k in ["id", "task", "domain", "error"] if k in row})
        print(f"reference {len(completed)}/{len(rows)} {row['id']} {'ERROR: '+row['error'] if row.get('error') else 'OK'}", flush=True)
    summary = {"expected": len(rows), "completed": sum(not r.get("error") for r in completed),
               "errors": sum(bool(r.get("error")) for r in completed), "elapsed_seconds": time.monotonic() - start,
               "reference_kind": args.reference_kind, "metric": args.metric, "signature": signature}
    summary["status"] = "complete" if not summary["errors"] else "incomplete"
    write_jsonl(directory / "index.jsonl", sorted(completed, key=lambda r: r["id"]))
    write_json(directory / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if not summary["errors"] else 2


def evaluate(args):
    reference = Path(args.reference)
    metadata = json.loads((reference / "run.json").read_text())
    baseline_summary = json.loads((reference / "summary.json").read_text())
    if baseline_summary["status"] != "complete":
        raise ValueError("Reference cache is incomplete; resume it before evaluating")
    if metadata["metric"] not in ("full", "agreement", "chat", "responses"):
        raise ValueError("Unsupported reference metric; recapture a full-vocabulary reference")
    rows = metadata["dataset"]
    server = Server(args.base_url, args.model, args.api_key_env, args.timeout)
    server.discover()
    server.deadline = time.monotonic() + args.time_budget
    metric = metadata["metric"]
    run_metadata = {"reference_signature": metadata["signature"],
        "reference_kind": metadata["reference_kind"], "reference_model": metadata["reference_model"],
        "dataset_sha256": metadata["dataset_sha256"], "candidate_model": server.model,
        "candidate_url": public_url(server.base), "metric": metric, "tokens": metadata["tokens"],
        "positions": metadata["positions"], "prompt_policy": "reference_raw_ids" if args.use_reference_prompt else "matching_rendered_tokens"}
    if metric == "full":
        run_metadata["probability_storage"] = "reference: full raw logprobs; candidate: full distribution scored in memory, digest and top-token evidence retained"
    reused = {}
    prior_elapsed = 0.0
    if args.resume_from:
        source = Path(args.resume_from)
        previous = json.loads((source / "run.json").read_text())
        for key, value in run_metadata.items():
            if previous.get(key) != value:
                raise ValueError(f"Cannot resume evaluation with different {key}")
        previous_summary = json.loads((source / "summary.json").read_text())
        previous_rows = read_jsonl(source / "samples.jsonl")
        expected_ids = {row["id"] for row in rows}
        source_ids = [row["id"] for row in previous_rows]
        if (len(set(source_ids)) != len(source_ids) or set(source_ids) != expected_ids
                or previous_summary.get("expected") != len(rows)
                or previous_summary.get("completed") != sum(not r.get("error") for r in previous_rows)
                or previous_summary.get("errors") != sum(bool(r.get("error")) for r in previous_rows)):
            raise ValueError("Resume source must contain unique expected samples and consistent summary counts")
        reused = {row["id"]: row for row in previous_rows if not row.get("error")}
        prior_elapsed = float(previous_summary["elapsed_seconds"])
        run_metadata.update(resumed_from=str(source.resolve()), resume_source_samples_sha256=digest(previous_rows),
                            reused_samples=len(reused), retried_samples=len(rows) - len(reused),
                            prior_elapsed_seconds=prior_elapsed)
    directory = new_output(args.output, run_metadata)
    start = time.monotonic()

    def worker(row):
        if row["id"] in reused:
            return reused[row["id"]]
        cache = _load_sample(reference, row["id"])
        result = {"id": row["id"], "task": row.get("task", row["id"]),
                  "domain": row.get("domain", "unknown"), "split": row.get("split", "unspecified")}
        if metric in ("chat", "responses"):
            candidate = getattr(server, metric)(row, metadata["tokens"])
            result.update(chat=candidate, reference_chat=cache["chat"],
                          chat_message_exact=(_normalize_chat(candidate) == _normalize_chat(cache["chat"]))
                          if metric == "chat" else candidate["output"] == cache["chat"]["output"])
            return result
        own_tokens = server.tokenize(row)
        if not args.use_reference_prompt and own_tokens != cache["prompt_tokens"]:
            raise CapabilityError("Rendered prompts differ: align tokenizer/template, or explicitly use --use-reference-prompt")
        # Check reference continuation IDs retain their meaning on candidate tokenizer.
        if server.detokenize(cache["generation"]["token_ids"]) != cache["continuation_text"]:
            raise CapabilityError("Reference token IDs decode differently: incompatible tokenizers")
        tokens = cache["prompt_tokens"]
        generated = server.generate(tokens, metadata["tokens"])
        result.update(agreement(cache["generation"]["token_ids"], generated["token_ids"]))
        result["candidate_generation"] = generated
        result["reference_generation"] = cache["generation"]
        result["scored_positions"] = []
        for scored in cache["scores"]:
            position = scored["position"]
            p = {int(k): float(v) for k, v in scored["logprobs"].items()}
            try:
                generated_score = server.generate(tokens + cache["generation"]["token_ids"][:position], 1, -1)
                q = _distribution(generated_score["distributions"][0])
                measured = kl(p, q)
            except Exception as exc:
                result["scored_positions"].append({"position": position, "kl": None,
                    "error": str(exc), "reference_coverage": None})
                result["error"] = "One or more requested positions failed; see scored_positions"
                continue
            # raw_logprobs are temperature-1 probabilities; sampled IDs are greedy.
            evidence = {"candidate_distribution_sha256": digest(q),
                         "candidate_vocab_size": len(q),
                         "reference_top1": scored["top1"],
                         "candidate_top1": generated_score["token_ids"][0],
                         "reference_top1_logprob": p[scored["top1"]],
                         "candidate_reference_top1_logprob": q[scored["top1"]]}
            result["scored_positions"].append({"position": position, **measured,
                                               "top1_agreement": generated_score["token_ids"][0] == scored["top1"],
                                               **evidence})
        informative = [r for r in result["scored_positions"] if r["kl"] is not None]
        result["mean_kl"] = statistics.mean(r["kl"] for r in informative) if informative else None
        result["kl_positions"] = len(informative)
        result["kl_unavailable_positions"] = len(result["scored_positions"]) - len(informative)
        result["min_reference_coverage"] = min((r["reference_coverage"] for r in informative), default=None)
        top1_checks = [r["top1_agreement"] for r in result["scored_positions"] if "top1_agreement" in r]
        result["same_context_top1_agreement"] = statistics.mean(top1_checks) if top1_checks else None
        return result

    completed = []
    with (directory / "samples.jsonl").open("w", encoding="utf-8") as f:
        for row in _parallel(rows, worker, args.concurrency):
            f.write(json.dumps(json_compatible(row), ensure_ascii=False, allow_nan=False) + "\n")
            f.flush()
            completed.append(row)
            print(f"evaluate {len(completed)}/{len(rows)} {row['id']} {'ERROR: '+row['error'] if row.get('error') else 'OK'}", flush=True)
    if metric in ("chat", "responses"):
        ok = [r for r in completed if not r.get("error")]
        summary = {"metric": f"parsed_{metric}_message_exact; not token agreement or KL", "expected": len(rows),
                   "completed": len(ok), "errors": len(rows) - len(ok),
                   "chat_message_exact_rate": statistics.mean(r["chat_message_exact"] for r in ok) if ok else None,
                   "status": "complete" if len(ok) == len(rows) else "incomplete"}
    else:
        summary = summarize(completed, "full_kl" if metric == "full" else "greedy_token_agreement")
        summary["by_split"] = {split: summarize([r for r in completed if r.get("split") == split], summary["metric"])
                               for split in sorted({r.get("split") for r in completed if r.get("split")})}
    current_elapsed = time.monotonic() - start
    summary.update(reference_kind=metadata["reference_kind"], reference_signature=metadata["signature"],
                   candidate_model=server.model, elapsed_seconds=prior_elapsed + current_elapsed)
    if args.resume_from:
        summary.update(reused_samples=len(reused), retried_samples=len(rows) - len(reused),
                       prior_elapsed_seconds=prior_elapsed, resume_elapsed_seconds=current_elapsed,
                       elapsed_scope="sum of original and recovery evaluations; excludes idle time between attempts")
    summary["conclusion_scope"] = "fidelity on this dataset; task capability improvement requires verifier results"
    write_json(directory / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if not summary["errors"] else 2


def compare(old_directory, new_directory, output, noise_floor=0.0, aggregation="task"):
    if noise_floor < 0:
        raise ValueError("Noise floor must be nonnegative")
    a = json.loads((Path(old_directory) / "run.json").read_text())
    b = json.loads((Path(new_directory) / "run.json").read_text())
    for key in ["reference_signature", "metric", "prompt_policy"]:
        if a.get(key) != b.get(key):
            raise ValueError(f"Cannot compare runs with different {key}")
    old_rows = read_jsonl(Path(old_directory) / "samples.jsonl")
    new_rows = read_jsonl(Path(new_directory) / "samples.jsonl")
    for directory, rows in [(old_directory, old_rows), (new_directory, new_rows)]:
        summary = json.loads((Path(directory) / "summary.json").read_text())
        if summary.get("status") != "complete" or summary.get("expected") != len(rows):
            raise ValueError("Comparison requires complete evaluation runs and every expected sample")
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError("Comparison sample IDs must be unique")
    old = {r["id"]: r for r in old_rows}
    new = {r["id"]: r for r in new_rows}
    if old.keys() != new.keys():
        raise ValueError("Runs have different sample IDs")
    pairs = []
    errors = []
    unscored = []
    for key in old:
        x, y = old[key], new[key]
        if x.get("error") or y.get("error"):
            errors.append(key)
            continue
        if x.get("mean_kl") is not None and y.get("mean_kl") is not None:
            pairs.append({"id": key, "task": x["task"], "domain": x.get("domain", "unknown"),
                          "delta_kl": y["mean_kl"] - x["mean_kl"]})
        else:
            unscored.append(key)
    if aggregation not in ("task", "domain_macro"):
        raise ValueError("Unknown comparison aggregation")
    confidence = domain_bootstrap(pairs, "delta_kl") if aggregation == "domain_macro" else cluster_bootstrap(pairs, "delta_kl")
    conclusion = "inconclusive"
    if not errors and not unscored and confidence and confidence["clusters"] >= 2:
        lo, hi = confidence["ci95"]
        if hi < -noise_floor:
            conclusion = "lower measured divergence; promote to task verification"
        elif lo > noise_floor:
            conclusion = "higher measured divergence"
    report = {"paired_prompts": len(pairs), "error_pairs": errors, "unscored_pairs": unscored,
              "noise_floor_nat": noise_floor, "aggregation": aggregation, "delta_kl": confidence,
              "by_domain": {d: cluster_bootstrap([r for r in pairs if r["domain"] == d], "delta_kl")
                            for d in sorted({r["domain"] for r in pairs})},
              "conclusion": conclusion, "scope": "fixed-reference fidelity, not task accuracy"}
    write_json(output, report)
    print(json.dumps(report, indent=2))
    return 0 if not errors else 2


def doctor(args):
    server = Server(args.base_url, args.model, args.api_key_env, args.timeout)
    report = {"url": public_url(server.base), "models": server.discover(), "model": server.model}
    smoke = {"messages": [{"role": "user", "content": "What is the capital of China? and how many cities in China?"}]}
    try:
        tokens = server.tokenize(smoke)
        report["tokenize"] = True
        if args.probe:
            generated = server.generate(tokens, 100, 2)
            report["generation"] = generated
            if generated["distributions"]:
                ids = list(generated["distributions"][0])
                measured = server.generate(tokens, 1, 2, requested=ids, score=True)
                report["requested_token_logprobs"] = set(ids) <= measured["distributions"][0].keys()
        if getattr(args, "probe_full", False):
            measured = server.generate(tokens, 1, -1)
            distribution = _distribution(measured["distributions"][0])
            kl(distribution, distribution)
            expected = getattr(args, "expected_vocab_size", None)
            if expected is not None and set(distribution) != set(range(expected)):
                raise CapabilityError(f"Full vocabulary mismatch: returned {len(distribution)}, expected all {expected} IDs")
            report["full_vocabulary_logprobs"] = True
            report["full_vocab_size"] = len(distribution)
            report["full_probability_mass"] = math.fsum(math.exp(v) for v in distribution.values())
    except Exception as exc:
        report["native_error"] = str(exc)
        if args.probe:
            try:
                report["chat_smoke"] = server.chat(smoke, 100)
            except Exception as chat_exc:
                report["chat_error"] = str(chat_exc)
                report["responses_smoke"] = server.responses(smoke, 100)
    report["guidance"] = "Native tokenization/completions needed for KL. A chat/Responses-only proxy can use --metric chat/responses; it cannot measure token KL."
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0
