import argparse
import json
import sys


def _server_options(p):
    p.add_argument("--base-url", required=True)
    p.add_argument("--model")
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--timeout", type=float, default=120)


def _quick_options(p):
    _server_options(p)
    p.add_argument("--output", required=True)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--time-budget", type=float, default=1800, help="Soft wall-clock budget in seconds; in-flight HTTP requests share the deadline")


def parser():
    p = argparse.ArgumentParser(prog="m31-eval")
    commands = p.add_subparsers(dest="command", required=True)
    mixed = commands.add_parser("mixed", help="Fetch/build/audit a stratified multi-domain fidelity dataset")
    mixsub = mixed.add_subparsers(dest="action", required=True)
    mixfetch = mixsub.add_parser("fetch")
    mixfetch.add_argument("--assets", required=True)
    mixbuild = mixsub.add_parser("build")
    mixbuild.add_argument("--assets", required=True)
    mixbuild.add_argument("--terminal-version", choices=["2.1", "4.0"], default="4.0")
    mixbuild.add_argument("--seed", type=int, default=31)
    mixbuild.add_argument("--deepswe-prompts", help="Frozen 60-prompt JSONL; defaults to bundled data/mixed/deepswe60.jsonl")
    mixbuild.add_argument("--public-only", action="store_true", help="Build the four-category 240-prompt variant without shared DeepSWE trace inputs")
    mixbuild.add_argument("--output", default="data/mixed")
    mixaudit = mixsub.add_parser("audit")
    _server_options(mixaudit)
    mixaudit.add_argument("--dataset", required=True)
    mixaudit.add_argument("--max-prompt-tokens", type=int, default=65536)
    mixaudit.add_argument("--output", required=True)
    doc = commands.add_parser("doctor", help="Discover native token/logprob API capabilities")
    _server_options(doc)
    doc.add_argument("--output", required=True)
    doc.add_argument("--probe", action="store_true", help="Generate the fixed 100-token smoke question and test logprob support")
    doc.add_argument("--probe-full", action="store_true", help="Check full-vocabulary raw logprobs with logprobs=-1")
    doc.add_argument("--expected-vocab-size", type=int, help="Require every token ID in the configured model vocabulary")
    ref = commands.add_parser("reference", help="Cache greedy generation and reference probability data once")
    _quick_options(ref)
    ref.add_argument("--dataset", required=True)
    ref.add_argument("--reference-kind", required=True, choices=["bf16", "nvfp4", "mxfp8", "mxfp4", "other"])
    ref.add_argument("--metric", default="coarse", choices=["coarse", "full", "agreement", "chat", "responses"])
    ref.add_argument("--tokens", type=int, default=32)
    ref.add_argument("--positions", type=int, nargs="+", default=[0, 1, 2, 3, 7, 15, 23, 31])
    ref.add_argument("--top-k", type=int, default=20)
    ref.add_argument("--max-prompt-tokens", type=int, default=8192)
    ref.add_argument("--limit", type=int)
    ref.add_argument("--split", choices=["development", "holdout"])
    ref.add_argument("--calibration-manifest")
    ref.add_argument("--resume", action="store_true")
    ev = commands.add_parser("evaluate", help="Connect candidate server to an existing reference cache")
    _quick_options(ev)
    ev.add_argument("--reference", required=True)
    ev.add_argument("--resume-from", help="Reuse successful samples from an identical evaluation in a new output directory; retry errors")
    ev.add_argument("--use-reference-prompt", action="store_true", help="Explicitly evaluate reference raw input IDs despite a template mismatch; same tokenizer still required")
    co = commands.add_parser("compare", help="Task-cluster paired bootstrap of two candidate runs")
    co.add_argument("--old", required=True)
    co.add_argument("--new", required=True)
    co.add_argument("--output", required=True)
    co.add_argument("--noise-floor", type=float, default=0.0, help="Require paired CI to exceed this A/A-calibrated absolute KL difference (nat)")
    co.add_argument("--aggregation", choices=["task", "domain_macro"], default="task",
                    help="Use equal domain weight with task resampling for mixed datasets")
    return p


def main():
    args = parser().parse_args()
    try:
        if args.command == "mixed":
            from .mixed import fetch, build, audit
            return {"fetch": fetch, "build": build, "audit": audit}[args.action](args)
        from . import quick
        if args.command == "compare":
            return quick.compare(args.old, args.new, args.output, args.noise_floor, args.aggregation)
        return getattr(quick, {"reference": "capture", "evaluate": "evaluate", "doctor": "doctor"}[args.command])(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
