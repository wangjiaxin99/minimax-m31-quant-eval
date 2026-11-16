"""Versioned public sources and stratified, outcome-independent fidelity inputs."""
import collections
import concurrent.futures
import hashlib
import json
import re
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

from .common import digest, read_jsonl, write_json, write_jsonl
BASH_TOOL = {'type': 'function', 'function': {'name': 'bash', 'description': 'Execute a bash command', 'parameters': {'type': 'object', 'properties': {'command': {'type': 'string'}}, 'required': ['command'], 'additionalProperties': False}}}


TERMINAL_PINS = {
    "2.1": ("harbor-framework/terminal-bench-2-1", "c5ee500c185224c97cd6caff7866a990a0057f41"),
    "4.0": ("harbor-framework/terminal-bench", "452bf305c6daa62fc59061d22133a7cbc7c1572e"),
}
MATH_PIN = "d3d08664755704f422af97d43a7ff0ded4bd95df"
MMLU_PIN = "36c2fd756f19ccf13a9a96c8e53ccecc02192b8b"
PYTHON_PIN = "ebf955df7a89ed0c7968f79faec1de49f61ed7cb"
LANGUAGES = ("ar", "bn", "hi", "ja", "ko", "zh")
PYTHON_DOCS = (
    "Doc/c-api/typeobj.rst", "Doc/howto/logging-cookbook.rst", "Doc/library/os.rst",
    "Doc/library/stdtypes.rst", "Doc/library/typing.rst", "Doc/reference/datamodel.rst",
    "Doc/whatsnew/3.14.rst", "Doc/library/sqlite3.rst", "Doc/library/unittest.rst",
    "Doc/library/argparse.rst",
)
BOOKS = {
    1342: ("Pride and Prejudice", "Jane Austen"),
    84: ("Frankenstein", "Mary Wollstonecraft Shelley"),
    2701: ("Moby Dick", "Herman Melville"),
    98: ("A Tale of Two Cities", "Charles Dickens"),
    1661: ("The Adventures of Sherlock Holmes", "Arthur Conan Doyle"),
    345: ("Dracula", "Bram Stoker"),
    1232: ("The Prince", "Niccolo Machiavelli"),
    1497: ("The Republic", "Plato"),
    7370: ("On the Origin of Species", "Charles Darwin"),
    3300: ("An Inquiry into the Nature and Causes of the Wealth of Nations", "Adam Smith"),
}


def fetch(args):
    root = Path(args.assets)
    root.mkdir(parents=True, exist_ok=True)
    previous = root / "download-manifest.json"
    prior = {r["path"]: r for r in json.loads(previous.read_text())["files"]} if previous.exists() else {}
    specs = []
    def add(path, url):
        specs.append((path, url))
    for version, (repo, pin) in TERMINAL_PINS.items():
        tree_url = f"https://api.github.com/repos/{repo}/git/trees/{pin}?recursive=1"
        tree_path = root / f"terminal-{version}/tree.json"
        if tree_path.exists():
            tree = json.loads(tree_path.read_text())
            if tree.get("m31_source_commit") != pin:
                raise ValueError("Cached source tree belongs to a different commit")
        else:
            try:
                with urllib.request.urlopen(tree_url, timeout=60) as r:
                    tree = json.load(r)
            except urllib.error.HTTPError as exc:
                if exc.code != 403 or not shutil.which("gh"):
                    raise
                # Let gh use its existing authentication; no tokens enter artifacts.
                data = subprocess.check_output(["gh", "api", f"repos/{repo}/git/trees/{pin}?recursive=1"])
                tree = json.loads(data)
            tree["m31_source_commit"] = pin
        tasks = sorted({r["path"].split("/")[1] for r in tree["tree"]
                        if re.fullmatch(r"tasks/[^/]+/instruction.md", r["path"])})
        if len(tasks) < 60 or tree.get("truncated"):
            raise ValueError(f"Incomplete Terminal-Bench {version} source tree")
        write_json(tree_path, tree)
        for name in ["LICENSE", "README.md"]:
            add(f"terminal-{version}/{name}", f"https://raw.githubusercontent.com/{repo}/{pin}/{name}")
        for task in tasks:
            for name in ["instruction.md", "task.toml"]:
                add(f"terminal-{version}/tasks/{task}/{name}",
                    f"https://raw.githubusercontent.com/{repo}/{pin}/tasks/{task}/{name}")
        # Retain task-specific copyright/license notices with the instructions.
        for item in tree["tree"]:
            if re.fullmatch(r"tasks/[^/]+/(?:LICENSE(?:\.md|\.txt)?|NOTICE(?:\.md|\.txt)?)", item["path"]):
                add(f"terminal-{version}/{item['path']}",
                    f"https://raw.githubusercontent.com/{repo}/{pin}/{item['path']}")
    for language in LANGUAGES:
        add(f"mmlu/{language}.parquet", f"https://huggingface.co/datasets/CohereLabs/Global-MMLU-Lite/resolve/{MMLU_PIN}/{language}/test-00000-of-00001.parquet")
    add("mmlu/README.md", f"https://huggingface.co/datasets/CohereLabs/Global-MMLU-Lite/raw/{MMLU_PIN}/README.md")
    add("math.parquet", f"https://huggingface.co/datasets/nvidia/OpenMathReasoning/resolve/{MATH_PIN}/data/additional_problems-00000-of-00001.parquet")
    add("math/README.md", f"https://huggingface.co/datasets/nvidia/OpenMathReasoning/raw/{MATH_PIN}/README.md")
    add("licenses/CC-BY-4.0.txt", "https://creativecommons.org/licenses/by/4.0/legalcode.txt")
    add("python/LICENSE", f"https://raw.githubusercontent.com/python/cpython/{PYTHON_PIN}/LICENSE")
    for name in PYTHON_DOCS:
        add(f"python/{name}", f"https://raw.githubusercontent.com/python/cpython/{PYTHON_PIN}/{name}")
    for book in BOOKS:
        add(f"books/{book}.txt", f"https://www.gutenberg.org/cache/epub/{book}/pg{book}.txt")

    def download(spec):
        relative, url = spec
        path = root / relative
        if path.exists() and relative in prior:
            if prior[relative]["url"] != url:
                raise ValueError(f"Source pin changed: use a new assets directory for {relative}")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != prior[relative]["sha256"]:
                raise ValueError(f"Cached source checksum mismatch: {relative}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            req = urllib.request.Request(url, headers={"User-Agent": "m31-eval-source-download/0.2"})
            with urllib.request.urlopen(req, timeout=90) as response, path.with_suffix(path.suffix + ".part").open("wb") as out:
                shutil.copyfileobj(response, out)
            path.with_suffix(path.suffix + ".part").replace(path)
        record = {"path": relative, "url": url, "bytes": path.stat().st_size,
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        print(f"source {relative} {record['bytes']} bytes", flush=True)
        return record
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for record in pool.map(download, specs):
            records.append(record)
    write_json(previous, {"schema": 1, "files": records,
        "note": "Pins freeze Git/HF sources; Gutenberg snapshots require checksum cache for exact rebuild."})
    print(json.dumps({"files": len(records), "bytes": sum(r["bytes"] for r in records)}))
    return 0


def balanced_pick(rows, count, seed, group, unique):
    """Round robin strata; stable hash ranks; never uses model score or reward."""
    buckets = collections.defaultdict(list)
    for row in rows:
        buckets[group(row)].append(row)
    for key in buckets:
        buckets[key].sort(key=lambda r: digest([seed, r["id"]]))
    result, used = [], set()
    while len(result) < count:
        progressed = False
        for key in sorted(buckets):
            while buckets[key] and unique(buckets[key][0]) in used:
                buckets[key].pop(0)
            if buckets[key] and len(result) < count:
                row = buckets[key].pop(0)
                used.add(unique(row))
                result.append(row)
                progressed = True
        if not progressed:
            raise ValueError(f"Only {len(result)} eligible distinct inputs, need {count}")
    return result


def _prompt(identifier, task, domain, content, source, license_id, **metadata):
    return {"id": identifier, "task": task, "domain": domain,
            "messages": [{"role": "user", "content": content}],
            "source": source, "license": license_id, "calibration_disjoint": "unverified", **metadata}


def math_topic(problem):
    text = problem.lower()
    for name, expression in [
        ("geometry", r"triangle|circle|angle|trapezoid|polygon"),
        ("number_theory", r"prime|divisib|modulo|remainder|gcd"),
        ("combinatorics", r"probability|permutation|committee|arrang|how many"),
        ("calculus", r"derivative|integral|\\int|limit"),
        ("algebra", r"polynomial|equation|roots|sequence"),
    ]:
        if re.search(expression, text):
            return name
    return "other"


def strip_gutenberg(text):
    start = re.search(r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", text, re.I)
    end = re.search(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", text, re.I)
    if not start or not end:
        raise ValueError("Unrecognized Gutenberg header/footer; cannot silently strip")
    return text[start.end():end.start()].strip()


def set_splits(rows, excluded_tasks, seed):
    by_domain = collections.defaultdict(set)
    for row in rows:
        by_domain[row["domain"]].add(row["task"])
    holdout = set()
    for domain, tasks in by_domain.items():
        candidates = sorted(tasks - excluded_tasks, key=lambda t: digest([seed, "holdout", t]))
        holdout.update(candidates[:max(1, round(len(tasks) * .2))])
    for row in rows:
        row["split"] = "holdout" if row["task"] in holdout else "development"
    return rows


def validate_inputs(rows, expected=240):
    if len(rows) != expected or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Mixed dataset size/ID mismatch")
    hashes = [digest({k: r[k] for k in ["messages", "tools"] if k in r}) for r in rows]
    if len(set(hashes)) != len(hashes):
        raise ValueError("Duplicate rendered input structures")
    by_task = collections.defaultdict(set)
    for row, hash_value in zip(rows, hashes):
        row["input_sha256"] = hash_value
        by_task[row["task"]].add(row["split"])
    if any(len(s) > 1 for s in by_task.values()):
        raise ValueError("Same source task leaks across development/holdout")


def build(args):
    try:
        import pyarrow.parquet as pq
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
    except ImportError as exc:
        raise ValueError("Dataset rebuilding requires pip install -e '.[build]'; evaluation stays stdlib-only") from exc
    root, out = Path(args.assets), Path(args.output)
    snapshot = json.loads((root / "download-manifest.json").read_text())
    for record in snapshot["files"]:
        if hashlib.sha256((root / record["path"]).read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Source checksum mismatch: {record['path']}")
    seed, rows = args.seed, []
    terminal = []
    repo, pin = TERMINAL_PINS[args.terminal_version]
    instruction_paths = [root / r["path"] for r in snapshot["files"]
                         if r["path"].startswith(f"terminal-{args.terminal_version}/tasks/")
                         and r["path"].endswith("/instruction.md")]
    for path in sorted(instruction_paths):
        config = tomllib.loads((path.parent / "task.toml").read_text())
        metadata = config.get("metadata", {})
        task = path.parent.name
        row = _prompt(f"terminal-{args.terminal_version}:{task}", f"terminal-{args.terminal_version}:{task}",
            "terminal_bench", path.read_text(),
            f"https://github.com/{repo}/blob/{pin}/tasks/{task}/instruction.md", "Apache-2.0",
            source_version=args.terminal_version, source_task=task, source_commit=pin,
            category=metadata.get("category", "unknown"), difficulty=metadata.get("difficulty", "unknown"),
            prompt_kind="initial_task_instruction; no sandbox/filesystem snapshot")
        row["tools"] = [BASH_TOOL]
        terminal.append(row)
    rows.extend(balanced_pick(terminal, 60, seed, lambda r: (r["category"], r["difficulty"]), lambda r: r["task"]))

    math = []
    seen = set()
    table = pq.read_table(root / "math.parquet", columns=["problem", "problem_source"])
    for index, record in enumerate(table.to_pylist()):
        problem = record["problem"] or ""
        key = digest(" ".join(problem.split()))
        if key in seen or not 80 <= len(problem) <= 10000:
            continue
        seen.add(key)
        row = _prompt("openmath:" + key, "openmath:" + key, "math",
            "Solve the following mathematical problem. Explain your reasoning.\n\n" + problem,
            f"https://huggingface.co/datasets/nvidia/OpenMathReasoning/tree/{MATH_PIN}", "CC-BY-4.0",
            source_revision=MATH_PIN, source_split="additional_problems", source_row=index,
            problem_source=record["problem_source"], topic=math_topic(problem),
            topic_is_lexical_proxy=True, answers_in_prompt=False)
        math.append(row)
    rows.extend(balanced_pick(math, 60, seed, lambda r: (r["topic"], "long" if len(r["messages"][0]["content"]) > 800 else "short"), lambda r: r["task"]))

    instructions = {
        "ar": "أجب باللغة العربية. اشرح سبب اختيارك ثم حدد الإجابة الصحيحة.",
        "bn": "বাংলায় উত্তর দিন। যুক্তি ব্যাখ্যা করে সঠিক বিকল্পটি বেছে নিন।",
        "hi": "हिंदी में उत्तर दें। अपना तर्क समझाएँ और सही विकल्प चुनें।",
        "ja": "日本語で答えてください。理由を説明して、正しい選択肢を選んでください。",
        "ko": "한국어로 답하세요. 이유를 설명하고 정답을 선택하세요.",
        "zh": "请用中文回答。解释理由，然后选择正确的选项。",
    }
    used_questions = set()
    for language in LANGUAGES:
        candidates = []
        for record in pq.read_table(root / f"mmlu/{language}.parquet").to_pylist():
            sample = record["sample_id"]
            if sample in used_questions:
                continue
            content = instructions[language] + "\n\n" + record["question"] + "\n\n"
            content += "\n".join(f"{letter.upper()}. {record['option_' + letter]}" for letter in "abcd")
            candidates.append(_prompt(f"gmmlu:{language}:{sample}", "gmmlu:" + sample, "non_latin", content,
                f"https://huggingface.co/datasets/CohereLabs/Global-MMLU-Lite/tree/{MMLU_PIN}", "Apache-2.0",
                language=language, source_revision=MMLU_PIN, source_split="test", source_sample_id=sample,
                subject=record["subject"], subject_category=record["subject_category"], answers_in_prompt=False))
        selected = balanced_pick(candidates, 10, seed, lambda r: r["subject_category"], lambda r: r["task"])
        used_questions.update(r["source_sample_id"] for r in selected)
        rows.extend(selected)

    documents = []
    for path in PYTHON_DOCS:
        documents.append(("python:" + path, (root / "python" / path).read_text(), path,
            f"https://github.com/python/cpython/blob/{PYTHON_PIN}/{path}", "PSF-2.0", "python_3.14_docs"))
    for book, (title, author) in BOOKS.items():
        documents.append((f"gutenberg:{book}", strip_gutenberg((root / f"books/{book}.txt").read_text()),
            f"{title} — {author}", f"https://www.gutenberg.org/ebooks/{book}",
            "Public-domain text; Project Gutenberg terms retained in source snapshot", "public_domain_book"))
    for task, text, title, url, license_id, kind in documents:
        if len(text) <= 64000:
            raise ValueError(f"Long source is too short for three distinct windows: {title}")
        for label, budget in [("32k_chars", 32000), ("64k_chars", 64000), ("128k_chars", 128000)]:
            length = min(budget, len(text))
            start = 0 if label == "32k_chars" else (len(text) - length) // 2 if label == "64k_chars" else len(text) - length
            excerpt = text[start:start + length]
            content = (f"Read this excerpt from {title}. Summarize its main points and important details, "
                       "using only information in the excerpt.\n\n<document>\n" + excerpt + "\n</document>\n\nSummary:")
            rows.append(_prompt(task + ":" + label, task, "long_document", content, url, license_id,
                document_kind=kind, length_band=label, requested_chars=budget, actual_excerpt_chars=length,
                excerpt_start_char=start, context_shortened=True, source_title=title,
                source_commit=PYTHON_PIN if kind == "python_3.14_docs" else None))

    set_splits(rows, set(), seed)
    validate_inputs(rows)
    rows.sort(key=lambda r: (r["domain"], r["id"]))
    write_jsonl(out / "quick240.jsonl", rows)
    write_jsonl(out / "holdout.jsonl", [r for r in rows if r["split"] == "holdout"])
    manifest = {"schema": 1, "profile": "public-multidomain-v2", "count": len(rows),
        "seed": seed, "dataset_sha256": digest(rows), "tasks": len({r["task"] for r in rows}),
        "domain_counts": dict(collections.Counter(r["domain"] for r in rows)),
        "split_counts": dict(collections.Counter(r["split"] for r in rows)),
        "terminal_version": args.terminal_version,
        "source_pins": {"terminal_bench": TERMINAL_PINS, "openmathreasoning": MATH_PIN,
                        "global_mmlu_lite": MMLU_PIN, "cpython": PYTHON_PIN},
        "suggested_max_prompt_tokens": 65536, "calibration_disjoint": "unverified",
        "selection": "fixed quota, strata and seed; does not use baseline/candidate model outcomes",
        "math_note": "OpenMathReasoning (2025) substitutes for NC-licensed MathArena; not MathArena scores",
        "harbor_note": "Framework/source registry; not an independent dataset",
        "long_note": "Character bands are not token counts; audit actual rendered tokens before evaluation"}
    write_json(out / "manifest.json", manifest)
    write_json(out / "source-downloads.json", snapshot)
    licenses = out / "licenses"
    licenses.mkdir(parents=True, exist_ok=True)
    for source, name in [(root / f"terminal-{args.terminal_version}/LICENSE", "Apache-2.0.txt"),
                         (root / "python/LICENSE", "Python-LICENSE.txt"),
                         (root / "licenses/CC-BY-4.0.txt", "CC-BY-4.0.txt")]:
        shutil.copyfile(source, licenses / name)
    for source, name in [(root / "math/README.md", "OpenMathReasoning-card.md"),
                         (root / "mmlu/README.md", "Global-MMLU-Lite-card.md")]:
        shutil.copyfile(source, licenses / name)
    for record in snapshot["files"]:
        relative = record["path"]
        if re.fullmatch(rf"terminal-{re.escape(args.terminal_version)}/tasks/[^/]+/(?:LICENSE(?:\.md|\.txt)?|NOTICE(?:\.md|\.txt)?)", relative):
            source = root / relative
            shutil.copyfile(source, licenses / f"terminal-{args.terminal_version}-{source.parent.name}-{source.name}")
    full = (root / "books/1342.txt").read_text()
    marker = re.search(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", full, re.I)
    (licenses / "Project-Gutenberg-terms.txt").write_text(full[marker.end():])
    print(json.dumps(manifest, indent=2))
    return 0


def audit(args):
    from .api import Server
    rows = read_jsonl(args.dataset)
    server = Server(args.base_url, args.model, args.api_key_env, args.timeout)
    server.discover()
    def count(row):
        n = len(server.tokenize(row))
        return {"id": row["id"], "task": row["task"], "domain": row["domain"],
                "tokens": n, "over_budget": n > args.max_prompt_tokens}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        measured = list(pool.map(count, rows))
    write_jsonl(Path(args.output).with_suffix(".samples.jsonl"), measured)
    report = {"model": server.model, "dataset_sha256": digest(rows), "count": len(rows),
        "max_tokens": max(r["tokens"] for r in measured),
        "over_budget": sum(r["over_budget"] for r in measured), "by_domain": {}}
    for domain in sorted({r["domain"] for r in measured}):
        values = sorted(r["tokens"] for r in measured if r["domain"] == domain)
        report["by_domain"][domain] = {"count": len(values), "min": min(values),
            "median": values[len(values)//2], "max": max(values)}
    write_json(args.output, report)
    print(json.dumps(report, indent=2))
    return 0 if not report["over_budget"] else 2
