import hashlib
import json
import math
import os
import time
from pathlib import Path


def json_compatible(value):
    """Keep zero-probability logprobs explicit without nonstandard JSON numbers."""
    if isinstance(value, dict):
        return {k: json_compatible(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        if value == -math.inf:
            return "-Infinity"
        raise ValueError("NaN/positive infinity cannot be stored as valid probability data")
    return value


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line.lstrip("\ufeff\u200b")))
                except ValueError as exc:
                    raise ValueError(f"{path}:{number}: {exc}") from exc
    return rows


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(json_compatible(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(json_compatible(row), ensure_ascii=False, allow_nan=False) + "\n")


def new_output(path, metadata):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    write_json(path / "run.json", {**metadata, "started_unix": time.time()})
    return path


def public_url(url):
    from urllib.parse import urlsplit, urlunsplit
    p = urlsplit(url)
    return urlunsplit((p.scheme, p.hostname + (f":{p.port}" if p.port else ""), p.path, "", ""))
