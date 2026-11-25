"""On-disk layouts.

answers.jsonl   one generated reference answer per line:
                {key, id, domain, draw, seed, prompt_len, finish_reason, prompt_token_ids, answer_token_ids}
reference/      manifest.json + seqs/<n>.npz per answer: topv f32 [n, K], topi i32 [n, K], log_rest f32 [n],
                tgt f32 [n] (the reference's log-probability of the answer token), where n = answer length and
                row j predicts answer token j.
score run       manifest.json + seqs/<n>.npz per answer: kl_<K> f64 [n] for each K, flip bool [n] (top-1 differs
                from the reference's top-1), tgt f32 [n] (the candidate's log-probability of the answer token).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def read_answers(path) -> list[dict]:
    return [json.loads(line) for line in open(path)]


def seq_file(root: Path, index: int) -> Path:
    return Path(root) / 'seqs' / f'{index:05d}.npz'


def write_manifest(root: Path, data: dict) -> None:
    Path(root).mkdir(parents=True, exist_ok=True)
    (Path(root) / 'manifest.json').write_text(json.dumps(data, indent=1))


def read_manifest(root: Path) -> dict:
    return json.loads((Path(root) / 'manifest.json').read_text())


def save_seq(root: Path, index: int, arrays: dict[str, np.ndarray]) -> None:
    path = seq_file(root, index)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.npz')
    np.savez(tmp, **arrays)
    tmp.replace(path)


def load_seq(root: Path, index: int) -> dict[str, np.ndarray]:
    with np.load(seq_file(root, index)) as z:
        return {k: z[k] for k in z.files}


def answers_digest(answers: list[dict]) -> str:
    h = hashlib.sha256()
    for a in answers:
        h.update(a['key'].encode())
        h.update(np.asarray(a['prompt_token_ids'] + a['answer_token_ids'], dtype='<i4').tobytes())
    return h.hexdigest()
