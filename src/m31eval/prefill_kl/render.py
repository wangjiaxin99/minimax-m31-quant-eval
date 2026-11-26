"""Render chat prompts into the ``prompt_token_ids`` rows that ``gen`` reads.

Input rows are the fidelity dataset's ``{id, domain, messages, tools?}`` (e.g. ``data/mixed/quick300.jsonl``).
Each conversation goes through the model's chat template with the generation prompt appended; the template's own
defaults apply (Qwen3.6: thinking on) unless ``chat_template_kwargs`` overrides them. Tool-call arguments stored as
JSON strings are parsed into objects first, as the Qwen templates expect.
"""
from __future__ import annotations

import json
from collections import Counter


def normalize(messages: list[dict]) -> list[dict]:
    """Copy of ``messages`` with every tool call's string ``arguments`` parsed as JSON when possible."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get('tool_calls'):
            calls = []
            for c in m['tool_calls']:
                c = json.loads(json.dumps(c))
                args = c.get('function', {}).get('arguments')
                if isinstance(args, str):
                    try:
                        c['function']['arguments'] = json.loads(args)
                    except json.JSONDecodeError:
                        pass
                calls.append(c)
            m['tool_calls'] = calls
        out.append(m)
    return out


def render_rows(tokenizer, rows: list[dict], chat_template_kwargs: dict | None = None) -> list[dict]:
    """``{id, domain, n_prompt_tokens, prompt_token_ids}`` for each input row."""
    out = []
    for r in rows:
        kwargs = dict(chat_template_kwargs or {})
        if r.get('tools'):
            kwargs['tools'] = r['tools']
        ids = tokenizer.apply_chat_template(normalize(r['messages']), add_generation_prompt=True, tokenize=True,
                                            **kwargs)
        if not isinstance(ids, list):  # transformers may return a BatchEncoding
            ids = ids['input_ids']
        ids = [int(t) for t in ids]
        out.append({'id': r['id'], 'domain': r['domain'], 'n_prompt_tokens': len(ids), 'prompt_token_ids': ids})
    return out


def cmd_render(args) -> None:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    with open(args.src) as f:
        rows = [json.loads(line) for line in f]
    out = render_rows(tokenizer, rows, json.loads(args.chat_template_kwargs or '{}'))
    with open(args.out, 'w') as f:
        for r in out:
            f.write(json.dumps(r) + '\n')
    longest = Counter()
    for r in out:
        longest[r['domain']] = max(longest[r['domain']], r['n_prompt_tokens'])
    print(json.dumps({'rows': len(out), 'max_prompt_tokens_per_domain': dict(longest)}))
