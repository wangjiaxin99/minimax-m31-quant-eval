"""Experimental fixed-reference sparse fidelity client (requires GPU-tail patch).

Reference: normalized full saved MXFP8 probabilities; fixed Top-2000 (or an explicitly configured fixed K).
Candidate: raw sparse values plus explicitly exported GPU normalization mass.
Top1 uses raw argmax, Exact32 uses an independent greedy free continuation.
The existing full-vocabulary CLI is unchanged.
"""
import argparse
import concurrent.futures
import gzip
import hashlib
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from .api import Server
from .common import digest, json_compatible, write_json
from .metrics import agreement


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def load_sample(reference, sample_id):
    with gzip.open(reference/'samples'/(digest(sample_id)+'.json.gz'), 'rt') as f:
        return json.load(f)


def lse(values):
    if not len(values):
        return -math.inf
    maximum = float(np.max(values))
    if maximum == -math.inf:
        return maximum
    return maximum+math.log(float(np.exp(values-maximum).sum()))


def top_ids(logprobs, k):
    """Stable reference Top-K: equal-probability tokens ordered by token ID."""
    if not isinstance(k, int) or not 1 <= k <= len(logprobs):
        raise ValueError("K must be between 1 and vocabulary size")
    if k == len(logprobs):
        return np.argsort(-logprobs, kind='stable')
    initial = np.argpartition(-logprobs, k-1)[:k]
    boundary = float(logprobs[initial].min())
    larger = np.flatnonzero(logprobs > boundary)
    tied = np.flatnonzero(logprobs == boundary)[:k-len(larger)]
    chosen = np.concatenate((larger, tied))
    assert len(chosen) == k
    return chosen[np.lexsort((chosen, -logprobs[chosen]))]


def prepare(args):
    reference, out = Path(args.reference), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    meta = json.loads((reference/'run.json').read_text())
    start = time.monotonic()
    records = []
    total_bytes = 0
    for row in meta['dataset']:
        key = digest(row['id'])
        record = load_sample(reference, row['id'])
        assert len(record['generation']['token_ids']) == 32
        layout = record['probability_arrays']
        assert layout['shape'] == [32,200064]
        full = np.memmap(reference/layout['logprobs'], dtype='<f8', mode='r', shape=(32,200064))
        ids_all, values_all, tails, norms, offsets, top1 = [], [], [], [], [0], []
        for pos in range(32):
            lp = full[pos]
            z = lse(lp)
            assert abs(math.expm1(z)) <= 2e-5
            k = args.top_k
            ids = top_ids(lp, k)
            selected = lp[ids]-z
            mask = np.ones(len(lp), dtype=bool)
            mask[ids] = False
            tail = lse(lp[mask])-z
            assert abs(math.expm1(lse(np.r_[selected,tail]))) < 1e-10
            ids_all.extend(ids.tolist())
            values_all.extend(selected.tolist())
            tails.append(tail);norms.append(z);top1.append(int(np.argmax(lp)))
            offsets.append(len(ids_all))
        path = out/(key+'.npz')
        np.savez(path,token_ids=np.array(ids_all,dtype='<u4'),logprobs=np.array(values_all,dtype='<f8'),
                 tail=np.array(tails,dtype='<f8'),normalization=np.array(norms,dtype='<f8'),
                 offsets=np.array(offsets,dtype='<i8'),top1=np.array(top1,dtype='<u4'))
        records.append({'id':row['id'],'domain':row['domain'],'file':path.name,
                        'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'k':np.diff(offsets).tolist()})
        total_bytes += path.stat().st_size
        print(f'prepare {len(records)}/{len(meta["dataset"])} {row["id"]}',flush=True)
    write_json(out/'run.json',{'reference':str(reference.resolve()),'reference_signature':meta['signature'],
        'selection':'fixed_top_k','top_k':args.top_k,'vocab_size':200064,'tokens':32,'records':records,
        'elapsed_seconds':time.monotonic()-start,'payload_bytes':total_bytes,'created':now(),
        'normalization':'reference float64 normalized; stable tail; no position exclusions'})


def request_sparse(server, prefix, ids):
    payload={'model':server.model,'prompt':prefix,'max_tokens':1,'temperature':0.0,
             'top_p':1.0,'top_k':-1,'seed':31,'add_special_tokens':False,'skip_special_tokens':False,
             'return_token_ids':True,'return_tokens_as_token_ids':True,
             'logprobs':len(ids),'logprob_token_ids':ids}
    response=server.request('/completions',payload)
    choice=response['choices'][0]
    extra=choice['logprobs'].get('sparse_fidelity')
    if not extra or len(extra)!=1:
        raise RuntimeError('Server did not return GPU sparse_fidelity; refusing a full-export fallback')
    item=extra[0]
    assert item['backend']=='gpu-float64-logsumexp' and item['schema']==1
    assert item['token_ids']==ids and item['vocab_size']==200064
    q=np.array([float(v) for v in item['logprobs']],dtype='<f8')
    tail=float(item['tail_logprob'])
    correction=float(item['normalization_log_mass'])
    assert not np.isnan(q).any() and not np.isposinf(q).any() and math.isfinite(correction)
    assert abs(math.expm1(correction)) <= 2e-5
    qn=q-correction;tn=tail-correction
    assert abs(math.expm1(lse(np.r_[qn,tn])))<1e-9
    return response,item,qn,tn


def score_json(value):
    """Encode infinite KL explicitly; probability inputs are validated separately."""
    if isinstance(value, dict):
        return {k: score_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [score_json(v) for v in value]
    if isinstance(value, float) and value == math.inf:
        return "Infinity"
    return json_compatible(value)


def measure(p, ptail, q, qtail):
    pp=np.r_[p,ptail];qq=np.r_[q,qtail]
    valid=np.isfinite(pp)
    if np.any(np.isneginf(qq[valid])):
        return math.inf
    value=float(np.sum(np.exp(pp[valid])*(pp[valid]-qq[valid])))
    if value < -1e-10:
        raise ValueError(f'Negative KL beyond tolerance: {value}')
    return max(0.,value)


def evaluate(args):
    prepared=Path(args.prepared_reference)
    prep=json.loads((prepared/'run.json').read_text())
    if prep.get('selection') != 'fixed_top_k':
        raise ValueError('Re-prepare the reference with fixed Top-K; adaptive caches are not accepted')
    if any(k != prep['top_k'] for row in prep['records'] for k in row['k']):
        raise ValueError('Prepared reference contains inconsistent K')
    reference=Path(prep['reference'])
    refmeta=json.loads((reference/'run.json').read_text())
    assert prep['reference_signature']==refmeta['signature']
    rows=refmeta['dataset']
    if args.limit:
        rows=rows[:args.limit]
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    (out/'probabilities').mkdir()
    server=Server(args.base_url,timeout=args.timeout);server.discover()
    metadata={'created':now(),'reference_signature':prep['reference_signature'],'reference':str(reference),
              'prepared_reference':str(prepared.resolve()),'model':server.model,'base_url':server.base,
              'concurrency':args.concurrency,'expected':len(rows),'tokens':32,'selection':'fixed_top_k',
              'top_k':prep['top_k'],'metric':'stable fixed-reference partition KL; normalized float64',
              'generation':{'temperature':0,'top_p':1,'top_k':-1,'seed':31},
              'top1':'actual raw-logprob argmax; verify greedy sampled token matches it',
              'exact32':'same independent greedy free continuation comparison as existing CLI'}
    write_json(out/'run.json',metadata)
    start=time.monotonic()
    prep_records={r['id']:r for r in prep['records']}

    def worker(row):
        t=time.monotonic();key=digest(row['id'])
        local=Server(args.base_url,model=server.model,timeout=args.timeout)
        cache=load_sample(reference,row['id'])
        result={'id':row['id'],'domain':row['domain'],'task':row.get('task',row['id']),
                'scored_positions':[],'reference_token_ids':cache['generation']['token_ids']}
        dest=out/'probabilities'/key;dest.mkdir()
        try:
            assert local.tokenize(row)==cache['prompt_tokens'],'Tokenizer/prompt mismatch'
            assert local.detokenize(cache['generation']['token_ids'])==cache['continuation_text']
            tfree=time.monotonic()
            gen=local.generate(cache['prompt_tokens'],32)
            result['free_generation_seconds']=time.monotonic()-tfree
            result['candidate_generation']=gen
            result.update(agreement(cache['generation']['token_ids'],gen['token_ids']))
            assert len(cache['generation']['token_ids'])==32
            data_path=prepared/(key+'.npz')
            assert hashlib.sha256(data_path.read_bytes()).hexdigest()==prep_records[row['id']]['sha256']
            data=np.load(data_path)
            with (dest/'positions.jsonl').open('w') as logfile:
                for pos in range(32):
                    lo,hi=data['offsets'][pos:pos+2]
                    ids=data['token_ids'][lo:hi].tolist()
                    p=data['logprobs'][lo:hi];pt=float(data['tail'][pos])
                    prefix=cache['prompt_tokens']+cache['generation']['token_ids'][:pos]
                    tq=time.monotonic()
                    response,sparse,q,qt=request_sparse(local,prefix,ids)
                    elapsed=time.monotonic()-tq
                    sampled=response['choices'][0]['token_ids'][0]
                    assert sampled==sparse['top1_token_id'],'Greedy sampled ID differs from raw Top1; inspect constraints'
                    score=measure(p,pt,q,qt)
                    values=np.r_[q,qt].astype('<f8')
                    values.tofile(dest/f'{pos:02d}.logprobs.f64')
                    record={'position':pos+1,'k':len(ids),'kl':score,'prefix_sha256':digest(prefix),
                            'selected_ids_sha256':digest(ids),'reference_top1':int(data['top1'][pos]),
                            'candidate_top1':sparse['top1_token_id'],
                            'top1_match':int(data['top1'][pos])==sparse['top1_token_id'],
                            'candidate_top1_logprob':sparse['top1_logprob'],
                            'normalization_log_mass':sparse['normalization_log_mass'],
                            'candidate_tail_logprob':qt,'reference_tail_logprob':pt,
                            'request_seconds':elapsed,'binary_sha256':hashlib.sha256(values.tobytes()).hexdigest(),
                            'binary_format':'little-endian float64 normalized selected logprobs followed by tail',
                            'usage':response.get('usage'), 'candidate_text':response['choices'][0].get('text','')}
                    logfile.write(json.dumps(score_json(record),ensure_ascii=False)+'\n');logfile.flush()
                    result['scored_positions'].append(record)
            result['mean_kl']=float(np.mean([p['kl'] for p in result['scored_positions']]))
            result['top1_matches']=sum(p['top1_match'] for p in result['scored_positions'])
        except Exception as exc:
            result['error']=repr(exc)
        result['elapsed_seconds']=time.monotonic()-t
        return result

    completed=[]
    with (out/'samples.jsonl').open('w') as log, concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures={pool.submit(worker,row):row for row in rows}
        for future in concurrent.futures.as_completed(futures):
            result=future.result();completed.append(result)
            log.write(json.dumps(score_json(result),ensure_ascii=False)+'\n');log.flush()
            count=sum(len(r['scored_positions']) for r in completed)
            state={'updated':now(),'completed_prompts':len(completed),'expected_prompts':len(rows),
                   'scored_positions':count,'errors':sum('error'in r for r in completed),
                   'elapsed_seconds':time.monotonic()-start}
            write_json(out/'progress.json',state)
            print(f'evaluate {len(completed)}/{len(rows)} {result["id"]} '+('ERROR '+result['error'] if 'error'in result else f'OK KL={result["mean_kl"]:.8f} Top1={result["top1_matches"]}/32 Exact32={result["sequence_exact"]}'),flush=True)
    ok=[r for r in completed if 'error'not in r]
    scored=[p for r in ok for p in r['scored_positions']]
    def group(rs):
        ps=[p for r in rs for p in r['scored_positions']]
        return {'prompts':len(rs),'positions':len(ps),'mean_kl':float(np.mean([p['kl'] for p in ps])) if ps else None,
                'top1_matches':sum(p['top1_match'] for p in ps),
                'same_prefix_top1_rate':sum(p['top1_match'] for p in ps)/len(ps) if ps else None,
                'exact32_matches':sum(r['sequence_exact'] for r in rs),
                'exact32_rate':sum(r['sequence_exact'] for r in rs)/len(rs) if rs else None}
    summary={**group(ok),'expected':len(rows),'errors':len(rows)-len(ok),
             'status':'complete' if len(ok)==len(rows) and len(scored)==len(rows)*32 else 'incomplete',
             'elapsed_seconds':time.monotonic()-start,'finished':now(),
             'candidate_probability_bytes':sum(p.stat().st_size for p in (out/'probabilities').rglob('*.f64')),
             'by_domain':{d:group([r for r in ok if r['domain']==d]) for d in sorted({r['domain'] for r in ok})}}
    write_json(out/'summary.json',score_json(summary))
    print(json.dumps(score_json(summary),ensure_ascii=False,indent=2),flush=True)
    if summary['status']!='complete':
        raise RuntimeError('Incomplete evaluation; inspect per-prompt errors, do not report partial metrics as final')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--reference',required=True)
    p.add_argument('--output',required=True);p.add_argument('--top-k',type=int,default=2000)
    p=sub.add_parser('evaluate');p.add_argument('--prepared-reference',required=True);p.add_argument('--base-url',required=True)
    p.add_argument('--output',required=True);p.add_argument('--concurrency',type=int,default=2);p.add_argument('--timeout',type=float,default=300)
    p.add_argument('--limit',type=int,default=0)
    args=parser.parse_args()
    (prepare if args.command=='prepare' else evaluate)(args)


if __name__=='__main__':
    main()
