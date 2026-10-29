"""Collect Task 1 machine-readable results into the repo folder results/task1/.
Usage: python compact.py --train DIR --gsm DIR --ll DIR --qual DIR --diag FILE --out OUT
  --train : training notebook output (contains runs/, data/, logs/)
  --gsm   : eval output with <cond>/gsm8k/...samples_gsm8k_cot_*.jsonl and results_*.json
  --ll    : eval output with <cond>/...samples_{hellaswag,boolq,arc_easy}_*.jsonl and results_*.json
"""
import os, json, glob, csv, argparse, shutil
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
for a in ['train', 'gsm', 'll', 'qual', 'diag', 'out']:
    ap.add_argument('--' + a, required=True)
args = ap.parse_args()
R = f'{args.out}/results/task1'
for d in ['training', 'data', 'benchmarks/gsm8k_per_example', 'benchmarks/likelihood_per_example', 'benchmarks/aggregate',
          'diagnostics', 'qualitative']:
    os.makedirs(f'{R}/{d}', exist_ok=True)
METHODS = ['full', 'l2sp', 'lora4', 'lora16', 'oplora']
CONDS = ['baseline', 'full_mid', 'full', 'l2sp', 'lora4', 'lora16', 'oplora']
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')

# ---- training logs and summaries
for m in METHODS:
    shutil.copy(f'{args.train}/runs/{m}/summary.json', f'{R}/training/{m}_summary.json')
    cols = ['update', 'examples', 'sft_loss', 'lr', 'grad_norm', 'scale', 'skipped', 'resp_tokens', 'step_time',
            'omega', 'lam_omega', 'total_loss']
    with open(f'{R}/training/{m}_train_log.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(cols)
        for line in open(f'{args.train}/runs/{m}/train_log.jsonl'):
            r = json.loads(line)
            w.writerow([('%.6g' % r[c]) if isinstance(r.get(c), float) else r.get(c, '') for c in cols])

# ---- data selection
for fn in ['train_info.json', 'probe_ids.json']:
    shutil.copy(f'{args.train}/data/{fn}', f'{R}/data/{fn}')
ids = json.load(open(f'{args.train}/data/train_ids.json'))
src = ids['src_idx']
json.dump(dict(rule='first 40,000 eligible MetaMathQA rows in source order (meta-math/MetaMathQA train split)',
               n=len(src), first_source_index=src[0], last_source_index=src[-1],
               excluded_before_last=ids['excluded'],
               reconstruct='[i for i in range(last_source_index+1) if i not in excluded indices]'),
          open(f'{R}/data/train_selection.json', 'w'), indent=1)
assert [i for i in range(src[-1] + 1) if i not in {e[0] for e in ids['excluded']}] == src

# ---- benchmarks
agg = {}
for c in CONDS:
    agg[c] = {}
    for sub, base in [('gsm8k', f'{args.gsm}/{c}/gsm8k'), ('ll', f'{args.ll}/{c}')]:
        fs = sorted(glob.glob(f'{base}/**/results_*.json', recursive=True))
        assert fs, (c, sub)
        res = json.load(open(fs[-1]))
        shutil.copy(fs[-1], f'{R}/benchmarks/aggregate/{c}_{sub}_results.json')
        for task, mm in res['results'].items():
            for k, v in mm.items():
                if isinstance(v, (int, float)):
                    agg[c][f'{task}|{k}'] = v
    # GSM8K per-example (flexible-extract rows)
    f = sorted(glob.glob(f'{args.gsm}/{c}/gsm8k/**/samples_gsm8k_cot_*.jsonl', recursive=True))[-1]
    n = tr = inv = 0
    with open(f'{R}/benchmarks/gsm8k_per_example/{c}.csv', 'w', newline='') as fo:
        w = csv.writer(fo); w.writerow(['doc_id', 'target', 'extracted', 'correct', 'resp_tokens', 'truncated'])
        for line in open(f):
            s = json.loads(line)
            if s.get('filter', 'flexible-extract') != 'flexible-extract':
                continue
            resp = s['resps'][0][0] if isinstance(s['resps'][0], list) else s['resps'][0]
            fr = s['filtered_resps']; fr = fr[0] if isinstance(fr, list) else fr
            L = len(tok(resp, add_special_tokens=False).input_ids)
            t = L >= 1020
            w.writerow([s['doc_id'], s['target'], fr, int(s['exact_match']), L, int(t)])
            n += 1; tr += t; inv += fr == '[invalid]'
    agg[c].update(gsm8k_n=n, gsm8k_truncated_frac=tr / n, gsm8k_unextractable_frac=inv / n)
    # likelihood tasks per-example
    for task in ['hellaswag', 'boolq', 'arc_easy']:
        f = sorted(glob.glob(f'{args.ll}/{c}/**/samples_{task}_*.jsonl', recursive=True))[-1]
        with open(f'{R}/benchmarks/likelihood_per_example/{c}_{task}.csv', 'w', newline='') as fo:
            w = csv.writer(fo); hdr = ['doc_id', 'acc'] + (['acc_norm'] if task != 'boolq' else []); w.writerow(hdr)
            for line in open(f):
                s = json.loads(line)
                w.writerow([s['doc_id'], int(s['acc'])] + ([int(s['acc_norm'])] if task != 'boolq' else []))
json.dump(agg, open(f'{R}/benchmarks/benchmark_table.json', 'w'), indent=1)

# ---- diagnostics + qualitative
shutil.copy(args.diag, f'{R}/diagnostics/diag_results.json')
for fn in ['qualitative_responses.jsonl', 'annotation_sheet_42.csv']:
    shutil.copy(f'{args.qual}/{fn}', f'{R}/qualitative/{fn}')

print('AGG', json.dumps({c: {k: round(v, 4) for k, v in d.items() if 'stderr' not in k} for c, d in agg.items()}))
