"""(Superseded by compact.py for the final results.) Print compact Task 1 benchmark table + GSM8K truncation/extraction stats from lm-eval outputs."""
import sys, json, glob, os
from transformers import AutoTokenizer
OUT = sys.argv[1]
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
CONDS = ['baseline', 'full_mid', 'full', 'l2sp', 'lora4', 'lora16', 'oplora']
table = {}
for c in CONDS:
    row = {}
    for sub in ['gsm8k', 'll']:
        fs = sorted(glob.glob(f'{OUT}/{c}/{sub}/**/results_*.json', recursive=True))
        if not fs:
            continue
        r = json.load(open(fs[-1]))['results']
        for task, m in r.items():
            for k, v in m.items():
                if isinstance(v, (int, float)) and 'stderr' not in k:
                    row[f'{task}|{k}'] = v
    fs = sorted(glob.glob(f'{OUT}/{c}/gsm8k/**/samples_gsm8k_cot_*.jsonl', recursive=True))
    if fs:
        n = trunc = noext = 0; lens = []
        for line in open(fs[-1]):
            s = json.loads(line)
            if s.get('filter', 'flexible-extract') != 'flexible-extract':
                continue
            n += 1
            resp = s['resps'][0][0] if isinstance(s['resps'][0], list) else s['resps'][0]
            L = len(tok(resp, add_special_tokens=False).input_ids); lens.append(L)
            trunc += L >= 1023
            fr = s['filtered_resps']
            noext += any(x in ('[invalid]', '') for x in (fr if isinstance(fr, list) else [fr]))
        row.update(gsm8k_n=n, gsm8k_trunc_rate=trunc / n, gsm8k_any_filter_invalid=noext / n,
                   gsm8k_mean_resp_tokens=sum(lens) / n)
    if os.path.exists(f'{OUT}/{c}/eval_time.json'):
        row.update(json.load(open(f'{OUT}/{c}/eval_time.json')))
    table[c] = row
json.dump(table, open(f'{OUT}/benchmark_table.json', 'w'), indent=1)
print('BENCHMARK_TABLE_JSON', json.dumps(table))
