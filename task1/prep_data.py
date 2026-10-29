"""Task 1 data preparation: select first 40,000 eligible MetaMathQA examples (source order),
pre-tokenize with the Qwen3 chat template (thinking disabled), and fix the activation-probe set."""
import os, re, json, random, argparse
from collections import Counter
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument('--out', default='/kaggle/working/data')
ap.add_argument('--n', type=int, default=40000)
ap.add_argument('--max_len', type=int, default=2048)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
SEED = 6304
MODEL = 'Qwen/Qwen3-0.6B'


def norm(s):
    s = (s or '').lower()
    s = re.sub(r'[^a-z0-9]+', ' ', s)
    return ' '.join(s.split())


tok = AutoTokenizer.from_pretrained(MODEL)
gsm = load_dataset('openai/gsm8k', 'main', split='test')
gsm_test = set(norm(q) for q in gsm['question'])
ds = load_dataset('meta-math/MetaMathQA', split='train')
print('MetaMathQA rows:', len(ds), 'cols:', ds.column_names)

stats = Counter()
ids_all, plens, src_idx, types = [], [], [], []
excluded = []
B = 1000
i0 = 0
while len(ids_all) < args.n and i0 < len(ds):
    chunk = ds[i0:i0 + B]
    rows = [dict(zip(chunk.keys(), v)) for v in zip(*chunk.values())]
    cand, cidx = [], []
    for j, ex in enumerate(rows):
        i = i0 + j
        q, r = ex.get('query'), ex.get('response')
        if not q or not q.strip() or not r or not r.strip():
            stats['empty'] += 1; excluded.append((i, 'empty')); continue
        if norm(q) in gsm_test or norm(ex.get('original_question')) in gsm_test:
            stats['gsm8k_test_match'] += 1; excluded.append((i, 'gsm8k_test_match')); continue
        cand.append(ex); cidx.append(i)
    prompts = [tok.apply_chat_template([{'role': 'user', 'content': ex['query']}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False) for ex in cand]
    p_ids = tok(prompts, add_special_tokens=False)['input_ids']
    r_ids = tok([ex['response'] + '<|im_end|>' for ex in cand], add_special_tokens=False)['input_ids']
    for ex, i, p, r in zip(cand, cidx, p_ids, r_ids):
        if len(ids_all) >= args.n:
            break
        if len(p) + len(r) > args.max_len:
            stats['too_long'] += 1; excluded.append((i, 'too_long')); continue
        ids_all.append(p + r); plens.append(len(p)); src_idx.append(i); types.append(ex.get('type'))
    i0 += B
    stats['scanned'] = min(i0, len(ds))

assert len(ids_all) == args.n, len(ids_all)
last_src = src_idx[-1]
excluded = [e for e in excluded if e[0] <= last_src]
stats = Counter({k: sum(1 for e in excluded if e[1] == k) for k in ['empty', 'gsm8k_test_match', 'too_long']})
lens = torch.tensor([len(x) for x in ids_all]); rl = lens - torch.tensor(plens)
info = dict(model=MODEL, n=len(ids_all), last_source_index=last_src, excluded=dict(stats),
            len_mean=lens.float().mean().item(), len_max=lens.max().item(), len_p50=lens.median().item(),
            resp_tokens_total=int(rl.sum()), tokens_total=int(lens.sum()),
            types=dict(Counter(types)))
print(json.dumps(info, indent=1))
torch.save(dict(ids=ids_all, plen=plens, src_idx=src_idx), f'{args.out}/train_40k.pt')
json.dump(info, open(f'{args.out}/train_info.json', 'w'), indent=1)
json.dump(dict(src_idx=src_idx, excluded=excluded), open(f'{args.out}/train_ids.json', 'w'))

# Activation-probe set (fixed before looking at any model results)
rng = random.Random(SEED)
gsm_probe = sorted(rng.sample(range(len(gsm)), 32))
n_hs = 10042  # HellaSwag validation size (lm-eval split)
hs_probe = sorted(rng.sample(range(n_hs), 32))
json.dump(dict(seed=SEED, gsm8k_test=gsm_probe, hellaswag_validation=hs_probe, n_hellaswag=n_hs),
          open(f'{args.out}/probe_ids.json', 'w'), indent=1)
print('probe set saved')
