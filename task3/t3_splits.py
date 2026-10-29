"""Task 3 splits: per dataset, sort the 1,000 question IDs lexicographically, permute with
numpy.random.default_rng(6304); first 200 = development, remaining 800 = held-out evaluation.
The first 50 MATH evaluation questions in this permutation form the live-generation subset.
Usage: python t3_splits.py --data pa3data --out DIR
"""
import json, argparse
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True)
ap.add_argument('--out', required=True)
args = ap.parse_args()
splits = {}
for name, folder in [('math', 'qwen25-math'), ('mathvista', 'qwen25vl-mathvista')]:
    g = pd.read_parquet(f'{args.data}/{folder}/generations.parquet', columns=['sample_id', 'question_text', 'gold_answer'])
    ids = sorted(g['sample_id'].unique().tolist())
    assert len(ids) == 1000
    perm = np.random.default_rng(6304).permutation(len(ids))
    order = [ids[i] for i in perm]
    splits[name] = dict(dev=order[:200], eval=order[200:])
    if name == 'math':
        first = g.drop_duplicates('sample_id').set_index('sample_id')
        live = [dict(live_index=k, sample_id=s, question_text=first.loc[s, 'question_text'],
                     gold_answer=first.loc[s, 'gold_answer']) for k, s in enumerate(order[200:250])]
        json.dump(live, open(f'{args.out}/live50.json', 'w'), indent=1)
json.dump(dict(seed=6304, rule='sorted IDs permuted with numpy default_rng(6304); first 200 dev, rest eval; '
               'live = first 50 MATH eval questions', **splits), open(f'{args.out}/splits.json', 'w'), indent=1)
print({k: (len(v['dev']), len(v['eval'])) for k, v in splits.items()})
