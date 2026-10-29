"""Grade the live-generation comparison (Step 4). Extraction: last literal \\boxed{ with balanced-brace scanning
(absent/empty/unclosed -> NO_ANSWER). Grading and answer grouping use math-verify, as for the offline MATH data.
Usage: python t3_live_eval.py --live DIR --out DIR
"""
import json, glob, argparse, collections
import numpy as np
from math_verify import parse, verify

ap = argparse.ArgumentParser()
ap.add_argument('--live', required=True)
ap.add_argument('--out', required=True)
args = ap.parse_args()
NO = 'NO_ANSWER'


def last_boxed(text):
    i = text.rfind('\\boxed{')
    if i < 0:
        return None
    j = i + len('\\boxed{'); depth = 1; k = j
    while k < len(text) and depth:
        depth += {'{': 1, '}': -1}.get(text[k], 0); k += 1
    if depth:
        return None
    s = text[j:k - 1].strip()
    return s or None


def eq(a, b):
    if a == b:
        return True
    try:
        return bool(verify(parse('\\boxed{' + a + '}'), parse('\\boxed{' + b + '}')))
    except Exception:
        return False


def sc(answers):
    groups = []  # [rep, count, first]
    for pos, a in enumerate(answers):
        if a is None:
            a = NO
        for g in groups:
            if (a == NO and g[0] == NO) or (a != NO and g[0] != NO and eq(g[0], a)):
                g[1] += 1; break
        else:
            groups.append([a, 1, pos])
    best = min(groups, key=lambda g: (-g[1], g[2]))
    return best[0]


rows = []
for f in sorted(glob.glob(f'{args.live}/live_shard*.jsonl')):
    rows += [json.loads(l) for l in open(f)]
rows.sort(key=lambda r: r['live_index'])
per_q = []
for r in rows:
    gold = r['gold_answer']
    s_ans = last_boxed(r['single']['response'])
    p_ans = [last_boxed(p['response']) for p in r['parallel']]
    par_sel = sc(p_ans)
    init_ans = p_ans[0]
    rev_ans = last_boxed(r['revision']['response'])
    ok = lambda a: a is not None and a != NO and eq(gold, a)
    rec = dict(sample_id=r['sample_id'], live_index=r['live_index'], gold=gold,
               single=dict(answer=s_ans, correct=ok(s_ans), out_tokens=r['single']['output_tokens'],
                           in_tokens=r['single']['input_tokens'], seconds=r['single']['seconds'],
                           truncated=r['single']['truncated'], extracted=s_ans is not None),
               parallel=dict(answers=p_ans, selected=par_sel, correct=ok(par_sel),
                             any_correct=any(ok(a) for a in p_ans),
                             out_tokens=sum(p['output_tokens'] for p in r['parallel']),
                             in_tokens=sum(p['input_tokens'] for p in r['parallel']),
                             seconds=sum(p['seconds'] for p in r['parallel']),
                             truncated_frac=np.mean([p['truncated'] for p in r['parallel']]),
                             extracted_frac=np.mean([a is not None for a in p_ans])),
               refinement=dict(initial=init_ans, initial_correct=ok(init_ans), revised=rev_ans, correct=ok(rev_ans),
                               # the shared initial response is charged to refinement as well
                               out_tokens=r['parallel'][0]['output_tokens'] + r['revision']['output_tokens'],
                               in_tokens=r['parallel'][0]['input_tokens'] + r['revision']['input_tokens'],
                               critique_out_tokens=r['revision']['output_tokens'],
                               seconds=r['parallel'][0]['seconds'] + r['revision']['seconds'],
                               truncated_initial=r['parallel'][0]['truncated'], truncated_revision=r['revision']['truncated'],
                               extracted=rev_ans is not None))
    per_q.append(rec)
summ = {}
for s in ['single', 'parallel', 'refinement']:
    d = [q[s] for q in per_q]
    summ[s] = dict(n=len(d), accuracy=float(np.mean([x['correct'] for x in d])),
                   mean_output_tokens=float(np.mean([x['out_tokens'] for x in d])),
                   mean_input_tokens=float(np.mean([x['in_tokens'] for x in d])),
                   mean_seconds=float(np.mean([x['seconds'] for x in d])))
summ['single'].update(truncation_rate=float(np.mean([q['single']['truncated'] for q in per_q])),
                      extraction_rate=float(np.mean([q['single']['extracted'] for q in per_q])))
summ['parallel'].update(truncation_rate=float(np.mean([q['parallel']['truncated_frac'] for q in per_q])),
                        extraction_rate=float(np.mean([q['parallel']['extracted_frac'] for q in per_q])),
                        oracle_any_of_3=float(np.mean([q['parallel']['any_correct'] for q in per_q])))
ref = [q['refinement'] for q in per_q]
summ['refinement'].update(
    initial_accuracy=float(np.mean([x['initial_correct'] for x in ref])),
    wrong_to_correct=int(sum((not x['initial_correct']) and x['correct'] for x in ref)),
    correct_to_wrong=int(sum(x['initial_correct'] and not x['correct'] for x in ref)),
    unchanged_correct=int(sum(x['initial_correct'] and x['correct'] for x in ref)),
    unchanged_wrong=int(sum((not x['initial_correct']) and not x['correct'] for x in ref)),
    truncation_rate_initial=float(np.mean([x['truncated_initial'] for x in ref])),
    truncation_rate_revision=float(np.mean([x['truncated_revision'] for x in ref])),
    extraction_rate=float(np.mean([x['extracted'] for x in ref])),
    mean_critique_tokens=float(np.mean([x['critique_out_tokens'] for x in ref])))
summ['peak_gpu_mem_bytes'] = max(r['peak_gpu_mem_bytes'] for r in rows)
summ['gpu'] = rows[0]['gpu'] if rows else None
json.dump(dict(summary=summ, per_question=per_q), open(f'{args.out}/live_results.json', 'w'), indent=1, default=float)
print(json.dumps(summ, indent=1, default=float))
