"""Task 1 qualitative generation: 24 prompts from first_plot_questions.yaml x 7 evaluation conditions = 168 responses.
Greedy decoding, thinking disabled, 512 new tokens, FP16. Also computes simple automatic checks and writes the
42-response annotation sheet (first and last plain questions x 3 formats x 7 conditions) with empty manual columns.
Usage: python qual_gen.py --runs RUNS_DIR --yaml first_plot_questions.yaml --out OUT_DIR
"""
import os, re, json, csv, argparse, time
import yaml, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument('--runs', required=True)
ap.add_argument('--yaml', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--max_new', type=int, default=512)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
MODEL = 'Qwen/Qwen3-0.6B'
CONDS = [('baseline', MODEL), ('full_mid', f'{args.runs}/full/model_mid'), ('full', f'{args.runs}/full/model_final'),
         ('l2sp', f'{args.runs}/l2sp/model_final'), ('lora4', f'{args.runs}/lora4/model_final'),
         ('lora16', f'{args.runs}/lora16/model_final'), ('oplora', f'{args.runs}/oplora/model_final')]

Q = yaml.safe_load(open(args.yaml))
assert len(Q) == 24, len(Q)
items = []
for q in Q:
    qid = q['id']
    fmt = 'json' if qid.endswith('_json') else 'template' if qid.endswith('_template') else 'plain'
    base = re.sub(r'_(json|template)$', '', qid)
    msgs = ([{'role': 'system', 'content': q['system']}] if q.get('system') else []) + \
           [{'role': 'user', 'content': q['paraphrases'][0]}]
    items.append(dict(id=qid, base_question=base, format=fmt, messages=msgs))
plain_order = [it['base_question'] for it in items if it['format'] == 'plain']
for it in items:
    it['plain_index'] = plain_order.index(it['base_question'])
ANNOT = {plain_order[0], plain_order[-1]}  # first and last underlying questions in plain order

tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = 'left'
prompts = [tok.apply_chat_template(it['messages'], tokenize=False, add_generation_prompt=True, enable_thinking=False)
           for it in items]
EOS = {tok.convert_tokens_to_ids('<|im_end|>'), tok.eos_token_id}


def auto_checks(it, text):
    c = {}
    if it['format'] == 'json':
        s = text.strip()
        try:
            json.loads(s); c['json_valid_strict'] = True
        except Exception:
            c['json_valid_strict'] = False
        m = re.search(r'```(?:json)?\s*(.*?)```', s, re.S)
        try:
            json.loads(m.group(1) if m else s); c['json_valid_after_fence_strip'] = True
        except Exception:
            c['json_valid_after_fence_strip'] = False
    if it['format'] == 'template':
        lines = re.findall(r'^\s*([a-z_0-9]+)\s*=(.*)$', it['messages'][-1]['content'], re.M)
        req = [k for k, rest in lines if '[Optional]' not in rest]
        c['template_required_keys_present'] = all(re.search(rf'\b{k}\s*=', text) for k in req)
    words = text.split()
    grams = [tuple(words[i:i + 4]) for i in range(max(0, len(words) - 3))]
    c['repeated_4gram_frac'] = round(1 - len(set(grams)) / len(grams), 4) if grams else 0.0
    c['math_markers'] = bool(re.search(r'\\boxed|\\\(|\\\[|\$[^$]+\$|\\frac|\\times|Step \d|The answer is', text))
    return c


rows = []
for cname, path in CONDS:
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float16).cuda().eval()
    enc = tok(prompts, return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=args.max_new, do_sample=False, temperature=None, top_p=None,
                             top_k=None, pad_token_id=tok.pad_token_id)
    gen = out[:, enc.input_ids.shape[1]:]
    for it, p, g in zip(items, prompts, gen):
        ids = g.tolist()
        n = next((i for i, t in enumerate(ids) if t in EOS), None)
        trunc = n is None
        ids = ids if trunc else ids[:n]
        while ids and ids[-1] == tok.pad_token_id:
            ids = ids[:-1]
        text = tok.decode(ids, skip_special_tokens=True)
        rows.append(dict(condition=cname, id=it['id'], base_question=it['base_question'], format=it['format'],
                         plain_index=it['plain_index'], prompt=p, response=text, n_new_tokens=len(ids),
                         truncated=trunc, annotation_subset=it['base_question'] in ANNOT, **auto_checks(it, text)))
    print(cname, f'{time.time() - t0:.1f}s', flush=True)
    del model; torch.cuda.empty_cache()

rows.sort(key=lambda r: (r['plain_index'], ['plain', 'json', 'template'].index(r['format']),
                         [c for c, _ in CONDS].index(r['condition'])))
with open(f'{args.out}/qualitative_responses.jsonl', 'w') as f:
    for r in rows:
        f.write(json.dumps(r) + '\n')
manual = ['format_compliance', 'coherence_relevance', 'repetition', 'unnecessary_math_framing',
          'inappropriate_content', 'excerpt', 'notes']
with open(f'{args.out}/annotation_sheet_42.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['base_question', 'format', 'condition', 'n_new_tokens', 'truncated', 'json_valid_after_fence_strip',
                'template_required_keys_present', 'repeated_4gram_frac', 'math_markers'] + manual)
    for r in rows:
        if r['annotation_subset']:
            w.writerow([r['base_question'], r['format'], r['condition'], r['n_new_tokens'], r['truncated'],
                        r.get('json_valid_after_fence_strip', ''), r.get('template_required_keys_present', ''),
                        r['repeated_4gram_frac'], r['math_markers']] + [''] * len(manual))
print('saved', len(rows), 'responses;', sum(r['annotation_subset'] for r in rows), 'in annotation subset')
