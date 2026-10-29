"""Task 3, Step 4: live comparison of a single long response, parallel sampling (3 x 1024, self-consistency) and
sequential refinement (shared initial 1024-token response + one 2048-token critique-and-revision) on the fixed
50-question MATH evaluation subset, with the same 3,072-token total output allowance per question.
Qwen/Qwen2.5-3B-Instruct (pinned revision), FP16, no quantization, batch size 1, temperature 0.7, top-p 0.8,
top-k 20, repetition penalty 1.0. Usage: python t3_live.py --questions live50.json --out DIR [--shard i --nshards n]
"""
import os, json, time, argparse, random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument('--questions', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--shard', type=int, default=0)
ap.add_argument('--nshards', type=int, default=1)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
MODEL, REV = 'Qwen/Qwen2.5-3B-Instruct', 'aa8e72537993ba99e69dfaafa59ed015b17504d1'
SYSTEM = 'You are a helpful assistant.'
INSTR = 'Think step by step inside <think> </think> tags, then give your final answer inside \\boxed{}.'
REVIEW = ('Review the solution above. Identify and correct any errors; if it is correct, retain its answer. '
          'Give your final answer inside \\boxed{}.')
SEED = 6304
tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, dtype=torch.float16).cuda().eval()
Q = json.load(open(args.questions))
Q = [q for i, q in enumerate(Q) if i % args.nshards == args.shard]
EOS = [tok.convert_tokens_to_ids('<|im_end|>'), tok.eos_token_id]


def generate(messages, max_new, seed):
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors='pt').input_ids.cuda()
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); random.seed(seed); np.random.seed(seed)
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new, do_sample=True, temperature=0.7, top_p=0.8, top_k=20,
                             repetition_penalty=1.0, eos_token_id=EOS, pad_token_id=tok.pad_token_id or EOS[0])
    torch.cuda.synchronize(); dt = time.time() - t0
    gen = out[0, ids.shape[1]:].tolist()
    stopped = len(gen) > 0 and gen[-1] in EOS
    if stopped:
        gen = gen[:-1]
    return dict(prompt=prompt, response=tok.decode(gen, skip_special_tokens=True), input_tokens=ids.shape[1],
                output_tokens=len(gen), truncated=not stopped and len(gen) >= max_new, seconds=dt, seed=seed,
                max_new_tokens=max_new)


torch.cuda.reset_peak_memory_stats()
path = f'{args.out}/live_shard{args.shard}.jsonl'
done = set()
if os.path.exists(path):
    done = {json.loads(l)['sample_id'] for l in open(path)}
f = open(path, 'a')
for q in Q:
    if q['sample_id'] in done:
        continue
    base = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': q['question_text'] + '\n\n' + INSTR}]
    s0 = SEED * 1000 + q['live_index'] * 10
    single = generate(base, 3072, s0)
    parallel = [generate(base, 1024, s0 + 1 + k) for k in range(3)]
    refine_msgs = base + [{'role': 'assistant', 'content': parallel[0]['response']}, {'role': 'user', 'content': REVIEW}]
    revision = generate(refine_msgs, 2048, s0 + 4)
    rec = dict(sample_id=q['sample_id'], live_index=q['live_index'], gold_answer=q['gold_answer'], single=single,
               parallel=parallel, revision=revision, peak_gpu_mem_bytes=torch.cuda.max_memory_allocated(),
               gpu=torch.cuda.get_device_name(0))
    f.write(json.dumps(rec) + '\n'); f.flush()
    print(q['live_index'], q['sample_id'], 'single', single['output_tokens'], 'par',
          [p['output_tokens'] for p in parallel], 'rev', revision['output_tokens'],
          f"{single['seconds'] + sum(p['seconds'] for p in parallel) + revision['seconds']:.0f}s", flush=True)
f.close()
print('DONE', flush=True)
