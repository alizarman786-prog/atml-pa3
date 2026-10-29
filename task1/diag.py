"""Task 1 weight-space and activation diagnostics on down_proj (layers 0, 13, 27), FP32.
rho_k = ||U_k U_k^T dW||_F^2 / ||dW||_F^2  (k=128, bases of the starting weight)
eta   = ||dW||_F / ||W0||_F
R(z)  = ||dW z||_2 / (||W0 z||_2 + 1e-8), z = starting model's down_proj input at the last prompt token.
Usage: python diag.py --runs /path/to/runs --probe /path/to/probe_ids.json --out diag_results.json
"""
import re, json, argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument('--runs', required=True)
ap.add_argument('--probe', required=True)
ap.add_argument('--out', default='diag_results.json')
args = ap.parse_args()
MODEL = 'Qwen/Qwen3-0.6B'; LAYERS = [0, 13, 27]; K = 128
dev = 'cuda'
CONDS = [('full_mid', 'full', 'mid'), ('full', 'full', 'final'), ('l2sp', 'l2sp', 'final'),
         ('lora4', 'lora4', 'final'), ('lora16', 'lora16', 'final'), ('oplora', 'oplora', 'final')]

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).float().eval()  # FP32 copy of original
W0 = {l: model.model.layers[l].mlp.down_proj.weight.detach().clone() for l in LAYERS}

# ---------------- probe prompts in benchmark formats
probe = json.load(open(args.probe))
gsm = load_dataset('openai/gsm8k', 'main', split='test')
hs = load_dataset('Rowan/hellaswag', split='validation')


def hs_pre(t):
    t = t.strip().replace(' [title]', '. ')
    t = re.sub('\\[.*?\\]', '', t)
    return t.replace('  ', ' ')


prompts = []
for i in probe['gsm8k_test']:
    msg = 'Q: ' + gsm[i]['question'] + '\nA:'
    prompts.append(('math', i, tok.apply_chat_template([{'role': 'user', 'content': msg}], tokenize=False,
                                                       add_generation_prompt=True, enable_thinking=False)))
for i in probe['hellaswag_validation']:
    d = hs[i]
    ctx = d['ctx_a'] + ' ' + d['ctx_b'].capitalize()
    prompts.append(('nonmath', i, hs_pre(d['activity_label'] + ': ' + ctx)))

# ---------------- cache z (input to down_proj) at the last prompt token, starting model
Z = {l: [] for l in LAYERS}
cur = {}
hooks = [model.model.layers[l].mlp.down_proj.register_forward_hook(
    lambda m, inp, out, l=l: cur.__setitem__(l, inp[0][0, -1].detach().clone())) for l in LAYERS]
with torch.no_grad():
    for _, _, p in prompts:
        ids = tok(p, return_tensors='pt', add_special_tokens=False).input_ids.to(dev)
        model(input_ids=ids)
        for l in LAYERS:
            Z[l].append(cur[l])
for h in hooks:
    h.remove()
Z = {l: torch.stack(v) for l, v in Z.items()}  # (64, d_in)
is_math = torch.tensor([c == 'math' for c, _, _ in prompts])
bases = {}
for l in LAYERS:
    U, S, Vh = torch.linalg.svd(W0[l], full_matrices=False)
    bases[l] = U[:, :K]

res = dict(layers=LAYERS, k=K, probe=probe, conditions={})
for name, run, tag in CONDS:
    try:
        d = torch.load(f'{args.runs}/{run}/diag_downproj_{tag}.pt')
    except FileNotFoundError:
        print('missing', name); continue
    out = {}
    for l in LAYERS:
        if f'{l}.W' in d:
            dW = d[f'{l}.W'].to(dev).float() - W0[l]
        else:
            dW = float(d[f'{l}.scale']) * d[f'{l}.B_eff'].to(dev).float() @ d[f'{l}.A_eff'].to(dev).float()
        Uk = bases[l]
        nd2 = dW.pow(2).sum()
        rho = (Uk @ (Uk.T @ dW)).pow(2).sum() / nd2 if nd2 > 0 else torch.tensor(float('nan'))
        eta = dW.norm() / W0[l].norm()
        z = Z[l]
        R = (z @ dW.T).norm(dim=1) / ((z @ W0[l].T).norm(dim=1) + 1e-8)
        out[str(l)] = dict(rho_k=rho.item(), eta=eta.item(), dW_fro=dW.norm().item(),
                           R_math_mean=R[is_math].mean().item(), R_nonmath_mean=R[~is_math].mean().item(),
                           R_math_median=R[is_math].median().item(), R_nonmath_median=R[~is_math].median().item(),
                           R_all=R.tolist())
        print(name, l, {k: round(v, 5) for k, v in out[str(l)].items() if k != 'R_all'}, flush=True)
    res['conditions'][name] = out
json.dump(res, open(args.out, 'w'), indent=1)
