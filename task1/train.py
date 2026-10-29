"""Task 1: fine-tune Qwen3-0.6B on MetaMathQA with Full FT, Full FT + L2-SP, LoRA (r=4,16) and OPLoRA (r=16,k=128).
Usage: python train.py --method {full,l2sp,lora4,lora16,oplora} [--resume] [--max_updates N]
"""
import os, sys, json, math, time, argparse, random, glob, shutil
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup

ap = argparse.ArgumentParser()
ap.add_argument('--method', required=True, choices=['full', 'l2sp', 'lora4', 'lora16', 'oplora'])
ap.add_argument('--data', default='/kaggle/working/data/train_40k.pt')
ap.add_argument('--out', default='/kaggle/working/runs')
ap.add_argument('--ckpt_dir', default='/tmp/ckpt')
ap.add_argument('--resume', action='store_true')
ap.add_argument('--max_updates', type=int, default=None, help='stop early (smoke test)')
ap.add_argument('--keep_all_ckpts', action='store_true')
ap.add_argument('--smoke', action='store_true', help='tiny run exercising checkpoint/merge code (updates 2 and 4)')
args = ap.parse_args()

MODEL = 'Qwen/Qwen3-0.6B'
SEED = 6304
MICRO, ACCUM = 2, 16
EFF = MICRO * ACCUM
N_EX = 40000
TOTAL_UPDATES = N_EX // EFF  # 1250
WARMUP = 38
CKPT_UPDATES = {312, 625, 937, 1250}
MID_UPDATE = 625
DIAG_LAYERS = [0, 13, 27]
FINAL_U = TOTAL_UPDATES
if args.smoke:
    CKPT_UPDATES, MID_UPDATE, FINAL_U = {2, 4}, 2, 4
    args.max_updates = 4
TARGETS = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
CFG = {
    'full':   dict(lr=2e-5, lam=0.0, r=None),
    'l2sp':   dict(lr=2e-5, lam=1e-3, r=None),
    'lora4':  dict(lr=2e-4, lam=0.0, r=4, alpha=8),
    'lora16': dict(lr=2e-4, lam=0.0, r=16, alpha=32),
    'oplora': dict(lr=2e-4, lam=0.0, r=16, alpha=32, k=128),
}[args.method]
IS_ADAPTER = CFG['r'] is not None
run_dir = f'{args.out}/{args.method}'
ck_dir = f'{args.ckpt_dir}/{args.method}'
os.makedirs(run_dir, exist_ok=True); os.makedirs(ck_dir, exist_ok=True)
dev = torch.device('cuda')


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


set_seed(SEED)


# ----------------------------------------------------------------------------- LoRA / OPLoRA
class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha, dropout, A_init, U=None, V=None):
        super().__init__()
        self.base = base
        self.r, self.alpha, self.scale = r, alpha, alpha / r
        self.A = nn.Parameter(A_init.to(dev, torch.float32))                       # r x d_in, random
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev))       # d_out x r, zero
        self.drop = nn.Dropout(dropout)
        if U is not None:
            self.register_buffer('U', U.contiguous(), persistent=False)            # d_out x k (fp32)
            self.register_buffer('V', V.contiguous(), persistent=False)            # d_in x k  (fp32)
        else:
            self.U = self.V = None

    def factors(self):
        """Effective (projected) factors, computed in FP32."""
        with torch.autocast('cuda', enabled=False):
            A, B = self.A, self.B
            if self.U is not None:
                B = B - self.U @ (self.U.T @ B)
                A = A - (A @ self.V) @ self.V.T
        return A, B

    def delta_w(self):
        A, B = self.factors()
        return self.scale * (B @ A)

    def forward(self, x):
        y = self.base(x)
        A, B = self.factors()
        h = F.linear(self.drop(x), A.to(x.dtype) if not torch.is_autocast_enabled() else A)
        return y + self.scale * F.linear(h, B.to(h.dtype) if not torch.is_autocast_enabled() else B)


def inject_lora(model, r, alpha, bases=None):
    g = torch.Generator().manual_seed(SEED)  # identical initial A for lora16 and oplora
    n_train = 0
    for li, layer in enumerate(model.model.layers):
        for parent_name, parent in [('self_attn', layer.self_attn), ('mlp', layer.mlp)]:
            for t in TARGETS:
                if not hasattr(parent, t):
                    continue
                base = getattr(parent, t)
                bound = 1.0 / math.sqrt(base.in_features)  # == kaiming_uniform_(a=sqrt(5)) as in PEFT/Hu et al.
                A0 = (torch.rand(r, base.in_features, generator=g) * 2 - 1) * bound
                U = V = None
                if bases is not None:
                    U, V = bases[f'{li}.{t}']
                m = LoRALinear(base, r, alpha, 0.05, A0, U, V)
                setattr(parent, t, m)
                n_train += m.A.numel() + m.B.numel()
    return n_train


def lora_modules(model):
    for li, layer in enumerate(model.model.layers):
        for parent in [layer.self_attn, layer.mlp]:
            for t in TARGETS:
                m = getattr(parent, t, None)
                if isinstance(m, LoRALinear):
                    yield f'{li}.{t}', m


# ----------------------------------------------------------------------------- model
tok = AutoTokenizer.from_pretrained(MODEL)
t_load = time.time()
orig = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)  # original checkpoint (bf16 on disk)
commit = getattr(orig.config, '_commit_hash', None)
theta0 = {n: p.detach().clone() for n, p in orig.named_parameters()}  # unique params (tied counted once), bf16 exact
method_overhead = {}

if not IS_ADAPTER:
    model = orig.to(torch.float32)  # FP32 master weights
    model.to(dev)
    for p in model.parameters():
        p.requires_grad_(True)
    if CFG['lam'] > 0:
        t0 = time.time()
        theta0_gpu = {n: v.to(dev) for n, v in theta0.items()}  # bf16 = exact copy of the starting checkpoint
        torch.cuda.synchronize()
        method_overhead['reference_bytes'] = sum(v.numel() * v.element_size() for v in theta0_gpu.values())
        method_overhead['reference_transfer_s'] = time.time() - t0
        method_overhead['reference_dtype'] = 'bfloat16 (exact copy of original checkpoint)'
    n_trainable = sum(p.numel() for p in model.parameters())
else:
    bases = None
    if args.method == 'oplora':
        t0 = time.time()
        bases = {}
        k = CFG['k']
        for li, layer in enumerate(orig.model.layers):
            for parent in [layer.self_attn, layer.mlp]:
                for t in TARGETS:
                    if hasattr(parent, t):
                        W = getattr(parent, t).weight.detach().to(dev, torch.float32)
                        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                        bases[f'{li}.{t}'] = (U[:, :k].contiguous(), Vh[:k].T.contiguous())
        torch.cuda.synchronize()
        method_overhead['svd_preprocess_s'] = time.time() - t0
        method_overhead['basis_bytes'] = sum(U.numel() * 4 + V.numel() * 4 for U, V in bases.values())
        method_overhead['k'] = k
    model = orig.to(torch.float16).to(dev)  # frozen FP16 backbone
    for p in model.parameters():
        p.requires_grad_(False)
    n_trainable = inject_lora(model, CFG['r'], CFG['alpha'], bases)
    del bases

model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
model.train()
print(f'[{args.method}] trainable params: {n_trainable:,}  load {time.time()-t_load:.1f}s', flush=True)

# ----------------------------------------------------------------------------- optimizer
decay, no_decay = [], []
for n, p in model.named_parameters():
    if not p.requires_grad:
        continue
    (no_decay if (p.ndim < 2 or n.endswith('.bias')) else decay).append(p)
opt = torch.optim.AdamW([{'params': decay, 'weight_decay': 0.01}, {'params': no_decay, 'weight_decay': 0.0}],
                        lr=CFG['lr'], betas=(0.9, 0.999), eps=1e-8)
sched = get_cosine_schedule_with_warmup(opt, WARMUP, TOTAL_UPDATES)
scaler = torch.amp.GradScaler('cuda')
train_params = decay + no_decay

# ----------------------------------------------------------------------------- data
D = torch.load(args.data)
ids_all, plen_all = D['ids'], D['plen']
assert len(ids_all) == N_EX
PAD = tok.pad_token_id


def make_micro(idxs):
    L = max(len(ids_all[i]) for i in idxs)
    x = torch.full((len(idxs), L), PAD, dtype=torch.long)
    am = torch.zeros((len(idxs), L), dtype=torch.long)
    lab = torch.full((len(idxs), L), -100, dtype=torch.long)
    for b, i in enumerate(idxs):
        s = torch.tensor(ids_all[i]); n = len(s)
        x[b, :n] = s; am[b, :n] = 1
        lab[b, plen_all[i]:n] = s[plen_all[i]:]  # assistant response + <|im_end|>
    return x.to(dev), am.to(dev), lab.to(dev)


def ce_chunk(h, y):
    return F.cross_entropy(model.lm_head(h).float(), y, reduction='sum')


def sft_loss_sum(x, am, lab):
    h = model.model(input_ids=x, attention_mask=am).last_hidden_state
    h = h[:, :-1]; y = lab[:, 1:]
    m = y != -100
    hs, ys = h[m], y[m]
    tot = h.new_zeros((), dtype=torch.float32)
    for c in range(0, hs.shape[0], 1024):  # chunked LM head + CE (checkpointed) to bound logit memory
        tot = tot + checkpoint(ce_chunk, hs[c:c + 1024], ys[c:c + 1024], use_reentrant=False)
    return tot, int(m.sum())


@torch.no_grad()
def displacement():
    sq = 0.0
    for n, p in model.named_parameters():
        sq += (p.detach().float() - theta0[n].to(p.device).float()).pow(2).sum().item()
    return math.sqrt(sq)


def rng_state():
    return dict(py=random.getstate(), np=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all())


def trainable_state():
    if IS_ADAPTER:
        return {f'{n}.{w}': getattr(m, w).detach().cpu() for n, m in lora_modules(model) for w in ['A', 'B']}
    return {n: p.detach().cpu() for n, p in model.named_parameters()}


def save_recovery(u_done):
    t0 = time.time()
    path = f'{ck_dir}/recovery_u{u_done}.pt'
    torch.save(dict(update=u_done, examples=u_done * EFF, trainable=trainable_state(), opt=opt.state_dict(),
                    sched=sched.state_dict(), scaler=scaler.state_dict(), rng=rng_state(), method=args.method), path)
    sz = os.path.getsize(path)
    if not args.keep_all_ckpts:
        for f in glob.glob(f'{ck_dir}/recovery_u*.pt'):
            if f != path:
                os.remove(f)
    return sz, time.time() - t0


def save_diag_weights(tag):
    """FP32 down_proj tensors for weight/activation diagnostics."""
    d = {}
    for li in DIAG_LAYERS:
        mod = model.model.layers[li].mlp.down_proj
        if IS_ADAPTER:
            A, B = mod.factors()
            d[f'{li}.A_eff'] = A.detach().cpu(); d[f'{li}.B_eff'] = B.detach().cpu(); d[f'{li}.scale'] = mod.scale
        else:
            d[f'{li}.W'] = mod.weight.detach().float().cpu()
    torch.save(d, f'{run_dir}/diag_downproj_{tag}.pt')


def dir_size(p):
    return sum(os.path.getsize(f) for f in glob.glob(f'{p}/**', recursive=True) if os.path.isfile(f))


def save_eval_model(tag):
    """Save an FP16 HF model directory for evaluation (adapters merged into a fresh backbone copy)."""
    out = f'{run_dir}/model_{tag}'
    info = {}
    if not IS_ADAPTER:
        sd = {n: p.detach().half().cpu() for n, p in model.state_dict().items()}
        fresh = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16)
        fresh.load_state_dict(sd, strict=True)
        fresh.save_pretrained(out, safe_serialization=True)
        del fresh, sd
    else:
        # adapter export (projected/effective factors for OPLoRA)
        ad = {}
        for n, m in lora_modules(model):
            A, B = m.factors()
            ad[f'{n}.A'] = A.detach().cpu(); ad[f'{n}.B'] = B.detach().cpu()
        ad_path = f'{run_dir}/adapter_{tag}.pt'
        torch.save(dict(factors=ad, r=CFG['r'], alpha=CFG['alpha'], targets=TARGETS, projected=args.method == 'oplora'),
                   ad_path)
        info['adapter_bytes'] = os.path.getsize(ad_path)
        fresh = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(dev)
        with torch.no_grad():
            for n, m in lora_modules(model):
                li, t = n.split('.')
                parent = fresh.model.layers[int(li)].self_attn if t in TARGETS[:4] else fresh.model.layers[int(li)].mlp
                W = getattr(parent, t).weight
                W.copy_((W.float() + m.delta_w()).half())
        # merged vs unmerged logits check on a small batch
        model.eval()
        x, am, _ = make_micro([0, 1])
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            l1 = model(input_ids=x, attention_mask=am).logits.float()
            l2 = fresh(input_ids=x, attention_mask=am).logits.float()
        mk = am.bool()
        info['merge_check_max_abs_logit_diff'] = (l1 - l2)[mk].abs().max().item()
        info['merge_check_mean_abs_logit_diff'] = (l1 - l2)[mk].abs().mean().item()
        info['merge_check_argmax_agree'] = (l1.argmax(-1) == l2.argmax(-1))[mk].float().mean().item()
        model.train()
        fresh.save_pretrained(out, safe_serialization=True)
        del fresh
    tok.save_pretrained(out)
    torch.cuda.empty_cache()
    info['eval_model_dir'] = out
    info['eval_model_bytes'] = dir_size(out)
    return info


# ----------------------------------------------------------------------------- resume
start_u = 0
log_path = f'{run_dir}/train_log.jsonl'
summary = dict(method=args.method, model=MODEL, commit=commit, seed=SEED, cfg=CFG, micro=MICRO, accum=ACCUM,
               total_updates=TOTAL_UPDATES, warmup=WARMUP, trainable_params=n_trainable,
               method_overhead=method_overhead, checkpoints={}, gpu=torch.cuda.get_device_name(0),
               torch=torch.__version__)
prev_train_time = 0.0
if args.resume:
    cks = sorted(glob.glob(f'{ck_dir}/recovery_u*.pt'), key=lambda f: int(f.split('_u')[-1][:-3]))
    if cks:
        ck = torch.load(cks[-1], weights_only=False)
        if IS_ADAPTER:
            mods = dict(lora_modules(model))
            for k_, v in ck['trainable'].items():
                n, w = k_.rsplit('.', 1)
                getattr(mods[n], w).data.copy_(v.to(dev))
        else:
            model.load_state_dict({k_: v for k_, v in ck['trainable'].items()}, strict=False)
        opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched']); scaler.load_state_dict(ck['scaler'])
        r = ck['rng']; random.setstate(r['py']); np.random.set_state(r['np']); torch.set_rng_state(r['torch'])
        torch.cuda.set_rng_state_all(r['cuda'])
        start_u = ck['update']
        if os.path.exists(f'{run_dir}/summary.json'):
            old = json.load(open(f'{run_dir}/summary.json'))
            summary['checkpoints'] = old.get('checkpoints', {}); prev_train_time = old.get('train_time_s', 0.0)
        # truncate log to resumed position
        if os.path.exists(log_path):
            lines = [l for l in open(log_path) if json.loads(l)['update'] <= start_u]
            open(log_path, 'w').writelines(lines)
        print(f'Resumed from {cks[-1]} at update {start_u}', flush=True)
elif os.path.exists(log_path):
    os.remove(log_path)

# ----------------------------------------------------------------------------- train
torch.cuda.reset_peak_memory_stats()
end_u = TOTAL_UPDATES if args.max_updates is None else min(TOTAL_UPDATES, start_u + args.max_updates)
t_train = time.time(); t_ckpt_total = 0.0
logf = open(log_path, 'a')
n_skipped = 0
lam = CFG['lam']
for u in range(start_u, end_u):
    t_u = time.time()
    ex_idx = list(range(u * EFF, (u + 1) * EFF))
    n_tok = sum(len(ids_all[i]) - plen_all[i] for i in ex_idx)  # valid response tokens in effective batch
    sft_sum = 0.0
    for mb in range(ACCUM):
        x, am, lab = make_micro(ex_idx[mb * MICRO:(mb + 1) * MICRO])
        with torch.autocast('cuda', dtype=torch.float16):
            ls, nt = sft_loss_sum(x, am, lab)
        scaler.scale(ls / n_tok).backward()
        sft_sum += ls.item()
    rec = {}
    if lam > 0:  # L2-SP: Omega = 1/2 sum (theta - theta0)^2 ; grad = lam (theta - theta0), added before unscale
        with torch.no_grad():
            sc = scaler.get_scale(); om = 0.0
            for n, p in model.named_parameters():
                d = p.detach() - theta0_gpu[n].float()
                om += 0.5 * d.pow(2).sum().item()
                p.grad.add_(d, alpha=lam * sc)
                del d
        rec.update(omega=om, lam_omega=lam * om)
    scaler.unscale_(opt)
    gn = torch.nn.utils.clip_grad_norm_(train_params, 1.0).item()
    s_before = scaler.get_scale()
    scaler.step(opt); scaler.update()
    skipped = scaler.get_scale() < s_before
    n_skipped += int(skipped)
    lr = sched.get_last_lr()[0]
    sched.step()
    opt.zero_grad(set_to_none=True)
    sft = sft_sum / n_tok
    rec.update(update=u + 1, examples=(u + 1) * EFF, sft_loss=sft, lr=lr, grad_norm=gn, scale=s_before,
               skipped=bool(skipped), resp_tokens=n_tok, step_time=time.time() - t_u)
    if lam > 0:
        rec['total_loss'] = sft + rec['lam_omega']
    logf.write(json.dumps(rec) + '\n')
    if (u + 1) % 10 == 0 or u == start_u:
        logf.flush()
        el = time.time() - t_train
        print(f'u {u+1}/{TOTAL_UPDATES} loss {sft:.4f} lr {lr:.2e} gn {gn:.2f} skip {n_skipped} '
              f'{el/(u+1-start_u):.2f}s/u mem {torch.cuda.max_memory_allocated()/2**30:.2f}G'
              + (f" omega {rec['omega']:.4f}" if lam > 0 else ''), flush=True)
    if (u + 1) in CKPT_UPDATES:
        t0 = time.time()
        ck = {}
        sz, _ = save_recovery(u + 1)
        ck['recovery_bytes'] = sz
        if (u + 1) == MID_UPDATE and args.method == 'full':
            ck['displacement_l2'] = displacement()
            save_diag_weights('mid')
            ck.update(save_eval_model('mid'))
        if (u + 1) == FINAL_U:
            if not IS_ADAPTER:
                ck['displacement_l2'] = displacement()
            save_diag_weights('final')
            ck.update(save_eval_model('final'))
        ck['save_time_s'] = time.time() - t0
        summary['checkpoints'][str(u + 1)] = ck
        t_ckpt_total += ck['save_time_s']
        summary['train_time_s'] = prev_train_time + (time.time() - t_train)
        json.dump(summary, open(f'{run_dir}/summary.json', 'w'), indent=1)
        print(f'checkpoint @ {u+1}: {ck}', flush=True)
logf.close()

summary['train_time_s'] = prev_train_time + (time.time() - t_train)
summary['ckpt_time_s_this_session'] = t_ckpt_total
summary['peak_gpu_mem_bytes_this_session'] = torch.cuda.max_memory_allocated()
summary['peak_gpu_reserved_bytes_this_session'] = torch.cuda.max_memory_reserved()
summary['skipped_updates_this_session'] = n_skipped
summary['finished_update'] = end_u
if IS_ADAPTER and args.method == 'oplora':
    with torch.no_grad():
        viol = []
        for n, m in lora_modules(model):
            A, B = m.factors()
            viol.append(max((m.U.T @ B).abs().max().item(), (A @ m.V).abs().max().item()))
    summary['oplora_max_constraint_violation'] = max(viol)
json.dump(summary, open(f'{run_dir}/summary.json', 'w'), indent=1)
print('DONE', json.dumps({k: v for k, v in summary.items() if k != 'checkpoints'}), flush=True)
