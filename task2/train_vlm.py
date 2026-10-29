"""Task 2: SigLIP-Qwen3 VLM. Frozen SigLIP features (cached) -> trainable 2-layer MLP connector -> frozen Qwen3-0.6B
(LoRA on q/k/v/o from Stage 2). Three-stage curriculum, development evaluation, final four-condition evaluation.
Usage: python train_vlm.py --data DIR --feats DIR --out DIR [--resume] [--smoke]
"""
import os, re, sys, json, math, time, glob, random, argparse, collections
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vqa_norm import normalize, soft_accuracy

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True)
ap.add_argument('--feats', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--ckpt_dir', default='/tmp/vlm_ckpt')
ap.add_argument('--resume', action='store_true')
ap.add_argument('--smoke', action='store_true', help='tiny budgets to exercise every code path')
ap.add_argument('--eval_only', default=None, help='path to final weights to only run final evaluation')
ap.add_argument('--pool', action='store_true', help='optional: 49 visual tokens (average each 2x2 block of the 14x14 grid)')
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True); os.makedirs(args.ckpt_dir, exist_ok=True)
SEED = 6304
MODEL = 'Qwen/Qwen3-0.6B'
dev = torch.device('cuda')
MICRO, ACCUM = 8, 2
EFF = MICRO * ACCUM
STAGES = [dict(name='caption', updates=2250, conn_lr=1e-3, lora_lr=None),
          dict(name='visual_instruction', updates=1500, conn_lr=1e-4, lora_lr=2e-5),
          dict(name='natural_vqa', updates=750, conn_lr=5e-5, lora_lr=1e-5)]
DEV_EVERY, CKPT_EVERY = 500, 250
if args.smoke:
    for s, u in zip(STAGES, [6, 4, 4]):
        s['updates'] = u
    DEV_EVERY, CKPT_EVERY = 6, 5
STAGE_ENDS = list(np.cumsum([s['updates'] for s in STAGES]))
TOTAL = STAGE_ENDS[-1]
CAPTION_PROMPT = 'Describe this image in one sentence.'
Q_CAP, CAP_CAP, ANS_CAP = 80, 64, 16


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


set_seed(SEED)
rng = random.Random(SEED)  # curriculum sampler

# ------------------------------------------------------------------ data
D = {k: json.load(open(f'{args.data}/{k}.json')) for k in
     ['caption_train', 'caption_dev', 'grounding_train_pool', 'grounding_dev', 'grounding_test', 'vqa_train',
      'vqa_dev', 'vqa_test']}
FE = {}
for split in ['train', 'val']:
    obj = torch.load(f'{args.feats}/feats_{split}.pt')
    FE[split] = (dict((i, n) for n, i in enumerate(obj['ids'])), obj['feats'])
train_img = set(FE['train'][0]); val_img = set(FE['val'][0])
dev_imgs = {c['image_id'] for c in D['caption_dev']} | {it['image_id'] for k in ['grounding_dev', 'vqa_dev'] for p in D[k] for it in p['items']}
test_imgs = {it['image_id'] for k in ['grounding_test', 'vqa_test'] for p in D[k] for it in p['items']}
assert not (dev_imgs & test_imgs), 'dev/test overlap'
assert dev_imgs <= val_img and test_imgs <= val_img
train_used = {c['image_id'] for c in D['caption_train']} | {it['image_id'] for k in ['grounding_train_pool', 'vqa_train'] for p in D[k] for it in p['items']}
assert train_used <= train_img and not (train_used & (dev_imgs | test_imgs))  # train2014 vs val2014 ids are disjoint
split_checks = dict(n_dev_images=len(dev_imgs), n_test_images=len(test_imgs), n_train_images=len(train_used),
                    dev_test_overlap=0, train_eval_overlap=0)


def feat(split, iid):
    idx, T = FE[split]
    f = T[idx[iid]]
    if args.pool:  # 14x14 -> 7x7 by averaging each spatial 2x2 block (row-major patch order)
        f = f.float().view(7, 2, 7, 2, 768).mean(dim=(1, 3)).reshape(49, 768).half()
    return f


# ------------------------------------------------------------------ model
tok = AutoTokenizer.from_pretrained(MODEL)
lm = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(dev)
for p in lm.parameters():
    p.requires_grad_(False)
lm.config.use_cache = False
EMB = lm.get_input_embeddings()
IM_END = tok.convert_tokens_to_ids('<|im_end|>')

# chat structure: visual tokens are inserted at the start of the user message, before the question
_tmpl = tok.apply_chat_template([{'role': 'user', 'content': '<IMG>\n<Q>'}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=False)
PRE, rest = _tmpl.split('<IMG>')
MID_PRE, POST = rest.split('<Q>')
PRE_IDS = tok(PRE, add_special_tokens=False).input_ids
print('template:', repr(PRE), '[196 visual]', repr(MID_PRE), '<question>', repr(POST), flush=True)


class Connector(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 1024); self.act = nn.GELU(); self.fc2 = nn.Linear(1024, 1024)
        self.register_buffer('scale', torch.tensor(1.0))

    def forward(self, x):
        return self.scale * self.fc2(self.act(self.fc1(x)))


class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha, dropout, A_init):
        super().__init__()
        self.base = base; self.scale = alpha / r
        self.A = nn.Parameter(A_init.to(dev, torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.base(x)
        h = F.linear(self.drop(x), self.A if torch.is_autocast_enabled() else self.A.to(x.dtype))
        return y + self.scale * F.linear(h, self.B if torch.is_autocast_enabled() else self.B.to(h.dtype))


conn = Connector().to(dev)
lora_modules = {}


def enable_lora():
    g = torch.Generator().manual_seed(SEED)
    for li, layer in enumerate(lm.model.layers):
        for t in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            base = getattr(layer.self_attn, t)
            bound = 1.0 / math.sqrt(base.in_features)
            m = LoRALinear(base, 16, 32, 0.05, (torch.rand(16, base.in_features, generator=g) * 2 - 1) * bound)
            setattr(layer.self_attn, t, m); lora_modules[f'{li}.{t}'] = m


def trainable_params():
    ps = list(conn.parameters())
    for m in lora_modules.values():
        ps += [m.A, m.B]
    return ps


def count(ps):
    return sum(p.numel() for p in ps)


# ------------------------------------------------------------------ sequence construction
def text_ids(s):
    return tok(s, add_special_tokens=False).input_ids


def build(example, split, image_mode='correct', image_id=None):
    """Return (pieces) for one example. image_mode: 'correct' | 'none' | 'other' (image_id given)."""
    q = example.get('question', CAPTION_PROMPT)
    tgt = example['answer'] if 'answer' in example else example['caption']
    q_ids = text_ids(q)[:Q_CAP]
    t_ids = text_ids(tgt)[:CAP_CAP if 'caption' in example else ANS_CAP]
    iid = example['image_id'] if image_mode == 'correct' else image_id
    f = None if image_mode == 'none' else feat(split, iid)
    return dict(pre=PRE_IDS, feat=f, mid=text_ids(MID_PRE) + q_ids + text_ids(POST), tgt=t_ids)


def assemble(batch, with_target=True, left_pad=False):
    """Embeds, attention mask, position ids, labels for a list of built pieces (visual embeddings computed here)."""
    seqs, labs = [], []
    feats = [b['feat'] for b in batch if b['feat'] is not None]
    vis = conn(torch.stack(feats).to(dev).float()) if feats else None
    vi = 0
    for b in batch:
        parts, lab = [EMB(torch.tensor(b['pre'], device=dev)).float()], [-100] * len(b['pre'])
        if b['feat'] is not None:
            parts.append(vis[vi]); lab += [-100] * vis[vi].shape[0]; vi += 1
        parts.append(EMB(torch.tensor(b['mid'], device=dev)).float()); lab += [-100] * len(b['mid'])
        if with_target:
            tgt = b['tgt'] + [IM_END]
            parts.append(EMB(torch.tensor(tgt, device=dev)).float()); lab += tgt
        seqs.append(torch.cat(parts)); labs.append(lab)
    L = max(s.shape[0] for s in seqs)
    E = torch.zeros(len(seqs), L, 1024, device=dev)
    M = torch.zeros(len(seqs), L, dtype=torch.long, device=dev)
    Y = torch.full((len(seqs), L), -100, dtype=torch.long, device=dev)
    for i, (s, l) in enumerate(zip(seqs, labs)):
        n = s.shape[0]
        sl = slice(L - n, L) if left_pad else slice(0, n)
        E[i, sl] = s; M[i, sl] = 1; Y[i, sl] = torch.tensor(l, device=dev)
    P = (M.cumsum(-1) - 1).clamp(min=0)
    return E, M, P, Y


def per_example_nll(E, M, P, Y, exclude_end=False):
    """Mean token NLL per example over target tokens (optionally excluding the end-of-turn token)."""
    out = lm(inputs_embeds=E.half() if not torch.is_autocast_enabled() else E, attention_mask=M, position_ids=P).logits
    lp = F.log_softmax(out[:, :-1].float(), -1)
    y = Y[:, 1:]
    mask = y != -100
    if exclude_end:
        mask = mask & (y != IM_END)
    tok_nll = -lp.gather(-1, y.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (tok_nll * mask).sum(1) / mask.sum(1).clamp(min=1)


# ------------------------------------------------------------------ connector-scale calibration (training data only)
with torch.no_grad():
    sample = rng.sample(D['caption_train'], 512)
    t_ids = torch.tensor([i for s in sample for i in text_ids(s['caption'])], device=dev)
    text_norm = EMB(t_ids).float().norm(dim=-1).mean().item()
    fv = torch.stack([feat('train', s['image_id']) for s in sample]).to(dev).float()
    vis_norm = conn(fv).norm(dim=-1).mean().item()
conn.scale.fill_(text_norm / vis_norm)
calib = dict(rule='scale = mean L2 norm of Qwen input embeddings of training-caption tokens / mean L2 norm of '
                  'initial connector outputs on the same 512 training images; fixed for all training',
             text_embedding_norm=text_norm, initial_visual_norm=vis_norm, scale=conn.scale.item())
print('calibration', calib, flush=True)
rng = random.Random(SEED)  # reset sampler after calibration draws

# ------------------------------------------------------------------ development evaluation
dev_examples = {}
for suite in ['grounding_dev', 'vqa_dev']:
    ex = []
    for p in D[suite]:
        a, b = p['items']
        ex.append((a, b['image_id'])); ex.append((b, a['image_id']))
    dev_examples[suite] = ex


@torch.no_grad()
def batched_nll(items, bs=16, exclude_end=False):
    res = []
    for i in range(0, len(items), bs):
        E, M, P, Y = assemble(items[i:i + bs])
        with torch.autocast('cuda', dtype=torch.float16):
            res.append(per_example_nll(E, M, P, Y, exclude_end))
    return torch.cat(res).cpu()


@torch.no_grad()
def dev_eval():
    lm.eval(); conn.eval()
    r = {}
    r['caption_ref_loss'] = batched_nll([build(c, 'val') for c in D['caption_dev']]).mean().item()
    for suite, ex in dev_examples.items():
        nl = {m: batched_nll([build(e, 'val', m if m != 'partner' else 'other', pid) for e, pid in ex],
                             exclude_end=True)
              for m in ['correct', 'partner', 'none']}
        r[f'{suite}_nll_correct'] = nl['correct'].mean().item()
        r[f'{suite}_nll_complementary'] = nl['partner'].mean().item()
        r[f'{suite}_nll_noimage'] = nl['none'].mean().item()
        r[f'{suite}_swap_penalty'] = (nl['partner'] - nl['correct']).mean().item()
        r[f'{suite}_noimage_penalty'] = (nl['none'] - nl['correct']).mean().item()
    lm.train(); conn.train()
    return r


# ------------------------------------------------------------------ curriculum sampler
def draw(stage_idx):
    """One draw = two examples: a complete question pair, or two caption examples (replay)."""
    u = rng.random()
    if stage_idx == 0:
        raise RuntimeError
    if stage_idx == 1:
        kind = 'presence' if u < 0.5 else 'counting' if u < 0.75 else 'caption'
    else:
        kind = 'vqa' if u < 0.7 else 'presence' if u < 0.8 else 'counting' if u < 0.9 else 'caption'
    if kind == 'caption':
        return kind, rng.sample(D['caption_train'], 2)
    if kind == 'vqa':
        return kind, rng.choice(D['vqa_train'])['items']
    pool = GROUND_POOLS[kind]
    return kind, rng.choice(pool)['items']


GROUND_POOLS = {k: [p for p in D['grounding_train_pool'] if p['kind'] == k] for k in ['presence', 'counting']}


def stage_example_stream(stage_idx):
    if stage_idx == 0:  # two passes over the 18,000 caption examples, each pass shuffled
        while True:
            order = list(range(len(D['caption_train'])))
            rng.shuffle(order)
            for i in order:
                yield 'caption', D['caption_train'][i]
    else:
        while True:
            kind, exs = draw(stage_idx)
            for e in exs:
                yield kind, e


# ------------------------------------------------------------------ training
def lr_lambda_factory(n):
    w = max(1, int(round(0.05 * n)))

    def f(step):
        if step < w:
            return (step + 1) / w
        prog = (step - w) / max(1, n - w)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    return f


def make_opt(stage):
    groups = []
    for ps, lr in [(list(conn.parameters()), stage['conn_lr']),
                   ([p for m in lora_modules.values() for p in (m.A, m.B)], stage['lora_lr'])]:
        if not ps or lr is None:
            continue
        groups.append(dict(params=[p for p in ps if p.ndim >= 2], lr=lr, weight_decay=0.01))
        nd = [p for p in ps if p.ndim < 2]
        if nd:
            groups.append(dict(params=nd, lr=lr, weight_decay=0.0))
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(stage['updates']))
    return opt, sched


def state_dict_trainable():
    d = {f'conn.{k}': v.detach().cpu() for k, v in conn.state_dict().items()}
    for n, m in lora_modules.items():
        d[f'lora.{n}.A'] = m.A.detach().cpu(); d[f'lora.{n}.B'] = m.B.detach().cpu()
    return d


def load_trainable(d):
    conn.load_state_dict({k[5:]: v for k, v in d.items() if k.startswith('conn.')})
    for n, m in lora_modules.items():
        m.A.data.copy_(d[f'lora.{n}.A']); m.B.data.copy_(d[f'lora.{n}.B'])


log_path = f'{args.out}/train_log.jsonl'
dev_path = f'{args.out}/dev_eval.jsonl'
summary = dict(visual_tokens=49 if args.pool else 196, seed=SEED, model=MODEL, vision='google/siglip-base-patch16-224', micro=MICRO, accum=ACCUM,
               stages=STAGES, calibration=calib, split_checks=split_checks, stage_info={},
               frozen_backbone_params=sum(p.numel() for p in lm.parameters()), template=dict(pre=PRE, mid=MID_PRE, post=POST))

if args.eval_only is None:
    lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    lm.train(); conn.train()
    scaler = torch.amp.GradScaler('cuda')
    start = 0
    if args.resume and glob.glob(f'{args.ckpt_dir}/ckpt.pt'):
        ck = torch.load(f'{args.ckpt_dir}/ckpt.pt', weights_only=False)
        start = ck['update']
        if start > STAGE_ENDS[0]:
            enable_lora()
        load_trainable(ck['trainable'])
        rng.setstate(ck['rng_sampler']); random.setstate(ck['rng_py']); torch.set_rng_state(ck['rng_torch'])
        torch.cuda.set_rng_state_all(ck['rng_cuda']); summary = ck['summary']
        print('resumed at', start, flush=True)
    else:
        for p in [log_path, dev_path]:
            if os.path.exists(p):
                os.remove(p)
        r = dev_eval(); r.update(update=0, stage='init')
        open(dev_path, 'a').write(json.dumps(r) + '\n'); print('dev@0', r, flush=True)
    logf = open(log_path, 'a')
    u = start
    for si, stage in enumerate(STAGES):
        s0 = STAGE_ENDS[si] - stage['updates']
        if u >= STAGE_ENDS[si]:
            continue
        if si == 1 and not lora_modules:
            enable_lora()
            lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        opt, sched = make_opt(stage)
        if args.resume and u > s0:
            opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched']); scaler.load_state_dict(ck['scaler'])
        stream = stage_example_stream(si)
        if args.resume and u > s0:  # fast-forward the stage-1 epoch stream (sampler state restored for stages 2-3)
            if si == 0:
                for _ in range((u - s0) * EFF):
                    next(stream)
        torch.cuda.reset_peak_memory_stats(); t_stage = time.time()
        n_tr = count(trainable_params())
        mix_total = collections.Counter()
        while u < STAGE_ENDS[si]:
            t_u = time.time(); mix = collections.Counter(); loss_sum = 0.0
            for mb in range(ACCUM):
                kinds, built = [], []
                for _ in range(MICRO):
                    kind, ex = next(stream)
                    kinds.append(kind); built.append(build(ex, 'train'))
                mix.update(kinds)
                E, M, P, Y = assemble(built)
                with torch.autocast('cuda', dtype=torch.float16):
                    nll = per_example_nll(E, M, P, Y)
                loss = nll.sum() / EFF
                scaler.scale(loss).backward()
                loss_sum += loss.item()
            if u == 0:
                g = conn.fc1.weight.grad
                # NaN/inf on the very first scaled step is normal (GradScaler then skips and lowers the scale)
                assert g is not None and (not torch.isfinite(g).all() or g.abs().sum() > 0), 'no gradient reaches the connector'
                summary['connector_grad_check'] = dict(grad_present=True, finite=bool(torch.isfinite(g).all()))
                assert all(not p.requires_grad for p in lm.parameters() if not any(p is q for q in trainable_params()))
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(trainable_params(), 1.0).item()
            s_before = scaler.get_scale(); scaler.step(opt); scaler.update()
            lrs = sched.get_last_lr(); sched.step(); opt.zero_grad(set_to_none=True)
            u += 1; mix_total.update(mix)
            rec = dict(update=u, stage=stage['name'], loss=loss_sum, conn_lr=lrs[0], lora_lr=(lrs[2] if len(lrs) > 2 else None),
                       grad_norm=gn, skipped=scaler.get_scale() < s_before, mix=dict(mix), step_time=time.time() - t_u)
            logf.write(json.dumps(rec) + '\n')
            if u % 50 == 0 or u == s0 + 1:
                logf.flush()
                print(f"u {u}/{TOTAL} [{stage['name']}] loss {loss_sum:.4f} lr {lrs[0]:.2e} gn {gn:.2f} "
                      f"{(time.time() - t_stage) / (u - s0):.2f}s/u mem {torch.cuda.max_memory_allocated() / 2**30:.2f}G", flush=True)
            if u % DEV_EVERY == 0 or u in STAGE_ENDS:
                r = dev_eval(); r.update(update=u, stage=stage['name'])
                open(dev_path, 'a').write(json.dumps(r) + '\n'); print('dev', r, flush=True)
            if u % CKPT_EVERY == 0 or u in STAGE_ENDS:
                summary['stage_info'].setdefault(stage['name'], {})['time_s_partial'] = time.time() - t_stage
                torch.save(dict(update=u, trainable=state_dict_trainable(), opt=opt.state_dict(), sched=sched.state_dict(),
                                scaler=scaler.state_dict(), rng_sampler=rng.getstate(), rng_py=random.getstate(),
                                rng_torch=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all(),
                                summary=summary), f'{args.ckpt_dir}/ckpt.pt')
        summary['stage_info'][stage['name']] = dict(time_s=time.time() - t_stage, updates=stage['updates'],
                                                    peak_gpu_mem_bytes=torch.cuda.max_memory_allocated(),
                                                    trainable_params=n_tr, mixture_counts=dict(mix_total))
        torch.save(state_dict_trainable(), f'{args.out}/weights_after_stage{si + 1}.pt')
        json.dump(summary, open(f'{args.out}/summary.json', 'w'), indent=1, default=str)
    logf.close()
else:
    enable_lora()
    load_trainable(torch.load(args.eval_only))

# ------------------------------------------------------------------ final evaluation (test suites)
lm.eval(); conn.eval()
lm.gradient_checkpointing_disable()


@torch.no_grad()
def greedy(built, max_new=16):
    E, M, P, _ = assemble(built, with_target=False, left_pad=True)
    out = [[] for _ in built]; done = torch.zeros(len(built), dtype=torch.bool, device=dev)
    with torch.autocast('cuda', dtype=torch.float16):
        o = lm(inputs_embeds=E, attention_mask=M, position_ids=P, use_cache=True)
    past = o.past_key_values; nxt = o.logits[:, -1].argmax(-1)
    pos = P[:, -1:]
    for step in range(max_new):
        for i in range(len(built)):
            if not done[i]:
                if nxt[i].item() == IM_END or nxt[i].item() == tok.eos_token_id:
                    done[i] = True
                else:
                    out[i].append(nxt[i].item())
        if done.all() or step == max_new - 1:
            break
        M = torch.cat([M, torch.ones(len(built), 1, dtype=M.dtype, device=dev)], 1); pos = pos + 1
        with torch.autocast('cuda', dtype=torch.float16):
            o = lm(inputs_embeds=EMB(nxt.unsqueeze(1)), attention_mask=M, position_ids=pos, past_key_values=past, use_cache=True)
        past = o.past_key_values; nxt = o.logits[:, -1].argmax(-1)
    trunc = [not done[i].item() for i in range(len(built))]
    return [tok.decode(x, skip_special_tokens=True).strip() for x in out], trunc


CONDS = ['correct', 'complementary', 'mismatched', 'noimage']
final = {}
per_example = []
for suite in ['grounding_test', 'vqa_test']:
    pairs = D[suite]
    # fixed mismatched mapping: each item gets an image from a different pair of the same suite (seeded)
    mrng = random.Random(SEED + 1)
    mismatch = {}
    for pi, p in enumerate(pairs):
        for it in p['items']:
            other = mrng.choice([q for qi, q in enumerate(pairs) if qi != pi])
            mismatch[(pi, it['image_id'])] = mrng.choice(other['items'])['image_id']
    items = [(pi, k, it) for pi, p in enumerate(pairs) for k, it in enumerate(p['items'])]
    cond_img = {
        'correct': lambda pi, k, it: ('correct', None),
        'complementary': lambda pi, k, it: ('other', pairs[pi]['items'][1 - k]['image_id']),
        'mismatched': lambda pi, k, it: ('other', mismatch[(pi, it['image_id'])]),
        'noimage': lambda pi, k, it: ('none', None)}
    res = {}
    for c in CONDS:
        built = [build(it, 'val', *cond_img[c](pi, k, it)) for pi, k, it in items]
        preds, truncs = [], []
        t0 = time.time()
        for b in range(0, len(built), 8):
            p_, t_ = greedy(built[b:b + 8]); preds += p_; truncs += t_
        gen_time = (time.time() - t0) / len(built)
        nll = batched_nll(built, bs=8, exclude_end=True).tolist()
        res[c] = dict(preds=preds, nll=nll, gen_time_per_answer=gen_time, truncated=truncs)
    # scoring
    rows = []
    for j, (pi, k, it) in enumerate(items):
        p = pairs[pi]; partner = p['items'][1 - k]
        row = dict(suite=suite, pair_id=p['pair_id'], pair_index=pi, member=k, image_id=it['image_id'],
                   question=it['question'], answer=it['answer'],
                   qtype=p.get('kind', p.get('answer_type')), partner_answer=partner['answer'])
        for c in CONDS:
            pred = res[c]['preds'][j]
            if suite == 'vqa_test':
                acc = soft_accuracy(pred, it['answers'])
                match = normalize(pred) == it['consensus']
                follows_partner = normalize(pred) == partner['consensus']
            else:
                acc = float(normalize(pred) == normalize(it['answer']))
                match = acc == 1.0
                follows_partner = normalize(pred) == normalize(partner['answer'])
            row[c] = dict(pred=pred, acc=acc, match=match, empty=pred == '', nll=res[c]['nll'][j],
                          follows_partner=follows_partner, truncated=res[c]['truncated'][j])
        rows.append(row)
    per_example += rows
    # aggregate
    B = 4000; brng = np.random.default_rng(SEED)
    npair = len(pairs)
    pair_rows = collections.defaultdict(list)
    for r in rows:
        pair_rows[r['pair_index']].append(r)
    # per-pair arrays (rows are ordered pair-major: row = 2 * pair_index + member)
    arr = {}
    for c in CONDS:
        npred = [normalize(r[c]['pred']) for r in rows]
        arr[c] = dict(acc=np.array([r[c]['acc'] for r in rows]).reshape(npair, 2),
                      empty=np.array([r[c]['empty'] for r in rows], float).reshape(npair, 2),
                      nll=np.array([r[c]['nll'] for r in rows]).reshape(npair, 2),
                      follows=np.array([r[c]['follows_partner'] for r in rows], float).reshape(npair, 2),
                      pair_match=np.array([rows[2 * i][c]['match'] and rows[2 * i + 1][c]['match'] for i in range(npair)], float),
                      identical=np.array([npred[2 * i] == npred[2 * i + 1] for i in range(npair)], float))

    def agg(c, pidx=None):
        a = arr[c]; ix = np.arange(npair) if pidx is None else pidx
        return dict(acc=a['acc'][ix].mean(), pair_acc=a['pair_match'][ix].mean(), identical=a['identical'][ix].mean(),
                    empty=a['empty'][ix].mean(), nll=a['nll'][ix].mean(), follows_partner=a['follows'][ix].mean())
    out = {c: dict(**{k: float(v) for k, v in agg(c).items()}, gen_time_per_answer=res[c]['gen_time_per_answer'],
                   truncated=float(np.mean(res[c]['truncated']))) for c in CONDS}
    types = sorted({r['qtype'] for r in rows})
    out['by_type'] = {t: {c: dict(acc=float(np.mean([r[c]['acc'] for r in rows if r['qtype'] == t])),
                                  nll=float(np.mean([r[c]['nll'] for r in rows if r['qtype'] == t])),
                                  pair_acc=float(np.mean([all(r[c]['match'] for r in pair_rows[i])
                                                          for i in range(npair) if pairs[i].get('kind', pairs[i].get('answer_type')) == t])))
                          for c in CONDS} for t in types}
    # pair bootstrap: correct-image advantage over each control (accuracy and NLL), same resampled pairs
    boots = collections.defaultdict(list)
    for _ in range(B):
        idx = brng.integers(0, npair, npair)
        a = {c: agg(c, idx) for c in CONDS}
        for c in CONDS:
            boots[f'acc_{c}'].append(a[c]['acc']); boots[f'pair_acc_{c}'].append(a[c]['pair_acc'])
            if c != 'correct':
                boots[f'adv_acc_vs_{c}'].append(a['correct']['acc'] - a[c]['acc'])
                boots[f'adv_nll_vs_{c}'].append(a[c]['nll'] - a['correct']['nll'])
    point = {}
    for c in CONDS:
        if c != 'correct':
            point[f'adv_acc_vs_{c}'] = out['correct']['acc'] - out[c]['acc']
            point[f'adv_nll_vs_{c}'] = out[c]['nll'] - out['correct']['nll']
    out['bootstrap_95ci'] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boots.items()}
    out['advantages'] = point
    out['n_pairs'] = npair
    final[suite] = out
    print(suite, json.dumps({c: {k: round(v, 4) for k, v in out[c].items()} for c in CONDS}), flush=True)
    print(suite, 'advantages', json.dumps(point), flush=True)

json.dump(final, open(f'{args.out}/final_eval.json', 'w'), indent=1)
with open(f'{args.out}/final_per_example.jsonl', 'w') as f:
    for r in per_example:
        f.write(json.dumps(r) + '\n')
summary['final_eval_peak_gpu_mem_bytes'] = torch.cuda.max_memory_allocated()
json.dump(summary, open(f'{args.out}/summary.json', 'w'), indent=1, default=str)
print('DONE', flush=True)
