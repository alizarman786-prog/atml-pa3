"""Task 3 offline experiments on the supplied rollouts (Steps 1, 2, 3, 5 and the data for Step 6).
Usage: python t3_offline.py --data pa3data --splits splits.json --meta testmini.json --out DIR
"""
import os, re, json, math, time, argparse, collections, difflib
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True)
ap.add_argument('--splits', required=True)
ap.add_argument('--meta', required=True, help='MathVista testmini.json (question metadata)')
ap.add_argument('--out', required=True)
ap.add_argument('--only', default=None, help='math or mathvista')
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
SEED = 6304
BUDGETS = [1, 2, 4, 8, 16, 32]
N_ORDERS = 10
NO_ANS = 'NO_ANSWER'
SPLITS = json.load(open(args.splits))

# ====================================================================== answer normalization / equivalence
from math_verify import parse as mv_parse, verify as mv_verify

_parse_cache, _ver_cache = {}, {}


def canon(s):
    s = str(s)
    for a, b in [('\\left', ''), ('\\right', ''), ('\\!', ''), ('\\,', ''), ('\\;', ''), ('\\dfrac', '\\frac'),
                 ('\\tfrac', '\\frac'), (' ', ''), ('\\%', '%')]:
        s = s.replace(a, b)
    return s.rstrip('.')


def mparse(s):
    if s not in _parse_cache:
        try:
            _parse_cache[s] = mv_parse('\\boxed{' + s + '}', parsing_timeout=2)
        except Exception:
            _parse_cache[s] = []
    return _parse_cache[s]


def mequal(a, b):
    """MATH equivalence with math-verify (gold-first argument order); identical strings are equal."""
    if a == b or canon(a) == canon(b):
        return True
    k = (a, b)
    if k not in _ver_cache:
        pa, pb = mparse(a), mparse(b)
        try:
            _ver_cache[k] = bool(pa) and bool(pb) and bool(mv_verify(pa, pb, timeout_seconds=2))
        except Exception:
            _ver_cache[k] = False
    return _ver_cache[k]


META = json.load(open(args.meta))


def _num(s):
    s = re.sub(r'\\text\{[^}]*\}|\\mathrm\{[^}]*\}|\\,|\\!|\\;|\$|\\%|%|°|\\circ|\^\{?\\circ\}?', ' ', str(s))
    s = s.replace('\\dfrac', '\\frac').replace(',', '')
    m = re.search(r'\\frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}', s)
    if m:
        return float(m.group(1)) / float(m.group(2))
    m = re.search(r'-?\d+(?:\.\d+)?(?:[eE]-?\d+)?', s)
    return float(m.group(0)) if m else None


def vnorm(ans, pid):
    """MathVista normalization (answer-type, choice and precision rules of the released metadata)."""
    if ans is None or str(ans).strip() == '':
        return NO_ANS
    meta = META[str(pid)]
    a = str(ans).strip()
    if meta['question_type'] == 'multi_choice':
        choices = [str(c) for c in meta['choices']]
        letters = [chr(ord('A') + i) for i in range(len(choices))]
        m = re.findall(r'\(([A-Za-z])\)', a)
        if m and m[0].upper() in letters:
            return m[0].upper()
        b = re.sub(r'^\\text\{(.*)\}$', r'\1', a).strip().strip('.').strip()
        if b.upper() in letters:
            return b.upper()
        for L, c in zip(letters, choices):
            if b.lower() == c.strip().lower():
                return L
        bn = _num(b)
        if bn is not None:
            for L, c in zip(letters, choices):
                cn = _num(c)
                if cn is not None and abs(cn - bn) < 1e-9:
                    return L
        # official MathVista behaviour: most similar choice by string similarity
        sims = [difflib.SequenceMatcher(None, b.lower(), c.lower()).ratio() for c in choices]
        return letters[int(np.argmax(sims))]
    at = meta['answer_type']
    if at == 'integer':
        x = _num(a)
        return NO_ANS if x is None else str(int(round(x)))
    if at == 'float':
        x = _num(a)
        prec = int(meta.get('precision') or 0)
        return NO_ANS if x is None else str(round(x, prec))
    return re.sub(r'\s+', ' ', a.lower()).strip()  # text / list


def vgold(pid):
    meta = META[str(pid)]
    if meta['question_type'] == 'multi_choice':
        choices = [str(c) for c in meta['choices']]
        return chr(ord('A') + choices.index(str(meta['answer'])))
    return vnorm(meta['answer'], pid)


# ====================================================================== data
def load(name):
    folder = {'math': 'qwen25-math', 'mathvista': 'qwen25vl-mathvista'}[name]
    g = pd.read_parquet(f'{args.data}/{folder}/generations.parquet')
    p = pd.read_parquet(f'{args.data}/{folder}/probability_metrics.parquet')
    df = g.merge(p, on=['dataset', 'sample_id', 'rollout_id'], how='inner', validate='one_to_one')
    if name == 'mathvista':
        a = pd.read_parquet(f'{args.data}/{folder}/attention_metrics.parquet')
        a = a.rename(columns={'n_generated_text_tokens': 'attn_prev_counts'})
        df = df.merge(a, on=['dataset', 'sample_id', 'rollout_id'], how='inner', validate='one_to_one')
    assert len(df) == 32000
    bad = 0
    Q = {}
    df = df.sort_values(['sample_id', 'rollout_id']).reset_index(drop=True)
    cols = {c: df[c].tolist() for c in df.columns if c not in ('generated_text',)}
    for k in range(0, len(df), 32):
        sid = cols['sample_id'][k]
        assert cols['sample_id'][k + 31] == sid and cols['rollout_id'][k:k + 32] == list(range(32))
        R = []
        for j in range(k, k + 32):
            n = int(cols['n_generated_text_tokens'][j])
            lp, kl = cols['logp'][j], cols['kl_u_p'][j]
            ok = len(cols['generated_token_ids'][j]) == n == len(lp) == len(kl)
            d = dict(rid=int(cols['rollout_id'][j]), ans=cols['extracted_answer'][j], ntok=n,
                     capped=bool(cols['hit_max_new_tokens'][j]),
                     logp=np.asarray(lp, np.float64), kl=np.asarray(kl, np.float64))
            if name == 'mathvista':
                img = np.asarray(cols['img_attn_total'][j], np.float64); pr = np.asarray(cols['prompt_text_attn_total'][j], np.float64)
                ge = np.asarray(cols['generated_text_attn_total'][j], np.float64)
                ok = ok and len(img) == n == len(pr) == len(ge)
                d['v'] = img / np.maximum(img + pr + ge, 1e-12)
            bad += not ok
            R.append(d)
        Q[sid] = dict(gold=cols['gold_answer'][k], question=cols['question_text'][k], rollouts=R)
    print(name, 'loaded', len(Q), 'questions; alignment mismatches', bad, flush=True)
    return Q, bad


def label(name, Q):
    """Answer groups (formed without gold answers) and correctness labels."""
    t0 = time.time()
    for qn, (sid, q) in enumerate(Q.items()):
        if qn % 100 == 0:
            print(name, 'labelling', qn, round(time.time() - t0, 1), 's', flush=True)
        reps, gid = [], []
        for r in q['rollouts']:
            a = r['ans']
            if name == 'math':
                key = NO_ANS if (a is None or str(a).strip() == '') else str(a).strip()
            else:
                key = vnorm(a, sid)
            r['norm'] = key
            if key == NO_ANS:
                gid.append(-1); continue
            for j, rep in enumerate(reps):
                if (mequal(rep, key) if name == 'math' else rep == key):
                    gid.append(j); break
            else:
                reps.append(key); gid.append(len(reps) - 1)
        q['group'] = np.array(gid)
        q['group_reps'] = reps
        gold = str(q['gold']).strip() if name == 'math' else vgold(sid)
        q['gold_norm'] = gold
        q['correct'] = np.array([(r['norm'] != NO_ANS) and (mequal(gold, r['norm']) if name == 'math' else r['norm'] == gold)
                                 for r in q['rollouts']])
        q['ntok'] = np.array([r['ntok'] for r in q['rollouts']], float)
        q['L'] = np.array([r['logp'].mean() if r['ntok'] else -np.inf for r in q['rollouts']])
        q['K'] = np.array([r['kl'].mean() if r['ntok'] else -np.inf for r in q['rollouts']])
        if name == 'mathvista':
            q['V'] = np.array([r['v'].mean() if r['ntok'] else 0.0 for r in q['rollouts']])
    print(name, 'labelled in', round(time.time() - t0, 1), 's', flush=True)


# ====================================================================== selection rules
def sc_select(q, P):
    """Self-consistency: one vote per rollout (NO_ANSWER is a group); ties -> earliest supporting rollout."""
    g = q['group'][P]
    best, best_key = None, None
    seen = {}
    for pos, x in enumerate(g):
        seen.setdefault(x, [0, pos])[0] += 1
    for x, (cnt, first) in seen.items():
        key = (-cnt, first)
        if best_key is None or key < best_key:
            best, best_key = x, key
    return P[best_key[1]]


def highest(scores, P):
    return P[int(np.argmax(scores[P]))]  # first maximum = earliest in current order


def rwc_weights(scores, P):
    N = len(P)
    r = rankdata(-scores[P], method='average')  # rank 1 = best
    return (N - r) / (N - 1)


def rwc_select(q, scores, P, return_margin=False):
    N = len(P)
    if N == 1:
        return (P[0], 1.0) if return_margin else P[0]
    w = rwc_weights(scores, P)
    g = q['group'][P]
    W, first = {}, {}
    for pos, (x, wi) in enumerate(zip(g, w)):
        W[x] = W.get(x, 0.0) + wi
        first.setdefault(x, pos)
    best = min(W, key=lambda x: (-W[x], first[x]))
    if return_margin:
        vals = sorted(W.values(), reverse=True)
        tot = sum(vals)
        margin = (vals[0] - (vals[1] if len(vals) > 1 else 0.0)) / tot if tot > 0 else 0.0
        return P[first[best]], margin
    return P[first[best]]


def visconf(q, P):
    """VisConf on the current candidate set: pooled token-KL 5th/95th percentiles (linear) of revealed rollouts."""
    pooled = np.concatenate([q['rollouts'][i]['kl'] for i in P if q['rollouts'][i]['ntok']] or [np.zeros(1)])
    lo, hi = np.percentile(pooled, [5, 95])
    Z, Zt = np.zeros(32), np.zeros(32)
    for i in P:
        r = q['rollouts'][i]
        if not r['ntok']:
            continue
        c = np.clip((r['kl'] - lo) / max(hi - lo, 1e-8), 0, 1)
        Z[i] = q['V'][i] * c.mean()
        Zt[i] = (c * r['v']).mean()
    return Z, Zt


def methods_for(name):
    m = ['first', 'sc', 'logp_highest', 'logp_rwc', 'selfcert_highest', 'selfcert_rwc', 'shortest']
    if name == 'mathvista':
        m += ['vision_highest', 'vision_rwc', 'visconf_highest', 'visconf_rwc', 'visconf_tok_highest', 'visconf_tok_rwc']
    return m


def select_all(name, q, P):
    out = {'first': P[0], 'sc': sc_select(q, P), 'logp_highest': highest(q['L'], P), 'logp_rwc': rwc_select(q, q['L'], P),
           'selfcert_highest': highest(q['K'], P), 'selfcert_rwc': rwc_select(q, q['K'], P),
           'shortest': P[int(np.argmin(q['ntok'][P]))]}
    if name == 'mathvista':
        Z, Zt = visconf(q, P)
        out.update(vision_highest=highest(q['V'], P), vision_rwc=rwc_select(q, q['V'], P),
                   visconf_highest=highest(Z, P), visconf_rwc=rwc_select(q, Z, P),
                   visconf_tok_highest=highest(Zt, P), visconf_tok_rwc=rwc_select(q, Zt, P))
    return out


# ====================================================================== main loop
def run(name):
    Q, bad = load(name)
    label(name, Q)
    ids = sorted(Q)
    rng = np.random.default_rng(SEED)
    orders = {sid: [rng.permutation(32) for _ in range(N_ORDERS)] for sid in ids}
    M = methods_for(name)
    # per (question, N, method): order-averaged accuracy and cost; plus oracle coverage and distinct groups
    rows = []
    acc = {m: np.zeros((len(ids), len(BUDGETS))) for m in M}
    cost = {m: np.zeros((len(ids), len(BUDGETS))) for m in M}
    oracle = np.zeros((len(ids), len(BUDGETS))); ngroups = np.zeros((len(ids), len(BUDGETS)))
    t0 = time.time()
    for qi, sid in enumerate(ids):
        q = Q[sid]
        for bi, N in enumerate(BUDGETS):
            sel_ids = collections.defaultdict(list)
            for o in orders[sid]:
                P = o[:N]
                oracle[qi, bi] += q['correct'][P].any() / N_ORDERS
                ngroups[qi, bi] += len(set(q['group'][P].tolist())) / N_ORDERS
                sel = select_all(name, q, P)
                for m in M:
                    i = sel[m]
                    acc[m][qi, bi] += q['correct'][i] / N_ORDERS
                    cost[m][qi, bi] += (q['ntok'][P[0]] if m == 'first' else q['ntok'][P].sum()) / N_ORDERS
                    sel_ids[m].append(int(i))
            for m in M:
                i0 = sel_ids[m][0]
                rows.append(dict(sample_id=sid, N=N, method=m, acc=acc[m][qi, bi], token_cost=cost[m][qi, bi],
                                 oracle=oracle[qi, bi], selected_rollouts=' '.join(map(str, sel_ids[m])),
                                 selected_answer_order0=q['rollouts'][i0]['norm']))
        if qi % 200 == 0:
            print(name, qi, round(time.time() - t0, 1), flush=True)
    pd.DataFrame(rows).to_csv(f'{args.out}/{name}_per_question.csv.gz', index=False)
    split_idx = {s: np.array([ids.index(x) for x in SPLITS[name][s]]) for s in ['dev', 'eval']}
    res = dict(n_questions=len(ids), alignment_mismatches=int(bad), budgets=BUDGETS, methods=M, splits={})
    brng = np.random.default_rng(SEED)
    for s, ix in split_idx.items():
        R = dict(oracle=oracle[ix].mean(0).tolist(), distinct_groups=ngroups[ix].mean(0).tolist(),
                 acc={m: acc[m][ix].mean(0).tolist() for m in M}, token_cost={m: cost[m][ix].mean(0).tolist() for m in M})
        # failure decomposition (order-averaged): sampling failure = no correct candidate; selection failure = missed
        R['sampling_failure'] = (1 - oracle[ix]).mean(0).tolist()
        R['selection_failure'] = {m: (oracle[ix] - acc[m][ix]).mean(0).tolist() for m in M}
        # paired question bootstrap (1,000) for accuracy and differences vs self-consistency
        B = brng.integers(0, len(ix), (1000, len(ix)))
        R['acc_ci95'] = {m: np.percentile(acc[m][ix][B].mean(1), [2.5, 97.5], axis=0).tolist() for m in M}
        R['diff_vs_sc_ci95'] = {m: np.percentile((acc[m][ix] - acc['sc'][ix])[B].mean(1), [2.5, 97.5], axis=0).tolist() for m in M}
        res['splits'][s] = R
    # ---------------------------------------------------------------- Step 3: what the signals capture (N = 32)
    sig = {'neg_length': lambda q: -q['ntok'], 'length': lambda q: q['ntok'], 'logp': lambda q: q['L'], 'selfcert': lambda q: q['K']}
    if name == 'mathvista':
        sig.update(vision=lambda q: q['V'], visconf=lambda q: visconf(q, np.arange(32))[0],
                   visconf_tok=lambda q: visconf(q, np.arange(32))[1])
    S3 = {}
    for s, ix in split_idx.items():
        out = {}
        for k, f in sig.items():
            diffs, corrs = [], []
            for qi in ix:
                q = Q[ids[qi]]
                sc = f(q); y = q['correct'].astype(float)
                o = np.argsort(-sc, kind='stable')
                diffs.append(y[o[:8]].mean() - y[o[-8:]].mean())
                if y.std() > 0 and np.std(sc) > 0:
                    corrs.append(np.corrcoef(rankdata(sc), y)[0, 1])
            out[k] = dict(top_minus_bottom_quartile_acc=float(np.mean(diffs)),
                          mean_within_q_spearman_with_correct=float(np.mean(corrs)) if corrs else None,
                          n_questions_mixed=len(corrs))
        # length vs scores within question
        for k in [x for x in sig if x not in ('length', 'neg_length')]:
            cs = []
            for qi in ix:
                q = Q[ids[qi]]
                a, b = sig[k](q), q['ntok']
                if np.std(a) > 0 and np.std(b) > 0:
                    cs.append(np.corrcoef(rankdata(a), rankdata(b))[0, 1])
            out[f'spearman_length_vs_{k}'] = float(np.mean(cs))
        S3[s] = out
    if name == 'mathvista':
        cv = []
        for sid in SPLITS[name]['eval']:
            q = Q[sid]
            pooled = np.concatenate([r['kl'] for r in q['rollouts'] if r['ntok']])
            lo, hi = np.percentile(pooled, [5, 95])
            for r in q['rollouts']:
                if r['ntok'] > 2:
                    c = np.clip((r['kl'] - lo) / max(hi - lo, 1e-8), 0, 1)
                    if c.std() > 0 and r['v'].std() > 0:
                        cv.append(np.corrcoef(c, r['v'])[0, 1])
        cv = np.array(cv)
        S3['confidence_vision_within_rollout_corr'] = dict(mean=float(cv.mean()), median=float(np.median(cv)),
                                                           frac_positive=float((cv > 0).mean()), n=int(len(cv)),
                                                           q10=float(np.percentile(cv, 10)), q90=float(np.percentile(cv, 90)))
        # illustrative trajectories: first 3 eval questions (permutation order) where VisConf and token fusion disagree at N=32
        traj = []
        for sid in SPLITS[name]['eval']:
            q = Q[sid]
            Z, Zt = visconf(q, np.arange(32))
            a, b = int(np.argmax(Z)), int(np.argmax(Zt))
            if a != b:
                pooled = np.concatenate([r['kl'] for r in q['rollouts'] if r['ntok']])
                lo, hi = np.percentile(pooled, [5, 95])
                for i, tag in [(a, 'visconf_top'), (b, 'visconf_tok_top')]:
                    r = q['rollouts'][i]
                    traj.append(dict(sample_id=sid, rollout=i, tag=tag, correct=bool(q['correct'][i]),
                                     c=np.clip((r['kl'] - lo) / max(hi - lo, 1e-8), 0, 1).round(4).tolist(),
                                     v=r['v'].round(4).tolist()))
            if len(traj) >= 6:
                break
        json.dump(traj, open(f'{args.out}/{name}_trajectories.json', 'w'))
    res['signals_N32'] = S3
    # ---------------------------------------------------------------- Step 5: adaptive sampling (offline simulation)
    TAUS = [0.5, 0.625, 0.75, 0.875, 1.0]; GAMMAS = [0.0, 0.1, 0.25, 0.5]
    CHECK = [4, 8, 16]
    qscore = 'selfcert' if name == 'math' else 'visconf'
    # precompute per (question, order, n) the stopping statistics and SC answers
    stats = {}
    for qi, sid in enumerate(ids):
        q = Q[sid]
        for oi, o in enumerate(orders[sid]):
            for n in CHECK + [32]:
                P = o[:n]
                g = q['group'][P]
                cnt = collections.Counter(g.tolist())
                top = cnt.most_common()
                unique_lead = len(top) == 1 or top[0][1] > top[1][1]
                frac = top[0][1] / n
                sc_i = sc_select(q, P)
                if qscore == 'selfcert':
                    s = q['K']
                else:
                    s = visconf(q, P)[0]
                rwc_i, margin = rwc_select(q, s, P, return_margin=True)
                stats[(qi, oi, n)] = dict(lead_ok=unique_lead and top[0][0] != -1, frac=frac,
                                          agree=q['group'][sc_i] == q['group'][rwc_i], margin=margin,
                                          correct=bool(q['correct'][sc_i]), cost=float(q['ntok'][P].sum()))

    def simulate(ixs, tau, gamma, quality):
        accs, costs, ns, early_wrong_fixed_right = [], [], collections.Counter(), 0
        for qi in ixs:
            for oi in range(N_ORDERS):
                stop = 32
                for n in CHECK:
                    st = stats[(qi, oi, n)]
                    ok = st['lead_ok'] and st['frac'] >= tau
                    if quality:
                        ok = ok and st['agree'] and st['margin'] >= gamma
                    if ok:
                        stop = n; break
                st = stats[(qi, oi, stop)]
                accs.append(st['correct']); costs.append(st['cost']); ns[stop] += 1
                if stop < 32 and not st['correct'] and stats[(qi, oi, 32)]['correct']:
                    early_wrong_fixed_right += 1
        return dict(acc=float(np.mean(accs)), token_cost=float(np.mean(costs)),
                    mean_rollouts=float(sum(k * v for k, v in ns.items()) / sum(ns.values())),
                    stop_dist={str(k): v / sum(ns.values()) for k, v in sorted(ns.items())},
                    incorrect_early_stops_where_fixed32_correct=early_wrong_fixed_right / (len(ixs) * N_ORDERS))

    dev_ix, ev_ix = split_idx['dev'], split_idx['eval']
    fixed32_dev = simulate(dev_ix, 2.0, 0, False)  # tau > 1 never stops -> always 32
    ad = dict(fixed32_dev=fixed32_dev, grid={}, selected={}, eval={})
    for fam, quality in [('agreement', False), ('quality_aware', True)]:
        cands = []
        for tau in TAUS:
            for gamma in (GAMMAS if quality else [None]):
                r = simulate(dev_ix, tau, gamma or 0.0, quality)
                ad['grid'][f'{fam}|tau={tau}|gamma={gamma}'] = r
                cands.append((tau, gamma, r))
        cands.append((None, None, fixed32_dev))  # always-32 fallback
        ok = [c for c in cands if c[2]['acc'] >= fixed32_dev['acc'] - 0.01]
        best = min(ok, key=lambda c: (c[2]['token_cost'], -c[2]['acc'], c[0] if c[0] is not None else 9, c[1] or 0))
        ad['selected'][fam] = dict(tau=best[0], gamma=best[1], dev=best[2])
        ad['eval'][fam] = (simulate(ev_ix, best[0], best[1] or 0.0, quality) if best[0] is not None
                           else simulate(ev_ix, 2.0, 0, False))
    ad['eval']['fixed32_sc'] = simulate(ev_ix, 2.0, 0, False)
    res['adaptive'] = ad
    json.dump(res, open(f'{args.out}/{name}_results.json', 'w'), indent=1, default=float)
    # per-question labels for later qualitative analysis
    lab = [dict(sample_id=sid, gold=Q[sid]['gold_norm'], n_groups=len(Q[sid]['group_reps']) + int((Q[sid]['group'] == -1).any()),
                frac_correct=float(Q[sid]['correct'].mean()), answers=[r['norm'] for r in Q[sid]['rollouts']],
                correct=Q[sid]['correct'].astype(int).tolist(), ntok=Q[sid]['ntok'].astype(int).tolist())
           for sid in ids]
    json.dump(lab, open(f'{args.out}/{name}_labels.json', 'w'))
    E = res['splits']['eval']
    print(name, 'EVAL oracle', [round(x, 3) for x in E['oracle']])
    for m in M:
        print(name, m, [round(x, 3) for x in E['acc'][m]], 'cost', [round(x) for x in E['token_cost'][m]], flush=True)
    print(name, 'adaptive', json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'stop_dist'} for k, v in ad['eval'].items()}))
    print(name, 'selected', json.dumps({k: (v['tau'], v['gamma']) for k, v in ad['selected'].items()}), flush=True)


for nm in ([args.only] if args.only else ['math', 'mathvista']):
    run(nm)
print('DONE')
