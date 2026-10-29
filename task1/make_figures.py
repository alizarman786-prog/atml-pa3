"""Build Task 1 figures and tables from results/task1 (machine-readable outputs).
Usage: python task1/make_figures.py --res results/task1 --out results/task1/figures
"""
import os, json, csv, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('--res', default='results/task1')
ap.add_argument('--out', default='results/task1/figures')
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
T = f'{args.res}/tables'
os.makedirs(T, exist_ok=True)

METHODS = ['full', 'l2sp', 'lora4', 'lora16', 'oplora']
LABEL = {'baseline': 'Baseline', 'full_mid': 'Full FT (mid)', 'full': 'Full FT', 'l2sp': 'Full FT + L2-SP',
         'lora4': 'LoRA r=4', 'lora16': 'LoRA r=16', 'oplora': 'OPLoRA r=16'}
COLOR = {'full': '#2a78d6', 'l2sp': '#eb6834', 'lora4': '#1baf7a', 'lora16': '#eda100', 'oplora': '#e87ba4',
         'full_mid': '#9dbfe9', 'baseline': '#8a8984'}
CONDS = ['baseline', 'full_mid', 'full', 'l2sp', 'lora4', 'lora16', 'oplora']
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#e4e3df', 'grid.linewidth': 0.6, 'axes.edgecolor': '#8a8984',
                     'figure.dpi': 150, 'savefig.bbox': 'tight'})


def read_log(m):
    rows = list(csv.DictReader(open(f'{args.res}/training/{m}_train_log.csv')))
    f = lambda k: np.array([float(r[k]) if r[k] not in ('', 'None') else np.nan for r in rows])
    return {k: f(k) for k in rows[0].keys() if k != 'skipped'}


def ema(x, a=0.05):
    y = np.empty_like(x); s = x[0]
    for i, v in enumerate(x):
        s = a * v + (1 - a) * s; y[i] = s
    return y


logs = {m: read_log(m) for m in METHODS}
summ = {m: json.load(open(f'{args.res}/training/{m}_summary.json')) for m in METHODS}

# ---------------- Fig 1: SFT loss curves (+ L2-SP penalty panel)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={'width_ratios': [1.6, 1]})
ax = axes[0]
for m in METHODS:
    u, l = logs[m]['update'], logs[m]['sft_loss']
    ax.plot(u, l, color=COLOR[m], alpha=0.15, lw=0.6)
    ax.plot(u, ema(l), color=COLOR[m], lw=1.6, label=LABEL[m])
ax.set_xlabel('Optimizer update'); ax.set_ylabel('SFT loss (per response token)')
ax.set_ylim(0.1, 0.6); ax.legend(frameon=False, fontsize=7.5, ncol=1)
ax.set_title('Training loss (EMA 0.05; raw faint)', fontsize=9, loc='left')
ax = axes[1]
lg = logs['l2sp']
ax.plot(lg['update'], lg['lam_omega'], color=COLOR['l2sp'], lw=1.6, label=r'$\lambda\Omega$ (L2-SP)')
ax.set_xlabel('Optimizer update'); ax.set_ylabel(r'$\lambda\Omega$')
ax.set_title(r'L2-SP penalty ($\lambda=10^{-3}$)', fontsize=9, loc='left')
fig.tight_layout(); fig.savefig(f'{args.out}/fig1_training_loss.png'); fig.savefig(f'{args.out}/fig1_training_loss.pdf')
plt.close(fig)

# ---------------- Benchmarks table + Fig 2 (change from baseline, pp)
B = json.load(open(f'{args.res}/benchmarks/benchmark_table.json'))
PRIM = [('GSM8K', 'gsm8k_cot|exact_match,flexible-extract'), ('HellaSwag', 'hellaswag|acc_norm,none'),
        ('BoolQ', 'boolq|acc,none'), ('ARC-Easy', 'arc_easy|acc_norm,none')]
RAW = [('HellaSwag acc', 'hellaswag|acc,none'), ('ARC-Easy acc', 'arc_easy|acc,none')]
with open(f'{T}/benchmarks.md', 'w') as f:
    f.write('| Condition | ' + ' | '.join(n for n, _ in PRIM) + ' | ' + ' | '.join(n for n, _ in RAW) +
            ' | GSM8K trunc. % | GSM8K unextractable % |\n')
    f.write('|' + '---|' * (2 + len(PRIM) + len(RAW) + 1) + '\n')
    for c in CONDS:
        cells = []
        for n, k in PRIM:
            v = 100 * B[c][k]; d = v - 100 * B['baseline'][k]
            cells.append(f'{v:.1f}' + ('' if c == 'baseline' else f' ({d:+.1f})'))
        for n, k in RAW:
            cells.append(f"{100 * B[c][k]:.1f}")
        cells += [f"{100 * B[c]['gsm8k_truncated_frac']:.2f}", f"{100 * B[c]['gsm8k_unextractable_frac']:.2f}"]
        f.write(f'| {LABEL[c]} | ' + ' | '.join(cells) + ' |\n')
    f.write('\nPrimary metrics: GSM8K flexible-extract exact match; HellaSwag and ARC-Easy acc_norm; BoolQ acc. '
            'Parentheses: change from the shared baseline in percentage points. Strict-match GSM8K is 0.0 for all '
            'conditions (no response uses the "The answer is" pattern).\n')
with open(f'{T}/benchmarks.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['condition'] + [k for _, k in PRIM + RAW] + ['gsm8k_truncated_frac', 'gsm8k_unextractable_frac'])
    for c in CONDS:
        w.writerow([c] + [B[c][k] for _, k in PRIM + RAW] + [B[c]['gsm8k_truncated_frac'], B[c]['gsm8k_unextractable_frac']])

fig, axes = plt.subplots(1, 4, figsize=(7.2, 3.0), sharey=False)
order = ['full_mid', 'full', 'l2sp', 'lora4', 'lora16', 'oplora']
for ax, (n, k) in zip(axes, PRIM):
    d = [100 * (B[c][k] - B['baseline'][k]) for c in order]
    ax.bar(range(len(order)), d, color=[COLOR[c] for c in order], width=0.72, edgecolor='white', linewidth=1)
    ax.axhline(0, color='#52514e', lw=0.8)
    ax.set_xticks(range(len(order))); ax.set_xticklabels([LABEL[c] for c in order], rotation=60, ha='right', fontsize=7)
    ax.set_title(f'{n} (base {100 * B["baseline"][k]:.1f})', fontsize=8.5, loc='left')
    ax.grid(axis='x', visible=False)
    off = 0.03 * max(1.0, max(abs(v) for v in d))
    for i, v in enumerate(d):
        ax.text(i, v + (off if v >= 0 else -off), f'{v:+.1f}', ha='center', va='bottom' if v >= 0 else 'top',
                fontsize=6.5, color='#0b0b0b')
    lo, hi = min(0, min(d)), max(0, max(d))
    ax.set_ylim(lo - 0.15 * max(1.0, hi - lo), hi + 0.15 * max(1.0, hi - lo))
axes[0].set_ylabel('Change from baseline (pp)')
fig.tight_layout(); fig.savefig(f'{args.out}/fig2_benchmark_deltas.png'); fig.savefig(f'{args.out}/fig2_benchmark_deltas.pdf')
plt.close(fig)

# ---------------- Resources table
with open(f'{T}/resources.md', 'w') as f:
    f.write('| Method | Trainable params | Train time (h) | s / update | Peak GPU mem allocated (GB) | Peak reserved (GB) | '
            'Saved trainable state | Final displacement ‖θ−θ0‖₂ | Method-specific overhead |\n|' + '---|' * 9 + '\n')
    for m in METHODS:
        s = summ[m]; ck = s['checkpoints']['1250']
        state = (f"{ck['adapter_bytes'] / 1e6:.1f} MB adapter" if 'adapter_bytes' in ck else
                 f"{ck['eval_model_bytes'] / 1e9:.2f} GB (FP16 model)")
        disp = f"{ck['displacement_l2']:.2f}" if 'displacement_l2' in ck else '—'
        oh = s.get('method_overhead', {})
        ohs = ('θ0 reference: %.2f GB (bf16), transfer %.2fs' % (oh['reference_bytes'] / 1e9, oh['reference_transfer_s'])
               if 'reference_bytes' in oh else
               'bases: %.0f MB, SVD %.1fs' % (oh['basis_bytes'] / 1e6, oh['svd_preprocess_s']) if 'basis_bytes' in oh else '—')
        su = np.nanmedian(logs[m]['step_time'])
        f.write(f"| {LABEL[m]} | {s['trainable_params']:,} | {s['train_time_s'] / 3600:.2f} | {su:.2f} | "
                f"{s['peak_gpu_mem_bytes_this_session'] / 1e9:.2f} | {s['peak_gpu_reserved_bytes_this_session'] / 1e9:.2f} | "
                f"{state} | {disp} | {ohs} |\n")
    f.write('\nRecovery checkpoint (trainable state + optimizer + scheduler + scaler + RNG): Full FT/L2-SP '
            f"{summ['full']['checkpoints']['1250']['recovery_bytes'] / 1e9:.2f} GB; LoRA-4 "
            f"{summ['lora4']['checkpoints']['1250']['recovery_bytes'] / 1e6:.1f} MB; LoRA-16/OPLoRA "
            f"{summ['lora16']['checkpoints']['1250']['recovery_bytes'] / 1e6:.1f} MB. "
            f"Full FT midpoint displacement: {summ['full']['checkpoints']['625']['displacement_l2']:.2f}. "
            'Skipped (overflow) updates: ' + ', '.join(f"{LABEL[m]} {summ[m]['skipped_updates_this_session']}" for m in METHODS) +
            '. Runs were trained two at a time on 2x Tesla T4 (one run per GPU).\n')

# ---------------- Diagnostics table + Fig 3
D = json.load(open(f'{args.res}/diagnostics/diag_results.json'))['conditions']
DC = ['full_mid', 'full', 'l2sp', 'lora4', 'lora16', 'oplora']
with open(f'{T}/diagnostics.md', 'w') as f:
    f.write('| Condition | Layer | ρ_k (k=128) | η | R math (mean) | R non-math (mean) | math/non-math |\n|' + '---|' * 7 + '\n')
    for c in DC:
        for l in ['0', '13', '27']:
            d = D[c][l]
            f.write(f"| {LABEL[c]} | {l} | {d['rho_k']:.4f} | {d['eta']:.4f} | {d['R_math_mean']:.4f} | "
                    f"{d['R_nonmath_mean']:.4f} | {d['R_math_mean'] / d['R_nonmath_mean']:.2f} |\n")
    f.write('\nComputed in FP32 on down_proj. For OPLoRA ρ_k is zero by construction (projected factors). '
            'z is the starting model\'s down_proj input at the last prompt token; GSM8K probe prompts all end in the same '
            'chat-template suffix, so math-prompt z vectors are highly similar.\n')

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
x = np.arange(3); w = 0.13
for i, c in enumerate(DC):
    axes[0].bar(x + (i - 2.5) * w, [D[c][l]['rho_k'] for l in ['0', '13', '27']], w, color=COLOR[c], label=LABEL[c])
    axes[1].bar(x + (i - 2.5) * w, [D[c][l]['eta'] for l in ['0', '13', '27']], w, color=COLOR[c])
    axes[2].bar(x + (i - 2.5) * w, [D[c][l]['R_math_mean'] / D[c][l]['R_nonmath_mean'] for l in ['0', '13', '27']], w,
                color=COLOR[c])
for ax, t in zip(axes, [r'Subspace interference $\rho_{128}$', r'Relative update $\eta$', r'$\bar R$(math) / $\bar R$(non-math)']):
    ax.set_xticks(x); ax.set_xticklabels(['L0', 'L13', 'L27']); ax.set_title(t, fontsize=8.5, loc='left')
    ax.grid(axis='x', visible=False)
axes[2].axhline(1, color='#52514e', lw=0.8, ls='--')
fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=6, frameon=False, fontsize=7,
           bbox_to_anchor=(0.5, -0.08))
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(f'{args.out}/fig3_diagnostics.png'); fig.savefig(f'{args.out}/fig3_diagnostics.pdf')
plt.close(fig)

# ---------------- Qualitative automatic-check summary
Q = [json.loads(l) for l in open(f'{args.res}/qualitative/qualitative_responses.jsonl')]
with open(f'{T}/qualitative_auto_checks.md', 'w') as f:
    f.write('| Condition | mean new tokens | truncated (/24) | JSON valid (/8, after fence strip) | template keys present (/8) | '
            'math markers (/24) | mean repeated-4gram frac |\n|' + '---|' * 7 + '\n')
    for c in CONDS:
        r = [q for q in Q if q['condition'] == c]
        f.write(f"| {LABEL[c]} | {np.mean([q['n_new_tokens'] for q in r]):.0f} | {sum(q['truncated'] for q in r)} | "
                f"{sum(q.get('json_valid_after_fence_strip', False) for q in r if q['format'] == 'json')} | "
                f"{sum(q.get('template_required_keys_present', False) for q in r if q['format'] == 'template')} | "
                f"{sum(q['math_markers'] for q in r)} | {np.mean([q['repeated_4gram_frac'] for q in r]):.3f} |\n")
    f.write('\nAutomatic checks only (regex/parse based); they are not a substitute for the manual 42-response annotation.\n')
print('figures and tables written to', args.out, T)
