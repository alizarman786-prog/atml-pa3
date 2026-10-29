# EE-5102/CS-6304 PA3 — Fine-Tuning, VLM Grounding and Test-Time Compute

This repository holds the code, configurations, split/example IDs and machine-readable results for Programming
Assignment 3. **Status: Task 1 (LLM fine-tuning) is complete. Tasks 2 and 3 are not yet included.**

All Task 1 experiments ran on Kaggle (2× NVIDIA Tesla T4, 16 GB each) with `torch 2.11.0+cu128`,
`transformers 5.16.1`, `datasets 4.8.5`, `accelerate 1.14.0`, `lm_eval 0.4.13` (Python 3.13). Base model:
`Qwen/Qwen3-0.6B` at commit `c1899de289a04d12100db370d81485cdf75e47ca`, with thinking disabled
(`enable_thinking=False`) for training and generation. Seed: 6304.

## Repository layout

```
README.md
requirements.txt
task1/
  prep_data.py              # MetaMathQA selection (first 40k eligible, source order) + tokenization + probe-set IDs
  train.py                  # Full FT, Full FT + L2-SP, LoRA r=4/16, OPLoRA r=16 k=128 (one script, --method)
  eval_lm.sh                # lm-evaluation-harness: GSM8K (gsm8k_cot, 0-shot, chat template) + HellaSwag/BoolQ/ARC-Easy
  run_eval_all.sh           # runs eval_lm.sh for the 7 conditions on 2 GPUs + diag.py
  diag.py                   # weight-space (rho_k, eta) and activation (R(z)) diagnostics on down_proj, layers 0/13/27
  qual_gen.py               # 24 prompts x 7 conditions = 168 greedy responses + 42-response annotation sheet
  compact.py                # collects results into results/task1 (CSV/JSON)
  summarize.py              # quick benchmark summary used during evaluation (superseded by compact.py)
  make_figures.py           # figures + markdown tables from results/task1
  first_plot_questions.yaml # qualitative prompts (supplied with the assignment)
  KAGGLE_RUNS.md            # exact notebook commands and run log (versions, runtimes, issues)
results/task1/
  data/                     # train_info.json, train_selection.json (example IDs), probe_ids.json
  training/                 # <method>_summary.json, <method>_train_log.csv (per-update loss, lr, grad-norm, Omega, ...)
  benchmarks/               # benchmark_table.json, aggregate/ (lm-eval results JSON), per-example CSVs
  diagnostics/diag_results.json
  qualitative/              # qualitative_responses.jsonl (168), annotation_sheet_42.csv (manual columns to fill)
  tables/                   # benchmarks.md/csv, resources.md, diagnostics.md, qualitative_auto_checks.md
  figures/                  # fig1_training_loss, fig2_benchmark_deltas, fig3_diagnostics (.png/.pdf)
```

Model weights, recovery checkpoints, raw datasets and full lm-eval sample dumps are **not** committed (size); they
remain in the Kaggle notebook outputs listed in `task1/KAGGLE_RUNS.md` and can be regenerated with the commands below.

## Task 1 — how to reproduce

Requirements: `pip install -r requirements.txt` (GPU with ≥15 GB; T4 used). Paths default to Kaggle's
`/kaggle/working`; override with the script arguments.

```bash
# 1) data: first 40,000 eligible MetaMathQA rows (empty / >2048 formatted tokens / GSM8K-test matches excluded)
python task1/prep_data.py --out /kaggle/working/data

# 2) training (one run per GPU; each ~2.5-3.7 h on a T4). Recovery checkpoints at updates 312/625/937/1250.
CUDA_VISIBLE_DEVICES=0 python task1/train.py --method full      # also saves the update-625 midpoint model
CUDA_VISIBLE_DEVICES=1 python task1/train.py --method l2sp
CUDA_VISIBLE_DEVICES=0 python task1/train.py --method lora4
CUDA_VISIBLE_DEVICES=1 python task1/train.py --method lora16
CUDA_VISIBLE_DEVICES=0 python task1/train.py --method oplora
#    (add --resume to continue from the latest recovery checkpoint; --smoke for a 4-update test)

# 3) evaluation of the 7 conditions + diagnostics
PROBE=/kaggle/working/data/probe_ids.json bash task1/run_eval_all.sh /kaggle/working/runs /kaggle/working/eval

# 4) qualitative generations (168) + annotation sheet
python task1/qual_gen.py --runs /kaggle/working/runs --yaml task1/first_plot_questions.yaml --out /kaggle/working/qual

# 5) collect results, then figures/tables
python task1/compact.py --train /kaggle/working --gsm /kaggle/working/eval --ll <dir with LL outputs> \
       --qual /kaggle/working/qual --diag /kaggle/working/eval/diag_results.json --out .
python task1/make_figures.py --res results/task1 --out results/task1/figures
```

### Implementation notes (Task 1)

* **Formatting / loss.** Each example is the Qwen3 chat template with one user turn (`query`) and the generation
  prompt for non-thinking mode (`<|im_start|>assistant\n<think>\n\n</think>\n\n`); the target is the supplied
  `response` followed by `<|im_end|>`. Cross-entropy is computed only on response tokens + `<|im_end|>`; the summed
  token loss of each micro-batch is divided by the number of valid response tokens in the whole effective batch of 32.
  Examples are consumed in source order (no shuffling, no truncation). The LM head + CE is evaluated in
  checkpointed chunks of 1,024 tokens to bound logit memory (numerically the same loss).
* **Precision.** FP16 autocast with dynamic loss scaling. Full FT / L2-SP keep FP32 master weights; adapter runs use
  a frozen FP16 backbone with FP32 adapters; optimizer states are FP32; gradients are unscaled before clipping
  (global norm 1.0). Skipped (overflow) updates are logged per update (`skipped` column) — none occurred.
* **Schedule.** AdamW (0.9, 0.999, 1e-8), weight decay 0.01 on ≥2-D parameters only, cosine decay to 0 after 38
  linear warm-up updates (`transformers.get_cosine_schedule_with_warmup`; the first update therefore uses lr = 0),
  1,250 updates = 40,000 examples, gradient checkpointing on.
* **L2-SP.** Ω = ½ Σ (θ − θ0)² over all unique parameters (tied embedding counted once), λ = 1e-3, θ0 kept fixed as
  an exact bf16 copy of the checkpoint on the GPU. The penalty gradient λ(θ − θ0) is added to the (scaled) gradients
  once per update; Ω, λΩ and SFT + λΩ are logged.
* **LoRA / OPLoRA.** Custom `LoRALinear` on q/k/v/o/gate/up/down in all 28 layers; A ~ U(−1/√d_in, 1/√d_in)
  (PEFT's default kaiming-uniform), B = 0, dropout 0.05 on the adapter input, scale α/r. LoRA-16 and OPLoRA share the
  same initial A (same generator and order). OPLoRA computes FP32 SVD bases of each starting weight once and uses the
  effective factors B̃ = B − U_k(U_kᵀB), Ã = A − (AV_k)V_kᵀ (no dense projection matrices); exported adapters are
  the projected factors. Adapters are merged into fresh backbone copies for evaluation; merged-vs-unmerged logits are
  compared on a small batch (results in `training/<method>_summary.json`).
* **Evaluation.** lm-eval 0.4.13, HF backend, FP16. GSM8K: `gsm8k_cot`, `num_fewshot=0`, `--apply_chat_template`,
  `enable_thinking=False`, greedy, `max_gen_toks=1024`, flexible-extract exact match as the primary metric.
  HellaSwag / BoolQ / ARC-Easy: stock prompts, no chat template, zero-shot. Truncation flag: response ≥ 1,020 tokens.
* **Diagnostics.** `diag.py` caches the starting model's `down_proj` inputs z at the last prompt token for the probe
  set (32 GSM8K-test + 32 HellaSwag-validation questions chosen with seed 6304 before any evaluation) and evaluates
  ρ_k, η and R(z) in FP32 for every final model and the Full FT midpoint (ΔW = W_trained − W0 for full variants,
  ΔW = (α/r)B̃Ã for adapters).

### Deviations / known issues

* Likelihood benchmarks were first launched with batch size 64 and ran out of GPU memory; they were rerun with batch
  size 8 (no effect on the scores, which are per-example log-likelihoods).
* GSM8K was evaluated twice with identical settings (first evaluation run and the final collection run, because the
  first run's sample files were not accessible as an input); `results/task1` uses the final run. See
  `task1/KAGGLE_RUNS.md` for both runs' scores.
* The manual-judgment columns of `annotation_sheet_42.csv` (format compliance, coherence/relevance, repetition,
  unnecessary mathematical framing, inappropriate content, excerpt, notes) contain an LLM-assisted draft, marked in
  the `annotation_source` column, which must be verified by the author before use. Automatic checks (JSON parse,
  template keys, 4-gram repetition, math markers, truncation) are computed by `qual_gen.py`.

## External code and attribution

* [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (MIT) — benchmark tasks, prompts and
  scoring; used unmodified as a library/CLI.
* Hugging Face `transformers` / `datasets` — model, tokenizer, schedule (`get_cosine_schedule_with_warmup`) and data
  loading. The HellaSwag preprocessing in `diag.py` mirrors lm-eval's `hellaswag/utils.py`.
* LoRA initialization follows Hu et al. (2021) / PEFT defaults; OPLoRA projections follow Xiong & Xie (Sec. 3.2–3.3).
* Code was written with assistance from an LLM coding assistant; all code was reviewed and is the author's
  responsibility, as permitted by the assignment policy.
