# EE-5102/CS-6304 PA3 — Fine-Tuning, VLM Grounding and Test-Time Compute

This repository holds the code, configurations, split/example IDs and machine-readable results for Programming
Assignment 3. Task 1 (LLM fine-tuning), Task 2 (SigLIP–Qwen3 VLM grounding) and Task 3 (test-time compute) are included.

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
task2/
  vqa_norm.py               # official VQAv2 answer normalization + soft accuracy (leave-one-annotator-out)
  prep_vlm.py               # COCO captions / grounding (presence, counting) / VQAv2 complementary pairs + image download
  cache_feats.py            # frozen SigLIP-base last_hidden_state cache (196 x 768, FP16)
  train_vlm.py              # connector + LoRA, 3-stage curriculum, dev diagnostics, final 4-condition evaluation
  make_report.py            # figures + markdown tables from the run outputs
task3/
  t3_splits.py              # seeded dev/eval split (200/800 per dataset) + 50-question live subset
  t3_offline.py             # labelling, selection rules x budgets x 10 orders, signal analysis, adaptive stopping
  t3_live.py                # live single / parallel / refinement generation with Qwen2.5-3B-Instruct
  t3_live_eval.py           # grading of the live comparison
  make_report.py            # figures + markdown tables
results/task1/
  data/                     # train_info.json, train_selection.json (example IDs), probe_ids.json
  training/                 # <method>_summary.json, <method>_train_log.csv (per-update loss, lr, grad-norm, Omega, ...)
  benchmarks/               # benchmark_table.json, aggregate/ (lm-eval results JSON), per-example CSVs
  diagnostics/diag_results.json
  qualitative/              # qualitative_responses.jsonl (168), annotation_sheet_42.csv (manual columns to fill)
  tables/                   # benchmarks.md/csv, resources.md, diagnostics.md, qualitative_auto_checks.md
  figures/                  # fig1_training_loss, fig2_benchmark_deltas, fig3_diagnostics (.png/.pdf)
results/task2/
  data/                     # split files (image/question IDs, answers), data_stats.json, cache_info.json
  run_196tok/, run_49tok/   # train_log.jsonl, dev_eval.jsonl, summary.json, final_eval.json, final_per_example.jsonl
  report/                   # tables.md, qualitative.md, fig_train_loss.png, fig_dev_curves.png
results/task3/
  offline/                  # splits.json, live50.json, <ds>_results.json, <ds>_per_question.csv.gz, <ds>_labels.json
  live/                     # live_shard*.jsonl (full prompts/responses), live_results.json
  report/                   # tables.md, scaling / cost / adaptive / trajectory figures
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

## Task 2 — SigLIP–Qwen3 VLM grounding

Frozen `google/siglip-base-patch16-224` vision encoder (features cached once, post-LN `last_hidden_state`,
196 × 768) → trainable 2-layer MLP connector (768 → 1024 → 1024, GELU) → frozen `Qwen/Qwen3-0.6B` (FP16,
`enable_thinking=False`). The connector output is multiplied by a fixed scale computed before training from training
data only (mean token-embedding norm / mean initial connector-output norm on 512 training captions). Visual tokens are
inserted at the start of the user message (chat template split at an `<IMG>` placeholder), before the question.
Loss: mean token NLL per example over the target tokens + `<|im_end|>`, averaged over examples.

```bash
python task2/prep_vlm.py --out /kaggle/working/vlm_data            # COCO 2014 + VQAv2 annotations, images
python task2/cache_feats.py --data /kaggle/working/vlm_data --out /kaggle/working/feats
CUDA_VISIBLE_DEVICES=0 python task2/train_vlm.py --data /kaggle/working/vlm_data --feats /kaggle/working/feats --out run196
CUDA_VISIBLE_DEVICES=1 python task2/train_vlm.py --data /kaggle/working/vlm_data --feats /kaggle/working/feats --out run49 --pool
python task2/make_report.py --run 196=run196 --run 49=run49 --out results/task2/report
```

* **Data.** Training images come from COCO train2014 only; development and test images from val2014 only, each val
  image used at most once (no train/dev/test overlap; asserted in `train_vlm.py`). Captions: 6,000 images × 3
  captions (train), 96 (dev). Grounding pairs built from COCO instances: presence (positive instance ≥ 1% of the image)
  and counting (1–4 instances, no crowd / tiny instances); dev 12 + 12, test 32 + 32 pairs. VQAv2 complementary pairs
  with ≥ 8/10 annotator agreement, different consensus answers and the same answer type; train 640/256/640,
  dev 12/8/12, test 36/24/36 (yes/no / number / other). All selections are seeded (6304).
* **Curriculum.** Stage 1 caption alignment, connector only, 2,250 updates (lr 1e-3). Stage 2 visual instruction
  (presence 50% / counting 25% / caption replay 25%, drawn as complete pairs), connector 1e-4 + LoRA r=16 α=32
  dropout 0.05 on q/k/v/o (2e-5), 1,500 updates. Stage 3 natural VQA (VQA 70% / presence 10% / counting 10% /
  caption 10%), 5e-5 / 1e-5, 750 updates. Effective batch 16 (8 × 2), AdamW, 5% warm-up + cosine to 10%,
  clip 1.0, FP16 autocast with loss scaling, gradient checkpointing; dev evaluation every 500 updates and at stage
  ends; recovery checkpoints every 250 updates.
* **Development diagnostics.** Caption reference loss; answer NLL with the correct image, the pair partner's image and
  no image; swap penalty = NLL(partner image) − NLL(correct image).
* **Final evaluation.** Greedy decoding (≤ 16 tokens, manual KV cache) on the test pairs under four image conditions
  (correct / complementary partner / mismatched image from another pair, fixed seeded mapping / no image). VQA uses
  VQAv2 soft accuracy with official normalization; grounding uses normalized exact match. Reported: accuracy, pair
  accuracy, identical-answer rate, follows-partner rate, answer NLL, empty / truncated rates, generation time; 95% CIs
  from 4,000 pair-level bootstrap resamples (both members of a pair resampled together).
* **Optional run.** `--pool` averages each 2 × 2 block of the 14 × 14 patch grid (49 visual tokens); everything else
  identical.

## Task 3 — test-time compute

Offline analysis of the supplied rollouts (32 per question; Qwen2.5-Math on MATH, Qwen2.5-VL on MathVista) and a
live comparison with `Qwen/Qwen2.5-3B-Instruct` (revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`).

```bash
python task3/t3_splits.py --data pa3data --out t3
wget https://raw.githubusercontent.com/lupantech/MathVista/main/data/testmini.json -O t3/testmini.json
python task3/t3_offline.py --data pa3data --splits t3/splits.json --meta t3/testmini.json --out t3
python task3/t3_live.py --questions t3/live50.json --out live   # all 50 questions on one GPU
python task3/t3_live_eval.py --live live --out live
python task3/make_report.py --offline t3 --live live --out results/task3/report
```

* **Splits.** Per dataset, sorted question IDs are permuted with `numpy.random.default_rng(6304)`: first 200 =
  development (threshold selection only), remaining 800 = held-out evaluation. The live subset is the first 50 MATH
  evaluation questions of that permutation.
* **Answer equivalence.** MATH: `math-verify` (gold first), with cheap canonical string equality first; answers are
  grouped without the gold answer. Comparisons that hit the 2 s math-verify timeout are treated as not equivalent.
  MathVista: normalization from the released `testmini.json` metadata (choice letters / choice text / numeric
  matching for multiple choice; integer rounding; float precision). Missing answers form one `NO_ANSWER` group that
  is never correct.
* **Selection rules.** Single rollout, self-consistency (ties → earliest rollout), highest score and rank-weighted
  consistency (average ranks, weight (N − r)/(N − 1)) for mean log-probability, self-certainty (mean KL(U‖p)),
  shortest response, and on MathVista the vision-dependence score, VisConf (per-candidate-set 5th/95th percentile
  calibration of token confidence × mean vision dependence) and token-level fusion. Budgets N ∈ {1, 2, 4, 8, 16, 32},
  10 seeded rollout orders per question (accuracy averaged over orders), 95% CIs from 1,000 question-level bootstrap
  resamples (paired for differences).
* **Adaptive sampling.** Check points after 4 / 8 / 16 rollouts; agreement rule (unique leading answer with share ≥ τ)
  and quality-aware rule (additionally SC and score-RWC agree and the RWC margin ≥ γ). τ and γ are selected on the
  development split (lowest cost within 1 point of fixed N = 32 SC) and applied unchanged to the evaluation split.
* **Live comparison.** Same 3,072-token output allowance per question: one 3,072-token response; three 1,024-token
  samples with self-consistency; one shared 1,024-token response plus one 2,048-token critique-and-revision turn.
  FP16, batch size 1, temperature 0.7, top-p 0.8, top-k 20; seeds `6304·1000 + 10·index + k`. Last `\boxed{}` with
  balanced braces is extracted; full prompts and responses are saved. The reported live run used one
  Tesla T4 in Google Colab (transformers 5.20.2, torch 2.11.0), because the weekly Kaggle GPU quota was used up.

## External code and attribution

* [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (MIT) — benchmark tasks, prompts and
  scoring; used unmodified as a library/CLI.
* Hugging Face `transformers` / `datasets` — model, tokenizer, schedule (`get_cosine_schedule_with_warmup`) and data
  loading. The HellaSwag preprocessing in `diag.py` mirrors lm-eval's `hellaswag/utils.py`.
* Hugging Face `transformers` SigLIP / Qwen3 / Qwen2.5 models; COCO 2014 annotations and images; VQAv2 questions,
  answers and complementary pairs; VQAv2 answer normalization follows the official evaluation script.
* `math-verify` (Hugging Face) for MATH answer equivalence; MathVista `testmini.json` metadata for normalization.
* LoRA initialization follows Hu et al. (2021) / PEFT defaults; OPLoRA projections follow Xiong & Xie (Sec. 3.2–3.3).
* Code was written with assistance from an LLM coding assistant; all code was reviewed and is the author's
  responsibility, as permitted by the assignment policy.
