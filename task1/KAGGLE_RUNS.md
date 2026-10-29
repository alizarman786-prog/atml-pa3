# Task 1 — Kaggle run log

All runs: Kaggle notebooks (private, account `zarmanaliawan`), accelerator GPU T4 ×2, "Latest Container Image"
(torch 2.11.0+cu128, transformers 5.16.1, datasets 4.8.5, lm_eval 0.4.13 installed with pip). Code was attached as
private Kaggle datasets containing the files in `task1/`.

| # | Notebook / version | Purpose | Runtime | Outcome |
|---|---|---|---|---|
| 1 | `notebook96000c5ddc` v1 "task1-train-all5" (scriptVersionId 363283592) | `prep_data.py`, then 5 training runs, 2 at a time (GPU0: full → oplora; GPU1: l2sp → lora16 → lora4) | 8 h 03 m | success, 0 skipped updates in all runs |
| 2 | `notebookcfa6f344c5` v1 "task1-eval" | `run_eval_all.sh`: GSM8K + likelihood tasks for 7 conditions, `diag.py` | 1 h 55 m | GSM8K OK; likelihood tasks OOM at batch 64 (see below) |
| 3 | `notebookcfa6f344c5` v2 "task1-eval-ll-bs8" | HellaSwag / BoolQ / ARC-Easy for 7 conditions, batch size 8 | 50 m | success |
| 4 | `notebookc2f2f3a7bf` v1 "task1-final" | GSM8K re-evaluation (same settings), `qual_gen.py`, `diag.py`, `compact.py`, `make_figures.py`, repo assembly | 1 h 44 m | success; GSM8K scores identical to run 2 |

## Commands (one notebook cell each)

Run 1 (training):
```bash
python prep_data.py            # writes /kaggle/working/data
CUDA_VISIBLE_DEVICES=0 bash -c 'for m in full oplora; do python train.py --method $m > logs/train_$m.log 2>&1; done' &
CUDA_VISIBLE_DEVICES=1 bash -c 'for m in l2sp lora16 lora4; do python train.py --method $m > logs/train_$m.log 2>&1; done' &
wait
```
Before the full runs, every method was smoke-tested (`--smoke`, 4 updates exercising checkpoint/merge code) and timed
for 12 updates in an interactive session.

Run 2 (evaluation): `bash run_eval_all.sh <run1>/runs /kaggle/working/eval` (GSM8K batch 48; likelihood batch 64).

Run 3 (likelihood tasks, batch 8), per condition:
```bash
lm_eval --model hf --model_args pretrained=<model>,dtype=float16 --tasks hellaswag,boolq,arc_easy \
        --num_fewshot 0 --batch_size 8 --seed 6304 --log_samples --output_path <out>
```

Run 4 (final collection): GSM8K exactly as in `eval_lm.sh` (`gsm8k_cot`, 0-shot, `--apply_chat_template`,
`enable_thinking=False`, `max_gen_toks=1024`, batch 48, seed 6304) for the 7 conditions; then
`qual_gen.py`, `diag.py`, `compact.py` (with `--ll` pointing to run 3) and `make_figures.py`.

## Issues and their handling

* **Likelihood OOM (run 2).** With `--batch_size 64` the HF backend tried to allocate 6.15 GiB for SDPA attention on
  long BoolQ passages and every likelihood job failed immediately; GSM8K results were unaffected. Run 3 repeated
  only the likelihood tasks with batch size 8 (`eval_lm.sh` now defaults to 8).
* **GSM8K evaluated twice.** Run 2's per-example GSM8K files could not be mounted together with run 3's outputs, so
  run 4 regenerated them with identical settings. Run-2 flexible-extract scores for cross-checking:
  baseline 43.59, Full FT mid 57.01, Full FT 64.75, L2-SP 65.05, LoRA-4 59.74, LoRA-16 61.79, OPLoRA 61.87 (%).
* **First update at lr = 0.** `get_cosine_schedule_with_warmup` starts the linear warm-up at 0, so update 1 applies
  no change; the remaining 1,249 updates follow the 38-update warm-up + cosine schedule.
* GPU-hour usage for Task 1 (Kaggle quota): ≈ 8.1 h training + ≈ 2.8 h evaluation + final run.
