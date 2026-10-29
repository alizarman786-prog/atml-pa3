#!/bin/bash
# run_eval_all.sh RUNS_DIR OUT_DIR   (set LIMIT=N for a quick test)
RUNS=$1; OUT=$2; C=$(dirname $0); PROBE=${PROBE:-$RUNS/../data/probe_ids.json}
mkdir -p $OUT
pip install -q "lm_eval[hf]==0.4.13" 2>&1 | tail -2
pip list 2>/dev/null | grep -i -E '^(torch|transformers|lm.eval|datasets) '
(CUDA_VISIBLE_DEVICES=0 bash -c "bash $C/eval_lm.sh baseline Qwen/Qwen3-0.6B $OUT; bash $C/eval_lm.sh full_mid $RUNS/full/model_mid $OUT; bash $C/eval_lm.sh full $RUNS/full/model_final $OUT; bash $C/eval_lm.sh l2sp $RUNS/l2sp/model_final $OUT" > $OUT/gpu0.log 2>&1 &
 CUDA_VISIBLE_DEVICES=1 bash -c "python $C/diag.py --runs $RUNS --probe $PROBE --out $OUT/diag_results.json > $OUT/diag.log 2>&1; bash $C/eval_lm.sh lora4 $RUNS/lora4/model_final $OUT; bash $C/eval_lm.sh lora16 $RUNS/lora16/model_final $OUT; bash $C/eval_lm.sh oplora $RUNS/oplora/model_final $OUT" > $OUT/gpu1.log 2>&1 &
 wait)
grep -v "R_all" $OUT/diag.log | tail -20
python $C/summarize.py $OUT
