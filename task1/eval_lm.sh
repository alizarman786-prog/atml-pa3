#!/bin/bash
# Usage: eval_lm.sh NAME MODEL_PATH OUTDIR
# GSM8K: gsm8k_cot, zero-shot, non-thinking chat template, greedy, 1024 new tokens.
# HellaSwag / BoolQ / ARC-Easy: stock completion prompts, no chat template, candidate likelihood.
set -x
# NOTE: likelihood batch size 64 ran out of memory on a 16 GB T4 (BoolQ passages); results use batch size 8.
NAME=$1; M=$2; OUT=$3
mkdir -p $OUT/$NAME
T0=$(date +%s)
lm_eval --model hf \
  --model_args pretrained=$M,dtype=float16,enable_thinking=False \
  --tasks gsm8k_cot --num_fewshot 0 --apply_chat_template \
  --gen_kwargs max_gen_toks=1024 \
  --batch_size ${GSM_BS:-48} --seed 6304 --log_samples ${LIMIT:+--limit $LIMIT} \
  --output_path $OUT/$NAME/gsm8k > $OUT/$NAME/gsm8k.log 2>&1
T1=$(date +%s)
lm_eval --model hf \
  --model_args pretrained=$M,dtype=float16 \
  --tasks hellaswag,boolq,arc_easy --num_fewshot 0 \
  --batch_size ${LL_BS:-8} --seed 6304 --log_samples ${LIMIT:+--limit $LIMIT} \
  --output_path $OUT/$NAME/ll > $OUT/$NAME/ll.log 2>&1
T2=$(date +%s)
echo "{\"gsm8k_s\": $((T1-T0)), \"ll_s\": $((T2-T1))}" > $OUT/$NAME/eval_time.json
tail -n 12 $OUT/$NAME/gsm8k.log; tail -n 15 $OUT/$NAME/ll.log
