#!/bin/bash
# assemble.sh SRC_DIR FINAL_DIR OUT_DIR : build the repository tree (code + results + figures) and zip it.
#   SRC_DIR   : flat folder with README.md, requirements.txt and all task1 source files
#   FINAL_DIR : output of compact.py (contains results/task1)
set -e
S=$1; F=$2; O=$3
rm -rf $O/pa3_repo; mkdir -p $O/pa3_repo/task1
cp $S/README.md $S/requirements.txt $O/pa3_repo/
for f in prep_data.py train.py eval_lm.sh run_eval_all.sh diag.py qual_gen.py compact.py summarize.py make_figures.py \
         assemble.sh first_plot_questions.yaml KAGGLE_RUNS.md; do cp $S/$f $O/pa3_repo/task1/; done
cp -r $F/results $O/pa3_repo/
cd $O/pa3_repo && python task1/make_figures.py --res results/task1 --out results/task1/figures
cd $O && rm -f pa3_repo.zip && zip -qr pa3_repo.zip pa3_repo && unzip -l pa3_repo.zip | tail -1
