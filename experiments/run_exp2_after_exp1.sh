#!/bin/bash
# Run exp2 CoT after exp1 finishes (running both at once exceeds 48GB RAM and swaps).
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
while pgrep -f run_exp1_resume.sh > /dev/null; do sleep 60; done
echo "[$(date '+%F %T')] exp2 CoT starting (sequential)" >> logs/run_when_idle.log
python3 exp2_prompt_sensitivity.py --n-per-subject 10 --seed 0 --decoding-variants greedy --prompt-variants cot \
  && python3 exp2_prompt_sensitivity.py --n-per-subject 20 --seed 0 --decoding-variants greedy --prompt-variants cot
r=$?
python3 analysis/paper_tables.py > logs/tables.log 2>&1
python3 analysis/meta_evals.py >> logs/tables.log 2>&1
echo "[$(date '+%F %T')] exp2 finished (exit $r); tables regenerated" >> logs/run_when_idle.log
