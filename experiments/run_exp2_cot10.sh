#!/bin/bash
# exp2 CoT on the nested 10/subject subset only, then regenerate tables.
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
echo "[$(date '+%F %T')] exp2 CoT 10/subject (resumed after Ollama restart)" >> logs/run_when_idle.log
python3 exp2_prompt_sensitivity.py --n-per-subject 10 --seed 0 --decoding-variants greedy --prompt-variants cot
r=$?
python3 analysis/paper_tables.py > logs/tables.log 2>&1
python3 analysis/meta_evals.py >> logs/tables.log 2>&1
echo "[$(date '+%F %T')] exp2 CoT finished (exit $r); tables regenerated" >> logs/run_when_idle.log
