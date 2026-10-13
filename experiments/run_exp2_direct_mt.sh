#!/bin/bash
# Robustness check: direct prompting with exemplars as multi-turn chat.
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
echo "[$(date '+%F %T')] exp2 direct_mt starting" >> logs/run_when_idle.log
python3 exp2_prompt_sensitivity.py --n-per-subject 20 --seed 0 --decoding-variants greedy \
  --prompt-variants direct_mt --models llama3.1:8b qwen3:8b qwen2.5:7b-instruct
r=$?
python3 analysis/paper_tables.py > logs/tables.log 2>&1
echo "[$(date '+%F %T')] exp2 direct_mt finished (exit $r); tables regenerated" >> logs/run_when_idle.log
