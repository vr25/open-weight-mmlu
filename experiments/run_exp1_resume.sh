#!/bin/bash
# Resume exp1 with one process per model (guarantees memory is released between models).
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
EXP1_MODELS="meta-llama/Llama-3.1-8B meta-llama/Meta-Llama-3-8B Qwen/Qwen3-8B-Base Qwen/Qwen2.5-7B"
set -e
for m in meta-llama/Llama-3.1-8B meta-llama/Meta-Llama-3-8B; do  # Qwen models already complete
  python3 exp1_reproduction.py --n-per-subject 0 --seed 0 --protocols letter_5shot --models "$m"
done
for m in $EXP1_MODELS; do
  python3 exp1_reproduction.py --n-per-subject 20 --seed 0 --protocols letter_0shot cloze_5shot --models "$m"
done
echo "[$(date '+%F %T')] exp1 resume finished" >> logs/run_when_idle.log
