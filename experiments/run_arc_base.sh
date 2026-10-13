#!/bin/bash
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
for m in meta-llama/Llama-3.1-8B Qwen/Qwen2.5-7B; do
  python3 exp3_arc.py base --models "$m"
done
echo "[$(date '+%F %T')] ARC base finished" >> logs/run_when_idle.log
