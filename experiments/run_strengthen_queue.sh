#!/bin/bash
# Runs after ARC base: step-1 ladder (instruct loglik + bf16 generation), then Ollama ARC.
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1
while pgrep -f run_arc_base.sh > /dev/null; do sleep 60; done
log() { echo "[$(date '+%F %T')] $*" >> logs/run_when_idle.log; }
log "ladder: instruct loglik starting"
for m in meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct Qwen/Qwen3-8B; do
  python3 exp1_reproduction.py --n-per-subject 20 --seed 0 --protocols letter_5shot --models "$m"
done
log "ladder: bf16 generation starting"
for m in meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-7B-Instruct Qwen/Qwen3-8B; do
  python3 exp4_instruct_bf16.py --models "$m"
done
brew services start ollama; sleep 15
log "ARC ollama starting"
python3 exp3_arc.py ollama --models llama3.1:8b qwen2.5:7b-instruct qwen3:8b --variants direct direct_mt
python3 analysis/paper_tables.py > logs/tables.log 2>&1
log "strengthen queue finished; tables regenerated"
