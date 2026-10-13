#!/usr/bin/env bash
# Waits until no other large GPU job is running, then runs the remaining paper
# experiments: exp1 on MPS and exp2 on Ollama run in parallel (~21 GB combined).
# Every step resumes from its per-item file, so re-running after an
# interruption continues where it stopped.
#
#   ./run_when_idle.sh            # wait for idle GPU, then run
#   ./run_when_idle.sh --now      # skip the idle check
#
# "Idle" = no other python process using >1 GB RSS or >20% CPU and no model
# loaded in Ollama, for 3 consecutive checks 60 s apart.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1

EXP1_MODELS="Qwen/Qwen3-8B-Base Qwen/Qwen2.5-7B meta-llama/Llama-3.1-8B meta-llama/Meta-Llama-3-8B"

log() { echo "[$(date '+%F %T')] $*" | tee -a logs/run_when_idle.log; }

busy() {
  # Other python jobs using >1 GB RSS or >20% CPU (MPS memory is not in RSS,
  # so CPU use catches GPU jobs that are still loading), or a loaded Ollama model.
  BLOCKER=$(ps -axo pid=,rss=,%cpu=,command= | awk -v me=$$ \
    '$4 ~ /[Pp]ython/ && $0 !~ /resource_tracker/ && $1 != me && ($2 > 1000000 || $3 > 20) {print $1, $5, $6; exit}')
  [[ -n "$BLOCKER" ]] && return 0
  BLOCKER=$(ollama ps 2>/dev/null | awk 'NR == 2 {print "ollama:" $1}')
  [[ -n "$BLOCKER" ]] && return 0
  return 1
}

if [[ "${1:-}" != "--now" ]]; then
  log "Waiting for idle GPU..."
  ok=0; n=0
  while (( ok < 3 )); do
    if busy; then
      ok=0
      (( n % 30 == 0 )) && log "Busy: $BLOCKER"
    else
      ok=$((ok + 1))
    fi
    n=$((n + 1))
    sleep 60
  done
fi
curl -sf http://localhost:11434/api/tags > /dev/null || { log "Ollama server not running"; exit 1; }
log "Starting runs."

(
  python3 exp1_reproduction.py --n-per-subject 0 --seed 0 --protocols letter_5shot --models $EXP1_MODELS \
    && python3 exp1_reproduction.py --n-per-subject 20 --seed 0 --protocols letter_0shot cloze_5shot --models $EXP1_MODELS
) > logs/exp1.log 2>&1 &
P1=$!

(
  # Direct on the full subset, then CoT on a nested 10/subject half first so a
  # balanced CoT sample exists early, then extend CoT to 20/subject.
  python3 exp2_prompt_sensitivity.py --n-per-subject 20 --seed 0 --decoding-variants greedy --prompt-variants direct \
    && python3 exp2_prompt_sensitivity.py --n-per-subject 10 --seed 0 --decoding-variants greedy --prompt-variants cot \
    && python3 exp2_prompt_sensitivity.py --n-per-subject 20 --seed 0 --decoding-variants greedy --prompt-variants cot
) > logs/exp2.log 2>&1 &
P2=$!

log "exp1 pid $P1 (logs/exp1.log), exp2 pid $P2 (logs/exp2.log)"
wait $P1; r1=$?; log "exp1 finished (exit $r1)"
wait $P2; r2=$?; log "exp2 finished (exit $r2)"

python3 analysis/paper_tables.py > logs/tables.log 2>&1
python3 analysis/meta_evals.py >> logs/tables.log 2>&1
log "Tables regenerated (exit codes exp1=$r1 exp2=$r2)."
