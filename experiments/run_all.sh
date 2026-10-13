#!/usr/bin/env bash
# Runs the full experiment suite end-to-end. Safe to re-run individual
# steps by commenting others out -- each script writes its own result file.
set -euo pipefail
cd "$(dirname "$0")"

echo "=== Checking Ollama server ==="
curl -sf http://localhost:11434/api/tags > /dev/null || {
  echo "Ollama server not running. Start it with: ollama serve"
  exit 1
}

EXP1_MODELS="Qwen/Qwen3-8B-Base Qwen/Qwen2.5-7B meta-llama/Llama-3.1-8B meta-llama/Meta-Llama-3-8B"
echo "=== Exp 1a: Reproduction (base models, full MMLU, letter 5-shot; no Ollama needed) ==="
python3 -u exp1_reproduction.py --n-per-subject 0 --seed 0 --protocols letter_5shot --models $EXP1_MODELS

echo "=== Exp 1b: Alternative protocols on the paired 20/subject subset ==="
# Resume is keyed on item uid, so this shares the items file with 1a.
python3 -u exp1_reproduction.py --n-per-subject 20 --seed 0 --protocols letter_0shot cloze_5shot --models $EXP1_MODELS

echo "=== Exp 2: Prompt/decoding sensitivity across chat models ==="
# Resume is keyed on uid and subsets are nested, so CoT can run on 10/subject
# first and later be extended to 20/subject.
python3 -u exp2_prompt_sensitivity.py --n-per-subject 20 --seed 0 --decoding-variants greedy --prompt-variants direct
python3 -u exp2_prompt_sensitivity.py --n-per-subject 10 --seed 0 --decoding-variants greedy --prompt-variants cot
python3 -u exp2_prompt_sensitivity.py --n-per-subject 20 --seed 0 --decoding-variants greedy --prompt-variants cot

echo "=== Exp 3: Base vs. chat checkpoint (Qwen3-8B) ==="
python3 -u exp3_base_vs_chat.py --n-per-subject 20 --seed 0 --chat-model qwen3:8b

echo "=== Aggregating results ==="
python3 analysis/aggregate.py

echo "=== Generating plots ==="
python3 analysis/plots.py

echo "=== Paper tables, macros and figure (paper/generated/) ==="
python3 analysis/paper_tables.py
python3 analysis/meta_evals.py  # needs accepted access to meta-llama/Llama-3.1-8B-evals

echo "Done. See experiments/results/, experiments/figures/ and paper/generated/."
