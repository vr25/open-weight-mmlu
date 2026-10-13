# Open-Weight Language Models: Landscape Survey + Reproducibility Study

This repo has:

1. **`experiments/`** — a self-contained empirical study testing whether
   published open-weight benchmark numbers actually reproduce under
   independent, controlled re-evaluation, and how much prompt/decoding
   choices alone move a model's score. Designed to run standalone, with no
   dependency on any particular assistant/session — see below.

## Experiments

**Motivating question:** the open-weight ecosystem's "the gap to closed
frontier models has basically closed" narrative rests on self-reported
leaderboard numbers, evaluated under inconsistent harnesses (prompt
templates, decoding params, and sometimes even different checkpoints than
what's publicly deployable). This study tests how much of that narrative
survives independent, controlled re-evaluation.

| Script | What it tests |
|---|---|
| `exp1_reproduction.py` | Re-evaluates the exact base checkpoints in the Qwen3 Technical Report's Table 6 (Qwen3-8B-Base 76.89, Qwen2.5-7B 74.16, Qwen2.5-14B 79.66, Llama-3-8B 66.60) and Meta's Llama 3.1 card (Llama-3-8B and Llama-3.1-8B, both 66.7 macro-avg) under three readings of the under-specified "MMLU (5-shot)" protocol (letter 5-shot, letter 0-shot, cloze 5-shot) and reports deltas and paired tests. Reuses a per-subject few-shot KV cache and batches letter-protocol items (`--batch-size`, default 8); `--verify-cache N` checks cached and batched scores against uncached scoring. |
| `exp2_prompt_sensitivity.py` | Runs several locally-available chat models through the same MMLU items under different prompt framings (direct vs. chain-of-thought) and decoding settings (greedy vs. temperature=0.7), to quantify score variance from methodology alone. |
| `exp3_base_vs_chat.py` | Compares the benchmarked base checkpoint (exp1) against its publicly-deployable chat/instruct variant (what most practitioners actually pull), on the same item sample. |
| `analysis/aggregate.py` | Combines all raw result JSON into one tidy `results/combined_results.csv`. |
| `analysis/plots.py` | Generates the matplotlib figures in `figures/` from that CSV. |
| `analysis/meta_evals.py` | Downloads Meta's released per-item MMLU details for Llama-3.1-8B (gated dataset `meta-llama/Llama-3.1-8B-evals`; accept its license on the Hub first), recomputes Meta's micro/macro `acc_char`, checks that Meta's prompts equal ours, and reports per-item agreement with our Llama-3.1-8B run. Writes `paper/generated/meta_numbers.tex`; Meta's data stays in the HF cache and is never copied into this repo. |
| `run_when_idle.sh` | Waits until no other large GPU job is running, then runs the remaining exp1 (MPS) and exp2 (Ollama) steps in parallel and regenerates the paper tables. `--now` skips the idle check. |
| `analysis/paper_tables.py` | Builds the paper's tables (`tab_repro.tex`, `tab_protocols.tex`, appendix `tab_subjects.tex`), `numbers.tex` macros and `fig_protocols.pdf` in `paper/generated/` directly from the per-item JSONL files (paired by item `uid`; McNemar + Bonferroni, Wilson CIs, Kendall tau vs. official ranking). |

### Setup

```bash
pip install -r requirements.txt

# Install Ollama (https://ollama.com) and pull the chat models used by exp2/exp3:
ollama pull qwen2.5:1.5b-instruct
ollama pull qwen2.5:7b-instruct
ollama pull llama3.1:8b
ollama pull qwen3:8b
ollama serve   # if not already running as a background service
```

`exp1_reproduction.py` downloads `Qwen/Qwen3-8B-Base` directly from Hugging
Face on first run (~16GB) via `transformers` — no Ollama needed for that one,
since Ollama only distributes the chat-tuned variant, not the base
checkpoint the vendor's number refers to (this mismatch is itself one of
the study's findings).

### Running

```bash
cd experiments
./run_all.sh
```

or run/tune each stage individually, e.g. to do a quick, low-cost check
before committing to the full sample size:

```bash
python3 exp1_reproduction.py --n-per-subject 5   # faster, noisier
python3 exp2_prompt_sensitivity.py --n-per-subject 5 --models qwen2.5:7b-instruct
python3 exp2_prompt_sensitivity.py --limit 20 --decoding-variants greedy --models qwen3:8b  # smoke test
python3 analysis/aggregate.py && python3 analysis/plots.py && python3 analysis/paper_tables.py
python3 analysis/meta_evals.py
```

### Expected runtime (measured, not estimated)

- `exp1`, letter 5-shot on the full 14,042-item test set, Qwen3-8B-Base, bf16 on
  an M4 Max (MPS), batch size 8 with the per-subject prefix cache: **6,519 s
  (0.46 s/item, ~1.8 h per 8B model)**. Uncached, unbatched scoring measured
  1.66 s/item. The output projection is computed in fp32; with bf16 logits the
  top two letters tie exactly on >10% of items.
- Cloze runs the query once on the cached prefix, then scores each option's
  tokens: 0.71 s/item for Qwen3-8B-Base on MPS (vs 1.08 s/item re-running
  query + option per option); letter 0-shot is batched like letter 5-shot.
- Pilot (171 items, sharing the GPU with exp1): Ollama qwen3:8b direct
  ~1.8 s/item, CoT ~6 s/item (mean 275 generated tokens).
- `exp2`/`exp3` use Ollama (llama.cpp-backed, quantized) instead of raw
  `transformers`/MPS, and are much faster per item in practice — tens of
  minutes for direct prompting. CoT (up to 1,024 tokens per item) takes hours.
- Always launch long runs detached (`nohup ... & disown`, or equivalent) and
  tail the log file — do not run them attached to a session you might close.

### Outputs

- `results/*.json` — raw per-experiment results (per-item and per-subject where applicable).
- `results/exp2_items.jsonl` — every exp2 generation (raw text, parsed letter, truncation flag). Exp2 resumes from this file if interrupted; delete it to start fresh. The exp2 summary reports unparsed/truncated rates per condition and a paired McNemar test for direct vs. CoT.
- `results/combined_results.csv` — tidy table across all three experiments.
- `figures/*.png` — reproduction-vs-official comparison, per-model/condition accuracy grid, a "min-max spread per model" chart, and the base-vs-chat comparison.

Results/figures are `.gitignore`d by default since they're regenerated by
running the scripts; remove those lines from `.gitignore` (or `git add -f`)
if you want to commit actual run output alongside the code.

## Repo layout

```
.
├── README.md
├── requirements.txt
└── experiments/
    ├── common/            # shared MMLU loading, Ollama client, scoring utils
    ├── exp1_reproduction.py
    ├── exp2_prompt_sensitivity.py
    ├── exp3_base_vs_chat.py
    ├── analysis/
    ├── run_all.sh
    ├── results/           # generated
    └── figures/           # generated
```
