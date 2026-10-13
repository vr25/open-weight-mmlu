"""Experiment 4: decompose the base -> deployed gap on the MMLU subset.

For each instruct checkpoint (bf16, transformers) we run greedy multi-turn
direct prompting with the model's own chat template -- the same messages
exp2 sends to the Q4 Ollama build (variant "direct_mt"). Together with exp1
(letter log-likelihood for base and instruct checkpoints) this gives the ladder
base loglik -> instruct loglik -> instruct bf16 generate -> instruct Q4 generate.

Usage:
    python exp4_instruct_bf16.py --models meta-llama/Llama-3.1-8B-Instruct
"""
import argparse
import gc
import json
import time
from pathlib import Path

import torch

from common.mmlu_utils import load_mmlu_subset
from common.scoring import extract_letter_answer
from exp1_reproduction import load_model, pick_device
from exp2_prompt_sensitivity import _multiturn

RESULTS_PATH = Path(__file__).parent / "results" / "exp4_instruct_items.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n-per-subject", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    device = pick_device()
    items = load_mmlu_subset(n_per_subject=args.n_per_subject, seed=args.seed)[: args.limit]
    done = set()
    if RESULTS_PATH.exists():
        done = {(r["model"], r["uid"]) for r in map(json.loads, RESULTS_PATH.read_text().splitlines())}
    for model_id in args.models:
        todo = [it for it in items if (model_id, it.uid) not in done]
        print(f"{model_id}: {len(todo)} to generate", flush=True)
        if not todo:
            continue
        model, tok = load_model(model_id, device)
        t0 = time.time()
        with RESULTS_PATH.open("a") as f:
            for n, it in enumerate(todo, 1):
                ids = tok.apply_chat_template(_multiturn(it), add_generation_prompt=True,
                                              return_tensors="pt", enable_thinking=False).to(device)
                with torch.no_grad():
                    out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=32,
                                         do_sample=False, pad_token_id=tok.eos_token_id)
                gen = out[0, ids.shape[1]:]
                text = tok.decode(gen, skip_special_tokens=True)
                pred = extract_letter_answer(text)
                gold = "ABCD"[it.answer_idx]
                trunc = len(gen) >= 32 and gen[-1].item() not in (tok.eos_token_id,)
                f.write(json.dumps({"model": model_id, "uid": it.uid, "subject": it.subject,
                                    "gold": gold, "pred": pred, "correct": pred == gold,
                                    "truncated": bool(trunc), "response": text}) + "\n")
                f.flush()
                if n % 100 == 0:
                    el = time.time() - t0
                    print(f"[{n}/{len(todo)}] {el:.0f}s eta {el / n * (len(todo) - n):.0f}s", flush=True)
        del model, tok
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
