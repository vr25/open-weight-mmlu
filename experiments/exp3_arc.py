"""Experiment 3: ARC-Challenge (25-shot), where vendors disagree on the same model.

Meta reports Llama-3-8B at 79.4 (acc_char) while the Qwen2.5 report lists it at
59.3, both "ARC-C, 25-shot". We score all 1,165 four-option test items with
Meta's exact prompts (letter scoring) and with a cloze reading built from the
same 25 exemplars, plus single- and multi-turn direct prompting of the Ollama
instruct models.

Usage:
    python exp3_arc.py base --models meta-llama/Llama-3.1-8B
    python exp3_arc.py ollama --models llama3.1:8b --variants direct direct_mt
"""
import argparse
import gc
import json
import random
import time
from pathlib import Path

import torch

from common.arc_utils import LETTERS, cloze_prefix, cloze_query, letter_prefix, letter_query, load_arc
from common.ollama_client import chat, generate
from common.scoring import extract_letter_answer

RESULTS_DIR = Path(__file__).parent / "results"
DIRECT_INSTR = "Reply with only the letter (A, B, C, or D) of the correct option."

OFFICIAL_ARC = {
    "meta-llama/Meta-Llama-3-8B": [
        {"value": 79.4, "source": "Meta Llama 3.1 model card (HF meta-llama/Llama-3.1-8B), base models table"},
        {"value": 59.3, "source": "Qwen2.5 Technical Report (arXiv:2412.15115), Table 4"}],
    "meta-llama/Llama-3.1-8B": [
        {"value": 79.7, "source": "Meta Llama 3.1 model card (HF meta-llama/Llama-3.1-8B), base models table"}],
    "Qwen/Qwen2.5-7B": [
        {"value": 63.7, "source": "Qwen2.5 Technical Report (arXiv:2412.15115), Table 4"}],
    "Qwen/Qwen3-8B-Base": [],
}


def _done(path, key):
    if not path.exists():
        return set()
    return {tuple(json.loads(l)[k] for k in key) for l in path.read_text().splitlines() if l.strip()}


def run_base(args, items):
    from exp1_reproduction import PrefixCachedScorer, load_model, pick_device, slug
    device = args.device or pick_device()
    for model_id in args.models:
        path = RESULTS_DIR / f"exp3_arc_{slug(model_id)}_items.jsonl"
        done = _done(path, ("uid",))
        todo = [it for it in items if (it.uid,) not in done]
        print(f"{model_id}: {len(done)} done, {len(todo)} to score", flush=True)
        if not todo:
            continue
        model, tok = load_model(model_id, device)
        sc = PrefixCachedScorer(model, tok, device)
        t0 = time.time()
        with path.open("a") as f:
            for n, it in enumerate(todo, 1):
                sc.set_prefix(letter_prefix(it))
                lps = sc.letter_scores(letter_query(it))
                sc.set_prefix(cloze_prefix(it))
                conts = [" " + c.strip() for c in it.choices]
                cl = sc.continuation_scores_batch(cloze_query(it), conts)
                norm = [s / max(1, len(c.encode())) for s, c in zip(cl, conts)]
                arg = lambda v: int(torch.tensor(v).argmax())
                f.write(json.dumps({"uid": it.uid, "gold": it.answer_idx, "letter": arg(lps),
                                    "cloze": arg(cl), "cloze_norm": arg(norm), "letter_lps": lps,
                                    "cloze_lps": cl, "meta_pred": it.meta_pred}) + "\n")
                f.flush()
                if n % 100 == 0:
                    el = time.time() - t0
                    print(f"[{n}/{len(todo)}] {el:.0f}s eta {el / n * (len(todo) - n):.0f}s", flush=True)
        print(f"cache fallbacks: {sc.fallbacks}", flush=True)
        del model, tok, sc
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()


def multiturn(it):
    msgs = []
    for q, c, a in it.shots:
        body = f"Question: {q}\n" + "\n".join(f"{L}. {x}" for L, x in zip(LETTERS, c))
        msgs += [{"role": "user", "content": (DIRECT_INSTR + "\n\n" if not msgs else "") + body},
                 {"role": "assistant", "content": LETTERS[a]}]
    msgs.append({"role": "user", "content": f"Question: {it.question}\n"
                 + "\n".join(f"{L}. {x}" for L, x in zip(LETTERS, it.choices))})
    return msgs


def run_ollama(args, items):
    path = RESULTS_DIR / "exp3_arc_ollama_items.jsonl"
    done = _done(path, ("model", "variant", "uid"))
    t0 = time.time()
    with path.open("a") as f:
        for model in args.models:
            for v in args.variants:
                for n, it in enumerate(items, 1):
                    if (model, v, it.uid) in done:
                        continue
                    try:
                        if v == "direct":
                            p = it.meta_prompt.rsplit("Answer:", 1)[0] + DIRECT_INSTR + "\nAnswer:"
                            out = generate(model, p, num_predict=32, seed=0)
                        else:
                            out = chat(model, multiturn(it), num_predict=32, seed=0)
                    except Exception as e:
                        out = {"response": f"[ERROR: {e}]", "done_reason": "error", "eval_count": None}
                    pred = extract_letter_answer(out["response"])
                    gold = LETTERS[it.answer_idx]
                    f.write(json.dumps({"model": model, "variant": v, "uid": it.uid, "gold": gold,
                                        "pred": pred, "correct": pred == gold,
                                        "done_reason": out["done_reason"],
                                        "response": out["response"]}) + "\n")
                    f.flush()
                    if n % 200 == 0:
                        print(f"[{time.time() - t0:.0f}s] {model} {v} {n}/{len(items)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["base", "ollama"])
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--variants", nargs="+", default=["direct", "direct_mt"])
    ap.add_argument("--n", type=int, default=500, help="random subset size (seed 0); 0 = all 1,165")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    items = load_arc()
    if args.n:
        random.Random(0).shuffle(items)
        items = items[: args.n]
    items = items[: args.limit]
    RESULTS_DIR.mkdir(exist_ok=True)
    (run_base if args.part == "base" else run_ollama)(args, items)


if __name__ == "__main__":
    main()
