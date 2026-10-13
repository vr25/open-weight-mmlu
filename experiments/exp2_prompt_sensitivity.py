"""Experiment 2: How much does eval methodology alone move a model's score?

For each Ollama chat model, run the same 5-shot MMLU items under multiple
prompt framings (direct vs. chain-of-thought) and decoding settings (greedy
vs. temperature 0.7), holding weights and items fixed.

Usage:
    python exp2_prompt_sensitivity.py --n-per-subject 5 --models qwen2.5:7b-instruct llama3.1:8b

Rigor notes:
- Every generation (raw text, parsed letter, truncation flag) is appended to
  results/exp2_items.jsonl, so every number is auditable and a killed run
  resumes where it stopped (already-completed items are skipped).
- Unparseable answers count as incorrect, but the unparsed and truncation
  rates are reported per condition so harness failures cannot masquerade as
  capability differences.
- Direct vs. CoT is compared with an exact paired McNemar test on greedy
  outputs over identical items.
"""
import argparse
import itertools
import json
import time
from pathlib import Path

from common.mmlu_utils import LETTERS, SUBJECT_DISPLAY, _format_question, format_mmlu_prompt, load_mmlu_subset
from common.ollama_client import chat, generate, is_server_up
from common.scoring import extract_letter_answer, mcnemar_exact, wilson_ci

# Instruct counterparts of the exp1 base checkpoints (see analysis/paper_tables.py
# BASE_TO_CHAT). Add "llama3:8b" to also pair Meta-Llama-3-8B.
DEFAULT_MODELS = [
    "qwen3:8b",
    "llama3.1:8b",
    "qwen2.5:7b-instruct",
]


def _stem(base_prompt):
    # Standard 5-shot prompt minus the trailing "Answer:" cue.
    return base_prompt.rsplit("Answer:", 1)[0]


DIRECT_INSTR = "Reply with only the letter (A, B, C, or D) of the correct option."


def _multiturn(item):
    # Each exemplar is its own user/assistant exchange, so the model sees the
    # expected one-letter reply format instead of a block of solved questions.
    header = (f"The following are multiple choice questions about "
              f"{SUBJECT_DISPLAY(item.subject)}. {DIRECT_INSTR}\n\n")
    msgs = []
    for ex in item.few_shot:
        q = _format_question(ex["question"], ex["choices"])
        msgs += [{"role": "user", "content": (header if not msgs else "") + q},
                 {"role": "assistant", "content": LETTERS[ex["answer"]]}]
    msgs.append({"role": "user", "content": (header if not msgs else "")
                 + _format_question(item.question, item.choices)})
    return msgs


# name -> (builder: MMLUItem -> prompt str or chat message list, token budget)
PROMPT_VARIANTS = {
    "direct": (
        lambda it: _stem(format_mmlu_prompt(it)) + DIRECT_INSTR + "\nAnswer:",
        32,
    ),
    "cot": (
        lambda it: _stem(format_mmlu_prompt(it)) + "Think step by step, then end your response with "
        "'Answer: X' where X is A, B, C, or D.",
        1024,
    ),
    # Robustness check for "direct": same exemplars, as multi-turn chat.
    "direct_mt": (_multiturn, 32),
}

DECODING_VARIANTS = {
    "greedy": {"temperature": 0.0, "n_samples": 1},
    "temp0.7": {"temperature": 0.7, "n_samples": 3},
}

RESULTS_DIR = Path(__file__).parent / "results"
ITEMS_PATH = RESULTS_DIR / "exp2_items.jsonl"
SUMMARY_PATH = RESULTS_DIR / "exp2_prompt_sensitivity.json"


def _key(r):
    # Keyed on uid so runs with different --n-per-subject share one file (subsets
    # are nested: the first k shuffled items per subject for the same seed).
    return (r["model"], r["prompt_variant"], r["decoding_variant"], r["uid"], r["sample"])


def load_done():
    done = {}
    if ITEMS_PATH.exists():
        for line in ITEMS_PATH.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[_key(r)] = r
    return done


def summarize(records, models, n_items):
    summary = {"conditions": {}, "paired_tests": {}}
    by_cond = {}
    for r in records:
        by_cond.setdefault((r["model"], r["prompt_variant"], r["decoding_variant"]), []).append(r)
    for (m, p, d), rs in sorted(by_cond.items()):
        k = sum(r["correct"] for r in rs)
        n = len(rs)
        lo, hi = wilson_ci(k, n)
        summary["conditions"][f"{m}|{p}|{d}"] = {
            "model": m, "prompt_variant": p, "decoding_variant": d,
            "correct": k, "total": n, "accuracy": k / n, "ci95": [lo, hi],
            "unparsed_rate": sum(r["pred"] is None for r in rs) / n,
            "truncated_rate": sum(r["done_reason"] == "length" for r in rs) / n,
            "complete": n == n_items * DECODING_VARIANTS[d]["n_samples"],
        }
    for m in models:
        a = {r["uid"]: r["correct"] for r in by_cond.get((m, "direct", "greedy"), [])}
        b = {r["uid"]: r["correct"] for r in by_cond.get((m, "cot", "greedy"), [])}
        common = sorted(set(a) & set(b))
        if not common:
            continue
        only_direct = sum(a[i] and not b[i] for i in common)
        only_cot = sum(b[i] and not a[i] for i in common)
        summary["paired_tests"][m] = {
            "n_items": len(common),
            "acc_direct": sum(a[i] for i in common) / len(common),
            "acc_cot": sum(b[i] for i in common) / len(common),
            "only_direct_correct": only_direct,
            "only_cot_correct": only_cot,
            "mcnemar_p": mcnemar_exact(only_direct, only_cot),
        }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-subject", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--prompt-variants", nargs="+", default=list(PROMPT_VARIANTS),
                    choices=list(PROMPT_VARIANTS))
    ap.add_argument(
        "--decoding-variants", nargs="+", default=list(DECODING_VARIANTS),
        choices=list(DECODING_VARIANTS),
        help="Use 'greedy' only for a faster run -- temp0.7 samples each item 3x.",
    )
    ap.add_argument("--limit", type=int, default=None,
                    help="Only use the first N items (quick smoke test).")
    args = ap.parse_args()

    if not is_server_up():
        raise SystemExit("Ollama server not reachable at localhost:11434 -- run `ollama serve` first.")

    items = load_mmlu_subset(n_per_subject=args.n_per_subject, seed=args.seed)
    if args.limit:
        items = items[: args.limit]
    print(f"Loaded {len(items)} MMLU items.", flush=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    done = load_done()
    if done:
        print(f"Resuming: {len(done)} generations already on disk.", flush=True)

    t0 = time.time()
    with ITEMS_PATH.open("a") as fout:
        for model, prompt_name, decoding_name in itertools.product(
            args.models, args.prompt_variants, args.decoding_variants
        ):
            build, budget = PROMPT_VARIANTS[prompt_name]
            dec = DECODING_VARIANTS[decoding_name]
            for idx, item in enumerate(items):
                prompt = build(item)
                gold = "ABCD"[item.answer_idx]
                for s in range(dec["n_samples"]):
                    key = (model, prompt_name, decoding_name, item.uid, s)
                    if key in done:
                        continue
                    try:
                        send = chat if isinstance(prompt, list) else generate
                        out = send(model, prompt, temperature=dec["temperature"],
                                   num_predict=budget, seed=args.seed + s)
                    except Exception as e:
                        out = {"response": f"[ERROR: {e}]", "done_reason": "error",
                               "eval_count": None, "prompt_eval_count": None}
                    pred = extract_letter_answer(out["response"])
                    rec = {
                        "model": model, "prompt_variant": prompt_name,
                        "decoding_variant": decoding_name, "item_idx": idx, "sample": s,
                        "uid": item.uid, "subject": item.subject, "gold": gold, "pred": pred,
                        "correct": pred == gold, "done_reason": out["done_reason"],
                        "eval_count": out["eval_count"], "response": out["response"],
                    }
                    fout.write(json.dumps(rec) + "\n")
                    fout.flush()
                    done[key] = rec
            uids = {it.uid for it in items}
            rs = [r for k, r in done.items() if k[:3] == (model, prompt_name, decoding_name) and k[3] in uids]
            acc = sum(r["correct"] for r in rs) / len(rs)
            unp = sum(r["pred"] is None for r in rs) / len(rs)
            trunc = sum(r["done_reason"] == "length" for r in rs) / len(rs)
            print(f"[{time.time()-t0:7.1f}s] {model}|{prompt_name}|{decoding_name:8s} "
                  f"acc={acc:.2%} unparsed={unp:.1%} truncated={trunc:.1%} (n={len(rs)})", flush=True)

    uid_set = {it.uid for it in items}
    records = [r for r in done.values() if r["model"] in args.models and r["uid"] in uid_set]
    summary = summarize(records, args.models, len(items))
    summary["config"] = {"n_per_subject": args.n_per_subject, "seed": args.seed,
                         "n_items": len(items), "limit": args.limit}
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

    print("\n=== Direct vs. CoT (greedy, paired McNemar) ===")
    for m, t in summary["paired_tests"].items():
        print(f"{m:25s} direct={t['acc_direct']:.2%} cot={t['acc_cot']:.2%} "
              f"delta={100*(t['acc_cot']-t['acc_direct']):+.1f}pp p={t['mcnemar_p']:.3g} (n={t['n_items']})")
    print(f"\nSaved to {SUMMARY_PATH} and {ITEMS_PATH}")


if __name__ == "__main__":
    main()
