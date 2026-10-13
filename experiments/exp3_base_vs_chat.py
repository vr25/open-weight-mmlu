"""Experiment 3: Base checkpoint vs. its publicly-deployable chat variant.

Motivation: vendors report flagship benchmark numbers for base/pretrained
checkpoints (see exp1), but the checkpoint most practitioners actually pull
and deploy (e.g. via Ollama) is the instruct/chat-tuned variant. This
experiment quantifies that gap on the *same* MMLU sample used in exp1,
so the two are directly comparable.

Usage:
    # Run exp1 first (or at least once) so its item sample/config is known,
    # then:
    python exp3_base_vs_chat.py --n-per-subject 20 --seed 0 --chat-model qwen3:8b
"""
import argparse
import json
import time
from pathlib import Path

from common.mmlu_utils import format_mmlu_prompt, load_mmlu_subset
from common.ollama_client import generate, is_server_up
from common.scoring import extract_letter_answer, wilson_ci

RESULTS_DIR = Path(__file__).parent / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-subject", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chat-model", default="qwen3:8b")
    args = ap.parse_args()

    if not is_server_up():
        raise SystemExit("Ollama server not reachable at localhost:11434 -- run `ollama serve` first.")

    items = load_mmlu_subset(n_per_subject=args.n_per_subject, seed=args.seed)
    print(f"Loaded {len(items)} MMLU items (same sampling config as exp1 for comparability).")

    t0 = time.time()
    correct = 0
    unparsed = 0
    per_item = []
    for i, item in enumerate(items):
        prompt = (format_mmlu_prompt(item).rsplit("Answer:", 1)[0]
                  + "Reply with only the letter (A, B, C, or D) of the correct option.\nAnswer:")
        try:
            out = generate(args.chat_model, prompt, temperature=0.0, num_predict=32)
        except Exception as e:
            out = {"response": f"[ERROR: {e}]", "done_reason": "error"}
        pred_letter = extract_letter_answer(out["response"])
        gold_letter = "ABCD"[item.answer_idx]
        correct += int(pred_letter == gold_letter)
        unparsed += int(pred_letter is None)
        per_item.append({"item_idx": i, "uid": item.uid, "subject": item.subject, "gold": gold_letter,
                         "pred": pred_letter, "correct": pred_letter == gold_letter,
                         "done_reason": out["done_reason"], "response": out["response"]})

        if (i + 1) % 100 == 0:
            print(f"[{i+1}/{len(items)}] running_acc={correct/(i+1):.2%}")

    total = len(items)
    acc = correct / total
    lo, hi = wilson_ci(correct, total)

    result = {
        "experiment": "exp3_base_vs_chat",
        "chat_model": args.chat_model,
        "n_per_subject": args.n_per_subject,
        "seed": args.seed,
        "total_items": total,
        "correct": correct,
        "accuracy": acc,
        "ci95": [lo, hi],
        "unparsed_rate": unparsed / total,
        "per_item": per_item,
        "runtime_seconds": time.time() - t0,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "exp3_base_vs_chat.json"
    out_path.write_text(json.dumps(result, indent=2))

    print("\n=== Experiment 3: Chat-variant accuracy ===")
    print(f"{args.chat_model}: {acc:.2%} (95% CI: {lo:.2%}-{hi:.2%}, n={total})")

    exp1_path = RESULTS_DIR / "exp1_Qwen__Qwen3-8B-Base.json"
    if exp1_path.exists():
        exp1 = json.loads(exp1_path.read_text())
        base_acc = exp1["protocols"]["letter_5shot"]["accuracy"]
        print(f"{exp1['model_id']} (base): {base_acc:.2%}")
        print(f"Delta (chat - base): {(acc - base_acc)*100:+.2f} pp")
    else:
        print("(Run exp1_reproduction.py to also see the base-model comparison.)")

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
