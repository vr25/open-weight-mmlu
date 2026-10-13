"""Loads all raw experiment JSON outputs into a single tidy CSV for analysis."""
import json
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_exp1():
    """One row per (model, protocol). Reads the multi-protocol exp1_<model>.json
    files, plus the legacy single-protocol exp1_reproduction.json if present."""
    rows = []
    for path in sorted(RESULTS_DIR.glob("exp1_*.json")):
        d = json.loads(path.read_text())
        if path.name == "exp1_reproduction.json":
            if any(RESULTS_DIR.glob(f"exp1_{d['model_id'].replace('/', '__')}.json")):
                continue  # superseded by the multi-protocol run
            protos = {"letter_5shot": {"total": d["total_items"], "accuracy": d["accuracy"],
                                       "ci95": d["ci95"]}}
        else:
            protos = d["protocols"]
        for name, e in protos.items():
            rows.append({
                "experiment": "exp1",
                "model": d["model_id"],
                "condition": name,
                "n": e["total"],
                "accuracy": e["accuracy"],
                "ci_lo": e["ci95"][0],
                "ci_hi": e["ci95"][1],
                "reference_accuracy": d.get("official_reported_accuracy"),
            })
    return rows


def load_exp2():
    path = RESULTS_DIR / "exp2_prompt_sensitivity.json"
    if not path.exists():
        return []
    d = json.loads(path.read_text())
    rows = []
    for r in d.get("conditions", {}).values():
        rows.append({
            "experiment": "exp2",
            "model": r["model"],
            "condition": f"{r['prompt_variant']}|{r['decoding_variant']}",
            "n": r["total"],
            "accuracy": r["accuracy"],
            "ci_lo": r["ci95"][0],
            "ci_hi": r["ci95"][1],
            "reference_accuracy": None,
        })
    return rows


def load_exp3():
    path = RESULTS_DIR / "exp3_base_vs_chat.json"
    if not path.exists():
        return []
    d = json.loads(path.read_text())
    return [{
        "experiment": "exp3",
        "model": d["chat_model"],
        "condition": "chat_generate_parse",
        "n": d["total_items"],
        "accuracy": d["accuracy"],
        "ci_lo": d["ci95"][0],
        "ci_hi": d["ci95"][1],
        "reference_accuracy": None,
    }]


def main():
    rows = load_exp1() + load_exp2() + load_exp3()
    if not rows:
        print("No result files found in experiments/results/ yet -- run exp1/2/3 first.")
        return
    df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "combined_results.csv"
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved combined table to {out_path}")


if __name__ == "__main__":
    main()
