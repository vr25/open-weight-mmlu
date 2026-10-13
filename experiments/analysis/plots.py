"""Generates matplotlib figures from experiments/results/combined_results.csv.

Run aggregate.py first. Figures are saved as PNGs to experiments/figures/.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"

# Okabe-Ito colorblind-safe qualitative palette.
COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442"]

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
})


def load():
    path = RESULTS_DIR / "combined_results.csv"
    if not path.exists():
        raise SystemExit("No combined_results.csv -- run analysis/aggregate.py first.")
    return pd.read_csv(path)


def plot_exp1(df):
    sub = df[df["experiment"] == "exp1"]
    if sub.empty:
        return
    models = list(dict.fromkeys(sub["model"]))
    conditions = list(dict.fromkeys(sub["condition"]))
    series = ["official"] + conditions
    width = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(max(4.5, 1.8 * len(models) + 1.5), 4))
    for si, name in enumerate(series):
        xs, ys, lo, hi = [], [], [], []
        for mi, m in enumerate(models):
            msub = sub[sub["model"] == m]
            if name == "official":
                ref = msub["reference_accuracy"].dropna()
                if ref.empty:
                    continue
                xs.append(mi); ys.append(ref.iloc[0] * 100); lo.append(0); hi.append(0)
            else:
                row = msub[msub["condition"] == name]
                if row.empty:
                    continue
                r = row.iloc[0]
                xs.append(mi); ys.append(r["accuracy"] * 100)
                lo.append((r["accuracy"] - r["ci_lo"]) * 100); hi.append((r["ci_hi"] - r["accuracy"]) * 100)
        pos = [x - 0.4 + width * (si + 0.5) for x in xs]
        ax.bar(pos, ys, width, label=name, color=COLORS[si % len(COLORS)])
        if name != "official":
            ax.errorbar(pos, ys, yerr=[lo, hi], fmt="none", ecolor="black", capsize=2, lw=0.8)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.split("/")[-1] for m in models], fontsize=8)
    ax.set_ylabel("MMLU accuracy (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("Exp. 1: Official vs. reproduced MMLU", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "exp1_reproduction.png")
    plt.close(fig)


def plot_exp2(df):
    sub = df[df["experiment"] == "exp2"]
    if sub.empty:
        return
    models = sorted(sub["model"].unique())
    conditions = sorted(sub["condition"].unique())

    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(models)), 5))
    n_cond = len(conditions)
    width = 0.8 / n_cond
    x = range(len(models))

    for ci, cond in enumerate(conditions):
        vals = []
        for m in models:
            row = sub[(sub["model"] == m) & (sub["condition"] == cond)]
            vals.append(row["accuracy"].values[0] * 100 if not row.empty else 0)
        offsets = [xi + (ci - n_cond / 2) * width + width / 2 for xi in x]
        ax.bar(offsets, vals, width=width, label=cond, color=COLORS[ci % len(COLORS)])

    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("MMLU accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Exp. 2: Accuracy by model x prompt/decoding condition")
    ax.legend(fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.25))
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "exp2_prompt_sensitivity.png", bbox_inches="tight")
    plt.close(fig)


def plot_exp2_spread(df):
    """The 'trust-adjusted leaderboard' view: min-max accuracy range per model."""
    sub = df[df["experiment"] == "exp2"]
    if sub.empty:
        return
    grouped = sub.groupby("model")["accuracy"].agg(["min", "max"]).sort_values("max")
    fig, ax = plt.subplots(figsize=(6, 0.5 * len(grouped) + 1.5))
    y = range(len(grouped))
    for yi, (model, row) in zip(y, grouped.iterrows()):
        ax.plot([row["min"] * 100, row["max"] * 100], [yi, yi], color=COLORS[0], lw=2, zorder=1)
        ax.scatter([row["min"] * 100], [yi], color=COLORS[2], zorder=2, label="min" if yi == 0 else None)
        ax.scatter([row["max"] * 100], [yi], color=COLORS[3], zorder=2, label="max" if yi == 0 else None)
    ax.set_yticks(list(y))
    ax.set_yticklabels(grouped.index)
    ax.set_xlabel("MMLU accuracy (%) across tested eval conditions")
    ax.set_title("How much does eval-config choice alone move each model's score?")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "exp2_spread.png")
    plt.close(fig)


def plot_exp3(df):
    sub = df[df["experiment"].isin(["exp1", "exp3"])]
    if sub.empty or "exp3" not in sub["experiment"].values:
        return
    fig, ax = plt.subplots(figsize=(4.5, 4))
    labels, values, colors = [], [], []
    exp1_row = sub[(sub["experiment"] == "exp1") & (sub["model"] == "Qwen/Qwen3-8B-Base")
                   & (sub["condition"] == "letter_5shot")]
    exp3_row = sub[sub["experiment"] == "exp3"]
    if not exp1_row.empty:
        labels.append("Base checkpoint\n(loglikelihood scoring)")
        values.append(exp1_row.iloc[0]["accuracy"] * 100)
        colors.append(COLORS[0])
    if not exp3_row.empty:
        labels.append("Chat/instruct checkpoint\n(generate + parse)")
        values.append(exp3_row.iloc[0]["accuracy"] * 100)
        colors.append(COLORS[1])
    bars = ax.bar(labels, values, color=colors, width=0.55)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1, f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_ylabel("MMLU accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Exp. 3: Benchmarked checkpoint vs.\npublicly-deployable checkpoint")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "exp3_base_vs_chat.png")
    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(exist_ok=True)
    df = load()
    plot_exp1(df)
    plot_exp2(df)
    plot_exp2_spread(df)
    plot_exp3(df)
    print(f"Figures written to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
