"""Page-1 figure: one dot plot per benchmark, from vendor claim to deployed model.

Usage: python3 analysis/hero_fig.py   (writes paper/generated/fig_hero.pdf)
"""
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_tables import EXP_DIR, load_exp1, load_exp2, macro, read_jsonl  # noqa: E402

RD = EXP_DIR / "results"
OUT = EXP_DIR.parent / "paper" / "generated" / "fig_hero.pdf"

MODELS = [  # base, ollama tag, label, colour, marker, MMLU claim (value, aggregation), ARC claim
    ("meta-llama/Llama-3.1-8B", "llama3.1:8b", "Llama-3.1-8B", "#1f77b4", "o", (66.7, "macro"), 79.7),
    ("Qwen/Qwen2.5-7B", "qwen2.5:7b-instruct", "Qwen2.5-7B", "#d95f02", "s", (74.16, "micro"), 63.7),
]
ROWS = ["Vendor claim", "Base, letter", "Base, cloze (norm)", "Q4 chat, multi-turn", "Q4 chat, 1-turn"]


def mean(d, keep=None):
    v = [x for u, x in d.items() if keep is None or u in keep]
    return 100 * sum(v) / len(v)


def mmlu_points():
    from common.mmlu_utils import load_mmlu_subset
    subset = {it.uid for it in load_mmlu_subset(n_per_subject=20, seed=0)}
    exp1 = load_exp1(RD)
    exp2, _, _ = load_exp2(RD)
    pts = {}
    for base, tag, _, _, _, (claim, agg), _ in MODELS:
        full = exp1[base]["letter_5shot"]
        pts[base] = [claim, 100 * macro(full) if agg == "macro" else mean(full),
                     mean(exp1[base]["cloze_5shot_norm"], subset),
                     mean(exp2[tag]["chat_direct_mt"], subset), mean(exp2[tag]["chat_direct"], subset)]
    return pts


def arc_points():
    from common.arc_utils import load_arc
    items = load_arc()
    random.Random(0).shuffle(items)
    sub = {it.uid for it in items[:500]}
    oll = {}
    for r in read_jsonl(RD / "exp3_arc_ollama_items.jsonl"):
        oll.setdefault((r["model"], r["variant"]), {})[r["uid"]] = bool(r["correct"])
    pts = {}
    for base, tag, _, _, _, _, claim in MODELS:
        recs = [r for r in read_jsonl(RD / f"exp3_arc_{base.replace('/', '__')}_items.jsonl") if r["uid"] in sub]
        acc = lambda k: 100 * sum(r[k] == r["gold"] for r in recs) / len(recs)
        pts[base] = [claim, acc("letter"), acc("cloze_norm"),
                     mean(oll[(tag, "direct_mt")], sub), mean(oll[(tag, "direct")], sub)]
    return pts


def main():
    plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.linewidth": 0.5,
                         "xtick.major.width": 0.5, "ytick.major.width": 0, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(3.15, 1.85), sharey=True)
    y = list(range(len(ROWS)))[::-1]
    for ax, (title, pts) in zip(axes, [("MMLU, 5-shot", mmlu_points()), ("ARC-C, 25-shot", arc_points())]):
        for yy in y:
            ax.axhline(yy, color="0.92", lw=0.5, zorder=0)
        ax.axhspan(y[0] - 0.5, y[0] + 0.5, color="0.95", zorder=0)
        ax.axhspan(y[-2] - 0.5, y[-1] - 0.5, color="#fdecea", zorder=0)
        for base, _, label, col, mk, _, _ in MODELS:
            vals = pts[base]
            for j, v in enumerate(vals):
                ax.scatter(v, y[j] - (0.16 if j == 0 and ax is axes[1] else 0), s=16, color="white" if j == 0 else col, edgecolor=col,
                           marker=mk, lw=0.9, zorder=3, label=label if (j == 1 and ax is axes[0]) else None)
        ax.set_title(title, fontsize=7.5, pad=3)
        ax.set_xlim(20, 97)
        ax.set_xticks([25, 50, 75])
        ax.tick_params(axis="x", length=2, pad=1)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
    # The cross-vendor disagreement on the same checkpoint (ARC-C, Llama-3-8B).
    ax = axes[1]
    ax.scatter([79.4, 59.3], [y[0] + 0.2, y[0] + 0.2], s=16, marker="^", color="white", edgecolor="0.35", lw=0.9, zorder=3)
    ax.annotate("", xy=(61.3, y[0] + 0.2), xytext=(77.4, y[0] + 0.2),
                arrowprops=dict(arrowstyle="<->", lw=0.6, color="0.35", shrinkA=0, shrinkB=0))
    ax.text(69.3, y[0] + 0.27, "20 pp", ha="center", va="bottom", fontsize=5.5, color="0.25")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(ROWS)
    axes[0].set_ylim(-0.6, len(ROWS) - 0.15)
    for a in axes:
        a.set_xlabel("Accuracy (%)", fontsize=6.5, labelpad=1)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], marker="^", ls="", mfc="white", mec="0.35", ms=4))
    labels.append("Llama-3-8B (claims)")
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=6,
               bbox_to_anchor=(0.6, 1.02), handletextpad=0.2, columnspacing=0.8)
    fig.tight_layout(pad=0.2, w_pad=0.6, rect=(0, 0, 1, 0.93))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.01)
    for name, p in [("MMLU", mmlu_points()), ("ARC", arc_points())]:
        print(name, {k.split("/")[1]: [round(v, 1) for v in vs] for k, vs in p.items()})
    print("wrote", OUT)


if __name__ == "__main__":
    main()
