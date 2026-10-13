"""Builds every number, table and figure used in the paper from raw per-item
results, so the paper never contains hand-copied values.

Inputs (experiments/results/):
  exp1_<model>_items.jsonl   per-item base-model log-likelihood scores (exp1)
  exp2_items.jsonl           per-item instruct-model generations (exp2)
Outputs (paper/generated/ by default):
  tab_repro.tex      official vs. reproduced (full test set, letter 5-shot)
  tab_protocols.tex  protocol / deployment conditions on the paired subset
  numbers.tex        \\newcommand macros for numbers quoted in the text
  tab_subjects.tex   per-subject letter 5-shot accuracy (appendix)
  fig_protocols.pdf  per-model accuracy under each condition

Usage:
    python analysis/paper_tables.py [--results-dir DIR] [--out-dir DIR]
"""
import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXP_DIR))

from common.scoring import extract_letter_answer_lenient, mcnemar_exact, wilson_ci  # noqa: E402

# Base checkpoint -> (display name, Ollama instruct tag it is paired with)
BASE_TO_CHAT = {
    "Qwen/Qwen3-8B-Base": ("Qwen3-8B", "qwen3:8b"),
    "Qwen/Qwen2.5-7B": ("Qwen2.5-7B", "qwen2.5:7b-instruct"),
    "meta-llama/Llama-3.1-8B": ("Llama-3.1-8B", "llama3.1:8b"),
    "meta-llama/Meta-Llama-3-8B": ("Llama-3-8B", "llama3:8b"),
}
SOURCE_SHORT = {"Qwen3 Technical Report": "Qwen3 TR", "Meta Llama 3.1 model card": "Meta card"}

CONDITIONS = [
    ("letter_5shot", "Letter 5-shot"),
    ("letter_0shot", "Letter 0-shot"),
    ("cloze_5shot", "Cloze"),
    ("cloze_5shot_norm", "Cloze (norm)"),
    ("chat_direct", "Instr.\\ direct"),
    ("chat_cot", "Instr.\\ CoT"),
]
N_COMPARISONS = len(CONDITIONS) - 1  # every condition vs. Letter 5-shot


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def short_source(src):
    return next((v for k, v in SOURCE_SHORT.items() if src.startswith(k)), src)


DIGITS = "Zero One Two Three Four Five Six Seven Eight Nine".split()


def alpha(s):
    """LaTeX-safe macro fragment: digits spelled out, other non-letters dropped."""
    return "".join(DIGITS[int(ch)] if ch.isdigit() else ch for ch in s if ch.isalnum())


def acc_stats(ok):
    k, n = sum(ok.values()), len(ok)
    lo, hi = wilson_ci(k, n)
    return {"k": k, "n": n, "acc": k / n, "lo": lo, "hi": hi}


def macro(ok):
    subj = {}
    for uid, v in ok.items():
        subj.setdefault(uid.rsplit(":", 1)[0], []).append(v)
    return sum(sum(v) / len(v) for v in subj.values()) / len(subj)


def kendall_tau(a, b):
    """Kendall tau-a between two {name: score} dicts over shared keys."""
    keys = sorted(set(a) & set(b))
    s = sum((a[x] - a[y]) * (b[x] - b[y]) > 0 for x, y in combinations(keys, 2)) \
        - sum((a[x] - a[y]) * (b[x] - b[y]) < 0 for x, y in combinations(keys, 2))
    pairs = len(keys) * (len(keys) - 1) / 2
    return s / pairs if pairs else float("nan")


def load_exp1(results_dir):
    """{base_model: {condition: {uid: correct}}}"""
    out = {}
    for base in BASE_TO_CHAT:
        recs = read_jsonl(results_dir / f"exp1_{base.replace('/', '__')}_items.jsonl")
        conds = {}
        for r in recs:
            conds.setdefault(r["protocol"], {})[r["uid"]] = r["pred"] == r["gold"]
            if "pred_norm" in r:
                conds.setdefault(r["protocol"] + "_norm", {})[r["uid"]] = r["pred_norm"] == r["gold"]
        if conds:
            out[base] = conds
    return out


def load_exp2(results_dir):
    """Greedy only. Returns ({tag: {cond: {uid: correct}}}, {tag: {cond: unparsed_rate}},
    {tag: {cond: {uid: correct under lenient last-letter parsing}}})."""
    ok, unp, lenient = {}, {}, {}
    for r in read_jsonl(results_dir / "exp2_items.jsonl"):
        if r["decoding_variant"] != "greedy":
            continue
        cond = "chat_" + r["prompt_variant"]
        ok.setdefault(r["model"], {}).setdefault(cond, {})[r["uid"]] = bool(r["correct"])
        unp.setdefault(r["model"], {}).setdefault(cond, []).append(r["pred"] is None)
        lp = extract_letter_answer_lenient(r.get("response", "")) if r["pred"] is None else r["pred"]
        lenient.setdefault(r["model"], {}).setdefault(cond, {})[r["uid"]] = lp == r["gold"]
    return ok, {m: {c: sum(v) / len(v) for c, v in cs.items()} for m, cs in unp.items()}, lenient


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}"


BASE_TO_INSTRUCT = {
    "Qwen/Qwen3-8B-Base": "Qwen/Qwen3-8B",
    "Qwen/Qwen2.5-7B": "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B": "meta-llama/Llama-3.1-8B-Instruct",
}
ARC_CLAIMS = {  # (source, value) as printed; see exp3_arc.OFFICIAL_ARC
    "meta-llama/Meta-Llama-3-8B": {"Meta": 79.4, "Qwen": 59.3},
    "meta-llama/Llama-3.1-8B": {"Meta": 79.7},
    "Qwen/Qwen2.5-7B": {"Qwen": 63.7},
}


def _sig(a, b):
    common = set(a) & set(b)
    x = sum(a[u] and not b[u] for u in common)
    y = sum(b[u] and not a[u] for u in common)
    return mcnemar_exact(x, y)


def _cell(cur, prev):
    v = pct(sum(cur.values()) / len(cur))
    if prev is not None and _sig(prev, cur) < 0.05:
        v += "$^{\\dagger}$"
    return v


def ladder_and_arc(args, exp1, exp2, subset, nums):
    """Appendix/table: base -> deployed decomposition on MMLU, and ARC-C."""
    rd = args.results_dir
    gen = {}
    for r in read_jsonl(rd / "exp4_instruct_items.jsonl"):
        if r["uid"] in subset:
            gen.setdefault(r["model"], {})[r["uid"]] = bool(r["correct"])
    rows = []
    for base, inst in BASE_TO_INSTRUCT.items():
        name, chat = BASE_TO_CHAT[base]
        il = {}
        for r in read_jsonl(rd / f"exp1_{inst.replace('/', '__')}_items.jsonl"):
            if r["protocol"] == "letter_5shot" and r["uid"] in subset:
                il[r["uid"]] = r["pred"] == r["gold"]
        steps = [
            {u: v for u, v in exp1.get(base, {}).get("letter_5shot", {}).items() if u in subset},
            il, gen.get(inst, {}),
            {u: v for u, v in exp2.get(chat, {}).get("chat_direct_mt", {}).items() if u in subset},
            {u: v for u, v in exp2.get(chat, {}).get("chat_direct", {}).items() if u in subset},
        ]
        if not all(steps[:2]):
            continue
        cells, prev = [], None
        for j, d in enumerate(steps):
            if not d:
                cells.append("--")
                continue
            # the last column (single-turn) is compared with Q4 multi-turn
            cells.append(_cell(d, prev))
            if j < 4:
                prev = d
            nums[f"lad{alpha(name)}{'ABCDE'[j]}"] = pct(sum(d.values()) / len(d))
        rows.append(" & ".join([name, *cells]) + " \\\\")
    if rows:
        (args.out_dir / "tab_ladder.tex").write_text("\n".join([
            "\\begin{table}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{3pt}",
            "\\resizebox{\\columnwidth}{!}{%", "\\begin{tabular}{lrrrrr}", "\\toprule",
            " & \\multicolumn{2}{c}{Log-lik., bf16} & \\multicolumn{3}{c}{Generate, direct} \\\\",
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}",
            "Model & Base & Instr. & bf16 & Q4 & Q4, 1-turn \\\\", "\\midrule",
            *rows, "\\bottomrule", "\\end{tabular}}",
            f"\\caption{{From vendor claim to deployment on the {len(subset):,}-item MMLU subset. "
            "Each column changes one factor relative to the one before: instruction tuning, "
            "log-likelihood scoring $\\rightarrow$ generation (multi-turn chat, greedy), bf16 "
            "$\\rightarrow$ Ollama Q4\\_K\\_M, and multi-turn $\\rightarrow$ single-turn exemplars. "
            "$^{\\dagger}$: differs from the previous column (paired exact McNemar, $p<0.05$).}",
            "\\label{tab:ladder}", "\\end{table}", ""]))

    # ---- ARC-Challenge -------------------------------------------------------
    ollama = {}
    for r in read_jsonl(rd / "exp3_arc_ollama_items.jsonl"):
        ollama.setdefault((r["model"], r["variant"]), {})[r["uid"]] = bool(r["correct"])
    import random
    from common.arc_utils import load_arc
    arc_items = load_arc()
    random.Random(0).shuffle(arc_items)
    arc_subset = {it.uid for it in arc_items[:500]}
    rows, n_arc = [], 0
    for base in ["meta-llama/Meta-Llama-3-8B", "meta-llama/Llama-3.1-8B", "Qwen/Qwen2.5-7B",
                 "Qwen/Qwen3-8B-Base"]:
        recs = {r["uid"]: r for r in read_jsonl(rd / f"exp3_arc_{base.replace('/', '__')}_items.jsonl")
                if r["uid"] in arc_subset}
        if not recs:
            continue
        name, chat = BASE_TO_CHAT[base]
        n_arc = max(n_arc, len(recs))
        acc = {k: sum(r[k] == r["gold"] for r in recs.values()) / len(recs)
               for k in ("letter", "cloze", "cloze_norm")}
        agree = sum(r["letter"] == r["meta_pred"] for r in recs.values()) / len(recs)
        claims = ARC_CLAIMS.get(base, {})
        o1, o2 = ollama.get((chat, "direct"), {}), ollama.get((chat, "direct_mt"), {})
        ou = set(recs)
        f = lambda d: pct(sum(d[u] for u in ou if u in d) / max(1, sum(u in d for u in ou))) \
            if any(u in d for u in ou) else "--"
        for k, v in acc.items():
            nums[f"arc{alpha(k)}{alpha(name)}"] = pct(v)
        if base == "meta-llama/Llama-3.1-8B":  # Meta's released predictions are for this model
            nums["arcMetaAgree"] = pct(agree)
        rows.append(" & ".join([
            name, f"{claims['Meta']:.1f}" if "Meta" in claims else "--",
            f"{claims['Qwen']:.1f}" if "Qwen" in claims else "--",
            pct(acc["letter"]), pct(acc["cloze"]), pct(acc["cloze_norm"]), f(o1), f(o2)]) + " \\\\")
    if rows:
        nums["arcN"] = f"{n_arc:,}"
        (args.out_dir / "tab_arc.tex").write_text("\n".join([
            "\\begin{table}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{3pt}",
            "\\resizebox{\\columnwidth}{!}{%", "\\begin{tabular}{lrrrrrrr}", "\\toprule",
            " & \\multicolumn{2}{c}{Claimed} & \\multicolumn{3}{c}{Base, ours} & \\multicolumn{2}{c}{Q4 instr.} \\\\",
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}\\cmidrule(lr){7-8}",
            "Model & Meta & Qwen & Letter & Cloze & Norm & 1-turn & Multi \\\\", "\\midrule",
            *rows, "\\bottomrule", "\\end{tabular}}",
            f"\\caption{{ARC-Challenge, 25-shot, on {n_arc:,} random four-option test items, using the "
            "exemplars in Meta's released prompts. Claims: Meta model card and Qwen2.5 report "
            "(Table~4), both labelled ``ARC-C, 25-shot''.}",
            "\\label{tab:arc}", "\\end{table}", ""]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=EXP_DIR / "results")
    ap.add_argument("--out-dir", type=Path, default=EXP_DIR.parent / "paper" / "generated")
    ap.add_argument("--subset-n-per-subject", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    from common.mmlu_utils import load_mmlu_subset
    from exp1_reproduction import OFFICIAL

    exp1 = load_exp1(args.results_dir)
    exp2, unparsed, lenient = load_exp2(args.results_dir)
    if not exp1:
        raise SystemExit(f"No exp1 item files in {args.results_dir}")
    subset = {it.uid for it in load_mmlu_subset(n_per_subject=args.subset_n_per_subject, seed=args.seed)}
    n_full = len(load_mmlu_subset(n_per_subject=0, seed=args.seed))
    nums = {}

    # ---- Table 1: reproduction on the full test set -------------------------
    rows = []
    for base, conds in exp1.items():
        name = BASE_TO_CHAT[base][0]
        full = conds.get("letter_5shot", {})
        if not full:
            continue
        st, mac = acc_stats(full), macro(full)
        partial = "" if st["n"] == n_full else "$^{\\ddagger}$"
        nums[f"acc{alpha(name)}"] = pct(st["acc"], 2)
        nums[f"macro{alpha(name)}"] = pct(mac, 2)
        for j, c in enumerate(OFFICIAL.get(base, [])):
            is_macro = c["agg"] == "macro"
            ours = mac if is_macro else st["acc"]
            delta = 100 * ours - c["value"]
            outside = not is_macro and not (st["lo"] <= c["value"] / 100 <= st["hi"])
            src = short_source(c["source"])
            rows.append(" & ".join([
                (name + partial) if j == 0 else "",
                src + (" (M)" if is_macro else ""),
                f"{c['value']:.2f}",
                pct(ours, 2),
                "" if is_macro else f"{pct(st['lo'])}--{pct(st['hi'])}",
                f"{delta:+.2f}" + ("$^{*}$" if outside else ""),
            ]) + " \\\\")
            nums[f"delta{alpha(name)}{alpha(src)}"] = f"{delta:+.2f}"
    (args.out_dir / "tab_repro.tex").write_text("\n".join([
        "\\begin{table}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{3pt}",
        "\\resizebox{\\columnwidth}{!}{%", "\\begin{tabular}{llrrcr}", "\\toprule",
        "Model & Source & Off. & Ours & 95\\% CI & $\\Delta$ \\\\", "\\midrule",
        *rows, "\\bottomrule", "\\end{tabular}}",
        f"\\caption{{Official vs.\\ reproduced MMLU (5-shot, letter log-likelihood, bf16) on all "
        f"{n_full:,} test items; $\\Delta$ in points. (M): source states macro averaging, so we "
        "compare subject-averaged accuracy; other sources state no aggregation and are compared "
        "with micro accuracy (95\\% Wilson CI). $^{*}$: official value outside our CI."
        + (" $^{\\ddagger}$: incomplete run." if "ddagger" in "".join(rows) else "") + "}",
        "\\label{tab:repro}", "\\end{table}", ""]))

    # ---- Table 2: paired subset across protocols + deployment ---------------
    per_model, rows, worst_unp, n_sig, n_tests, have_chat = {}, [], 0.0, 0, 0, False
    partial_n = set()
    max_lenient_gain = 0.0
    for base, conds in exp1.items():
        name, chat = BASE_TO_CHAT[base]
        allc = {**conds, **exp2.get(chat, {})}
        allc = {k: {u: v for u, v in d.items() if u in subset} for k, d in allc.items()}
        ref = allc.get("letter_5shot", {})
        cells, accs = [], {}
        for key, _ in CONDITIONS:
            d = allc.get(key)
            if not d:
                cells.append("--")
                continue
            accs[key] = acc_stats(d)["acc"]
            mark = ""
            if len(d) < len(subset):
                partial_n.add(len(d))
                mark = "$^{\\S}$"
            if key != "letter_5shot" and ref:
                common = set(ref) & set(d)
                b = sum(ref[u] and not d[u] for u in common)
                c = sum(d[u] and not ref[u] for u in common)
                n_tests += 1
                if mcnemar_exact(b, c) < 0.05 / N_COMPARISONS:
                    mark, n_sig = mark + "$^{\\dagger}$", n_sig + 1
            cells.append(pct(accs[key]) + mark)
        for key in ("chat_direct", "chat_cot"):
            strict_ok = allc.get(key)
            len_ok = {u: v for u, v in lenient.get(chat, {}).get(key, {}).items() if u in subset}
            if strict_ok and len_ok:
                gain = (sum(len_ok.values()) - sum(strict_ok.values())) / len(strict_ok)
                max_lenient_gain = max(max_lenient_gain, gain)
            if key in unparsed.get(chat, {}):
                have_chat = True
                worst_unp = max(worst_unp, unparsed[chat][key])
        per_model[name] = accs
        rows.append(" & ".join([name, *cells]) + " \\\\")
        if len(accs) > 1:
            nums[f"spread{alpha(name)}"] = f"{100 * (max(accs.values()) - min(accs.values())):.1f}"
    (args.out_dir / "tab_protocols.tex").write_text("\n".join([
        "\\begin{table*}[t]", "\\centering", "\\small",
        "\\begin{tabular}{l" + "r" * len(CONDITIONS) + "}", "\\toprule",
        " & \\multicolumn{4}{c}{Base checkpoint, bf16, log-likelihood}"
        " & \\multicolumn{2}{c}{Instruct, Ollama Q4, generate + parse} \\\\",
        "\\cmidrule(lr){2-5}\\cmidrule(lr){6-7}",
        "Model & " + " & ".join(h for _, h in CONDITIONS) + " \\\\", "\\midrule",
        *rows, "\\bottomrule", "\\end{tabular}",
        f"\\caption{{MMLU accuracy (\\%) on the same {len(subset):,} items "
        f"({args.subset_n_per_subject} per subject) under different readings of ``5-shot MMLU'' "
        "(left) and the instruct checkpoint practitioners typically deploy (right; greedy). "
        "$^{\\dagger}$: differs from Letter 5-shot (paired exact McNemar, Bonferroni over "
        f"{N_COMPARISONS} comparisons, $p<0.05$)."
        + (f" Max.\\ unparsed rate for instruct outputs: {pct(worst_unp)}\\%; lenient last-letter "
           f"parsing raises instruct accuracy by at most {pct(max_lenient_gain)} points." if have_chat else "")
        + (" $^{\\S}$: scored on a nested part of the subset ($n$\\,=\\,"
           + ", ".join(f"{n:,}" for n in sorted(partial_n)) + "); paired tests use shared items."
           if partial_n else "")
        + "}",
        "\\label{tab:protocols}", "\\end{table*}", ""]))

    # ---- Appendix: single-turn vs. multi-turn direct prompting ---------------
    mt_rows = []
    for base in exp1:
        name, chat = BASE_TO_CHAT[base]
        e = {k: {u: v for u, v in d.items() if u in subset} for k, d in exp2.get(chat, {}).items()}
        st_, mt_ = e.get("chat_direct", {}), e.get("chat_direct_mt", {})
        if not st_ or not mt_:
            continue
        common = set(st_) & set(mt_)
        b = sum(st_[u] and not mt_[u] for u in common)
        c = sum(mt_[u] and not st_[u] for u in common)
        a_st = sum(st_[u] for u in common) / len(common)
        a_mt = sum(mt_[u] for u in common) / len(common)
        u_st, u_mt = unparsed[chat]["chat_direct"], unparsed[chat]["chat_direct_mt"]
        p_ = mcnemar_exact(b, c)
        nums[f"mtAcc{alpha(name)}"] = pct(a_mt)
        nums[f"mtUnp{alpha(name)}"] = pct(u_mt)
        nums[f"stAcc{alpha(name)}"] = pct(a_st)
        nums[f"stUnp{alpha(name)}"] = pct(u_st)
        mt_rows.append(" & ".join([
            f"\\texttt{{{chat}}}", f"{pct(a_st)} ({pct(u_st)})", f"{pct(a_mt)} ({pct(u_mt)})",
            f"{100 * (a_mt - a_st):+.1f}", f"{p_:.1g}" if p_ >= 1e-3 else "$<$0.001",
            f"{len(common):,}"]) + " \\\\")
    if mt_rows:
        (args.out_dir / "tab_direct_mt.tex").write_text("\n".join([
            "\\begin{table}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{3pt}",
            "\\resizebox{\\columnwidth}{!}{%", "\\begin{tabular}{lrrrrr}", "\\toprule",
            "Model & Single-turn & Multi-turn & $\\Delta$ & $p$ & $n$ \\\\", "\\midrule",
            *mt_rows, "\\bottomrule", "\\end{tabular}}",
            "\\caption{Direct prompting of the Q4 instruct models with the five exemplars in one "
            "user turn (as in \\cref{tab:protocols}) vs.\\ as separate user/assistant turns. "
            "Accuracy (\\%) with unparsed rate in parentheses; paired exact McNemar test.}",
            "\\label{tab:direct_mt}", "\\end{table}", ""]))

    # ---- Rank stability vs. official (primary claim) ordering ----------------
    official = {BASE_TO_CHAT[b][0]: OFFICIAL[b][0]["value"] for b in exp1 if b in OFFICIAL}
    taus = {}
    for key, _ in CONDITIONS:
        scores = {m: a[key] for m, a in per_model.items() if key in a}
        if len(scores) >= 3:
            taus[key] = kendall_tau(official, scores)
            nums[f"tau{alpha(key)}"] = f"{taus[key]:.2f}"
    nums.update({"nSubset": f"{len(subset):,}", "nFull": f"{n_full:,}"})
    if have_chat:
        nums["maxUnparsed"] = pct(worst_unp)
        nums["maxLenientGain"] = pct(max_lenient_gain)
    if n_tests:
        nums["nSigCells"] = str(n_sig)
    ladder_and_arc(args, exp1, exp2, subset, nums)
    (args.out_dir / "numbers.tex").write_text(
        "% Auto-generated by experiments/analysis/paper_tables.py -- do not edit.\n"
        + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(nums.items())))

    # ---- Appendix: per-subject letter 5-shot accuracy (full test set) --------
    full_sets = {BASE_TO_CHAT[b][0]: c["letter_5shot"] for b, c in exp1.items() if c.get("letter_5shot")}
    if full_sets:
        names = list(full_sets)
        per_subj = {n: {} for n in names}
        for n, ok in full_sets.items():
            for uid, v in ok.items():
                per_subj[n].setdefault(uid.rsplit(":", 1)[0], []).append(v)
        subjects = sorted({s for d in per_subj.values() for s in d})
        count = {s: max(len(per_subj[n].get(s, [])) for n in names) for s in subjects}

        def subj_label(s):
            return s.replace("high_school_", "HS ").replace("_and_", " \\& ").replace("_", " ")

        body = [" & ".join([subj_label(s), str(count[s])] + [
            pct(sum(per_subj[n][s]) / len(per_subj[n][s])) if s in per_subj[n] else "--" for n in names])
            + " \\\\" for s in subjects]
        body += ["\\midrule",
                 " & ".join(["Macro", ""] + [pct(macro(full_sets[n])) for n in names]) + " \\\\",
                 " & ".join(["Micro", f"{sum(count.values()):,}"]
                            + [pct(acc_stats(full_sets[n])["acc"]) for n in names]) + " \\\\"]
        (args.out_dir / "tab_subjects.tex").write_text("\n".join([
            "\\begin{table}[h]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{3pt}",
            "\\renewcommand{\\arraystretch}{0.9}",
            "\\resizebox{\\columnwidth}{!}{%",
            "\\begin{tabular}{lr" + "r" * len(names) + "}", "\\toprule",
            "Subject & $n$ & " + " & ".join(names) + " \\\\", "\\midrule",
            *body, "\\bottomrule", "\\end{tabular}}",
            "\\caption{Per-subject letter 5-shot accuracy (\\%) on the full test set "
            "(HS: high school). Macro averages subjects; micro averages items.}",
            "\\label{tab:subjects}", "\\end{table}", ""]))

    # ---- Figure ---------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    xs = list(range(len(CONDITIONS) + 1))
    for base in exp1:
        name = BASE_TO_CHAT[base][0]
        accs = per_model.get(name, {})
        ys = [OFFICIAL[base][0]["value"] if base in OFFICIAL else None] + \
             [100 * accs[k] if k in accs else None for k, _ in CONDITIONS]
        pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
        ax.plot(*zip(*pts), marker="o", ms=3, lw=1, label=name)
    ax.set_xticks(xs)
    ax.set_xticklabels(["Official"] + [h.replace("\\ ", " ") for _, h in CONDITIONS],
                       rotation=35, ha="right", fontsize=6)
    ax.tick_params(axis="y", labelsize=6)
    ax.set_ylabel("MMLU accuracy (%)", fontsize=7)
    ax.legend(fontsize=5.5, frameon=False)
    ax.grid(alpha=0.3, lw=0.5)
    fig.tight_layout()
    fig.savefig(args.out_dir / "fig_protocols.pdf")
    plt.close(fig)

    print(f"Wrote tab_repro.tex, tab_protocols.tex, numbers.tex, fig_protocols.pdf -> {args.out_dir}")
    print("Kendall tau vs official ranking:", {k: round(v, 2) for k, v in taus.items()})


if __name__ == "__main__":
    main()
