"""Experiment 1: Independent reproduction of vendor-reported MMLU numbers,
under several defensible readings of the (under-specified) protocol.

Vendors report base-model MMLU as "5-shot" with little protocol detail
(Qwen3 Technical Report, Table 6: no template, scoring rule or aggregation;
Meta's Llama 3.1 card: "macro_avg/acc_char", internal harness). We
re-evaluate the exact base checkpoints under three protocols and report both
micro (item-pooled) and macro (subject-averaged) accuracy:

  letter_5shot : 5 lettered exemplars, score next-token logprob of " A".." D"
                 (original Hendrycks et al. / lm-eval-harness style).
  letter_0shot : same, without exemplars.
  cloze_5shot  : options not shown; score each answer *text* as a
                 continuation (sum logprob -> acc, per-char normalized ->
                 acc_norm), as in the cloze formulation used by OLMES.

Usage:
    python exp1_reproduction.py --models Qwen/Qwen3-8B-Base --n-per-subject 0   # full test set
    python exp1_reproduction.py --n-per-subject 3                              # quick check

Engineering notes:
- The per-subject few-shot prefix is encoded once and its KV cache reused for
  every item in that subject (cache is cropped back after each item). Scores
  are numerically equivalent to uncached scoring (see --verify-cache).
- Per-item logprobs are appended to results/exp1_<model>_items.jsonl; runs
  resume from that file if interrupted.
"""
import argparse
import copy
import gc
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.mmlu_utils import load_mmlu_subset, mmlu_prefix, mmlu_query
from common.scoring import mcnemar_exact, wilson_ci

# Official base-model MMLU (5-shot) claims. A model can carry claims from
# several sources; "agg" is the aggregation the source states (macro = mean of
# per-subject accuracies, micro = pooled over items, None = not stated).
QWEN3_TR = "Qwen3 Technical Report (arXiv:2505.09388), Table 6"
LLAMA31_CARD = "Meta Llama 3.1 model card (HF meta-llama/Llama-3.1-8B), base models table"
OFFICIAL = {
    "Qwen/Qwen3-8B-Base": [{"value": 76.89, "source": QWEN3_TR, "agg": None}],
    "Qwen/Qwen2.5-7B": [{"value": 74.16, "source": QWEN3_TR, "agg": None}],
    "Qwen/Qwen2.5-14B": [{"value": 79.66, "source": QWEN3_TR, "agg": None}],
    "meta-llama/Meta-Llama-3-8B": [
        {"value": 66.60, "source": QWEN3_TR, "agg": None},
        {"value": 66.7, "source": LLAMA31_CARD, "agg": "macro"},
    ],
    "meta-llama/Llama-3.1-8B": [{"value": 66.7, "source": LLAMA31_CARD, "agg": "macro"}],
}


PROTOCOLS = {
    "letter_5shot": {"style": "letter", "k_shot": 5},
    "letter_0shot": {"style": "letter", "k_shot": 0},
    "cloze_5shot": {"style": "cloze", "k_shot": 5},
}
LETTERS = ["A", "B", "C", "D"]
RESULTS_DIR = Path(__file__).parent / "results"


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def slug(model_id):
    return model_id.replace("/", "__")


class PrefixCachedScorer:
    """Scores (prefix, suffix) pairs reusing a KV cache for the prefix."""

    def __init__(self, model, tokenizer, device, use_cache=True):
        self.model, self.tok, self.device = model, tokenizer, device
        self.use_cache = use_cache
        self.letter_ids = [tokenizer.encode(" " + L, add_special_tokens=False)[-1] for L in LETTERS]
        self._prefix_text = None
        self.fallbacks = 0
        # fp32 LM head: bf16 logits are too coarse (ties between letters)
        self.head_w = model.lm_head.weight.detach().float()
        b = getattr(model.lm_head, "bias", None)
        self.head_b = b.detach().float() if b is not None else None

    def _logprobs(self, hidden):
        logits = torch.nn.functional.linear(hidden.float(), self.head_w, self.head_b)
        return torch.log_softmax(logits, dim=-1)

    def _ids(self, text, special=True):
        return self.tok(text, add_special_tokens=special)["input_ids"]

    @torch.no_grad()
    def _forward(self, ids, cache=None):
        t = torch.tensor([ids], device=self.device)
        out = self.model.model(input_ids=t, past_key_values=cache, use_cache=True)
        return self._logprobs(out.last_hidden_state[0]), out.past_key_values

    def set_prefix(self, prefix_text):
        if prefix_text == self._prefix_text:
            return
        self._prefix_text = prefix_text
        self.prefix_ids = self._ids(prefix_text)
        self.cache = None
        if self.use_cache:
            # Only the KV cache is needed; skip the (fp32, full-vocab) LM head.
            with torch.no_grad():
                t = torch.tensor([self.prefix_ids], device=self.device)
                self.cache = self.model.model(input_ids=t, use_cache=True).past_key_values

    def _split(self, full_text):
        """Token ids of full_text after the cached prefix, or None if the
        tokenization of prefix+suffix does not start with the prefix ids."""
        full = self._ids(full_text)
        n = len(self.prefix_ids)
        if full[:n] == self.prefix_ids:
            return full[n:], full
        return None, full

    def _run_suffix(self, suffix_text):
        """Returns (logprobs over suffix positions, full_ids, used_cache)."""
        suffix_ids, full = self._split(self._prefix_text + suffix_text)
        if self.use_cache and suffix_ids:
            lp, _ = self._forward(suffix_ids, self.cache)
            return lp, full, True
        self.fallbacks += int(self.use_cache)
        lp, _ = self._forward(full)
        return lp, full, False

    def _crop(self):
        if self.use_cache and self.cache is not None:
            self.cache.crop(len(self.prefix_ids))

    def letter_scores(self, query_text):
        lp, _, _ = self._run_suffix(query_text)
        self._crop()
        return [lp[-1, i].item() for i in self.letter_ids]

    @torch.no_grad()
    def letter_scores_batch(self, query_texts):
        """Letter logprobs for several queries sharing the current prefix, in
        one forward pass. Suffixes are right-padded; with causal attention the
        logits at each row's last real token are unaffected by later padding,
        so no attention mask is needed. Only those final hidden states go
        through the LM head (avoids a B x L x vocab logits tensor).
        Falls back to per-item scoring for rows without a clean prefix split.
        """
        if not self.use_cache or len(query_texts) == 1:
            return [self.letter_scores(q) for q in query_texts]
        splits = [self._split(self._prefix_text + q)[0] for q in query_texts]
        ok = [i for i, sp in enumerate(splits) if sp]
        results = [None] * len(query_texts)
        if ok:
            lens = [len(splits[i]) for i in ok]
            pad_id = splits[ok[0]][-1]
            ids = torch.tensor([splits[i] + [pad_id] * (max(lens) - len(splits[i])) for i in ok],
                               device=self.device)
            cache = copy.deepcopy(self.cache)
            cache.batch_repeat_interleave(len(ok))
            hidden = self.model.model(input_ids=ids, past_key_values=cache, use_cache=True).last_hidden_state
            last = hidden[torch.arange(len(ok), device=self.device),
                          torch.tensor(lens, device=self.device) - 1]
            lp = self._logprobs(last)
            for row, i in enumerate(ok):
                results[i] = [lp[row, t].item() for t in self.letter_ids]
            del cache, hidden
        for i, r in enumerate(results):
            if r is None:
                results[i] = self.letter_scores(query_texts[i])
        return results

    @torch.no_grad()
    def continuation_scores_batch(self, query_text, continuations):
        """Same scores as continuation_scores, but the query is run once on top
        of the cached prefix and only the (short) option tokens are run per
        continuation, cropping the cache back to prefix+query after each.
        Falls back to continuation_scores if any sequence does not extend the
        cached prefix+query tokens."""
        ctx_ids = self._ids(self._prefix_text + query_text)
        n, start = len(self.prefix_ids), len(ctx_ids)
        seqs = []
        for cont in continuations:
            full = self._ids(self._prefix_text + query_text + cont)
            cont_ids = full[start:] if full[:start] == ctx_ids else self._ids(cont, special=False)
            seqs.append(ctx_ids + cont_ids)
        if not self.use_cache or start <= n or ctx_ids[:n] != self.prefix_ids or \
                any(len(sq) <= start for sq in seqs):
            return self.continuation_scores(query_text, continuations)
        lp_q, _ = self._forward(ctx_ids[n:], self.cache)  # cache now holds prefix+query
        first = lp_q[-1]
        scores = []
        for sq in seqs:
            cont_ids = sq[start:]
            sc = first[cont_ids[0]].item()
            if len(cont_ids) > 1:
                lp, _ = self._forward(cont_ids[:-1], self.cache)
                sc += lp.gather(1, torch.tensor(cont_ids[1:], device=self.device)[:, None]).sum().item()
                self.cache.crop(start)
            scores.append(sc)
        self._crop()
        return scores

    def continuation_scores(self, query_text, continuations):
        """Sum logprob of each continuation given prefix+query."""
        ctx_text = self._prefix_text + query_text
        ctx_ids = self._ids(ctx_text)
        scores = []
        for cont in continuations:
            full = self._ids(ctx_text + cont)
            cont_ids = full[len(ctx_ids):] if full[: len(ctx_ids)] == ctx_ids \
                else self._ids(cont, special=False)
            seq = ctx_ids + cont_ids
            n = len(self.prefix_ids)
            if self.use_cache and seq[:n] == self.prefix_ids and len(seq) > n:
                lp, _ = self._forward(seq[n:], self.cache)
                offset = n
                self._crop()
            else:
                self.fallbacks += int(self.use_cache)
                lp, _ = self._forward(seq)
                offset = 0
            # logprob of token at absolute position j is lp[j-1-offset]
            start = len(ctx_ids)
            s = sum(lp[j - 1 - offset, seq[j]].item() for j in range(start, len(seq)))
            scores.append(s)
        return scores


def score_item(scorer, item, protocol):
    cfg = PROTOCOLS[protocol]
    scorer.set_prefix(mmlu_prefix(item, cfg["k_shot"], cfg["style"]))
    query = mmlu_query(item, cfg["style"])
    if cfg["style"] == "letter":
        lps = scorer.letter_scores(query)
        return {"logprobs": lps, "pred": int(torch.tensor(lps).argmax())}
    conts = [" " + str(c).strip() for c in item.choices]
    lps = scorer.continuation_scores_batch(query, conts)
    norm = [lp / max(1, len(c)) for lp, c in zip(lps, conts)]
    return {"logprobs": lps, "pred": int(torch.tensor(lps).argmax()),
            "pred_norm": int(torch.tensor(norm).argmax())}


def score_letter_batch(scorer, items, protocol):
    """Batched letter scoring for items that share one prefix."""
    cfg = PROTOCOLS[protocol]
    scorer.set_prefix(mmlu_prefix(items[0], cfg["k_shot"], cfg["style"]))
    lps = scorer.letter_scores_batch([mmlu_query(it, cfg["style"]) for it in items])
    return [{"logprobs": l, "pred": int(torch.tensor(l).argmax())} for l in lps]


def load_model(model_id, device):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()
    return model, tok


def verify_cache(model, tok, device, items, protocols):
    cached = PrefixCachedScorer(model, tok, device, use_cache=True)
    plain = PrefixCachedScorer(model, tok, device, use_cache=False)
    worst = 0.0
    for item in items:
        for p in protocols:
            a, b = score_item(cached, item, p), score_item(plain, item, p)
            worst = max(worst, max(abs(x - y) for x, y in zip(a["logprobs"], b["logprobs"])))
    # batched vs unbatched, per subject group
    worst_batch = 0.0
    for p in [p for p in protocols if PROTOCOLS[p]["style"] == "letter"]:
        for group in _prefix_groups(items, p):
            batched = score_letter_batch(cached, group, p)
            for it, br in zip(group, batched):
                ref = score_item(plain, it, p)
                worst_batch = max(worst_batch, max(abs(x - y) for x, y in zip(br["logprobs"], ref["logprobs"])))
    print(f"max |batched - uncached| letter logprob diff = {worst_batch:.2e}")
    return worst, cached.fallbacks


def _prefix_groups(items, protocol):
    """Consecutive runs of items sharing the same prefix for this protocol."""
    cfg = PROTOCOLS[protocol]
    groups, cur, key = [], [], None
    for it in items:
        k = mmlu_prefix(it, cfg["k_shot"], cfg["style"])
        if k != key and cur:
            groups.append(cur)
            cur = []
        key = k
        cur.append(it)
    if cur:
        groups.append(cur)
    return groups


def summarize(model_id, records, n_items, protocols):
    claims = OFFICIAL.get(model_id, [])
    out = {"model_id": model_id, "official_claims": claims,
           # primary claim (first listed), kept for aggregate/plots compatibility
           "official_reported_accuracy": claims[0]["value"] / 100 if claims else None,
           "protocols": {}}
    by_p = {}
    for r in records:
        by_p.setdefault(r["protocol"], {})[r["uid"]] = r
    metrics = []
    for p in protocols:
        metrics.append((p, "pred"))
        if PROTOCOLS[p]["style"] == "cloze":
            metrics.append((p, "pred_norm"))
    ref = by_p.get("letter_5shot", {})
    for p, field in metrics:
        rs = by_p.get(p, {})
        if not rs:
            continue
        name = p if field == "pred" else p + "_norm"
        ok = {i: r[field] == r["gold"] for i, r in rs.items()}
        k, n = sum(ok.values()), len(ok)
        lo, hi = wilson_ci(k, n)
        subj = {}
        for i, r in rs.items():
            subj.setdefault(r["subject"], []).append(ok[i])
        macro = sum(sum(v) / len(v) for v in subj.values()) / len(subj)
        e = {"correct": k, "total": n, "accuracy": k / n, "ci95": [lo, hi],
             "macro_accuracy": macro, "complete": n == n_items, "vs_official": []}
        for c in claims:
            acc = macro if c["agg"] == "macro" else k / n
            e["vs_official"].append({
                "source": c["source"], "official": c["value"], "agg_used": c["agg"] or "micro",
                "delta_pp": 100 * acc - c["value"],
                "official_in_micro_ci": lo <= c["value"] / 100 <= hi,
            })
        if name != "letter_5shot" and ref:
            common = set(ref) & set(rs)
            refok = {i: ref[i]["pred"] == ref[i]["gold"] for i in common}
            b = sum(refok[i] and not ok[i] for i in common)
            c_ = sum(ok[i] and not refok[i] for i in common)
            e["vs_letter_5shot"] = {"only_ref_correct": b, "only_this_correct": c_,
                                    "mcnemar_p": mcnemar_exact(b, c_)}
        out["protocols"][name] = e
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["Qwen/Qwen3-8B-Base"])
    ap.add_argument("--n-per-subject", type=int, default=20, help="0 = full test set (14,042 items)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--protocols", nargs="+", default=list(PROTOCOLS), choices=list(PROTOCOLS))
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="items per forward pass for letter protocols (1 = unbatched)")
    ap.add_argument("--limit", type=int, default=None, help="first N items only (smoke test)")
    ap.add_argument("--verify-cache", type=int, default=0,
                    help="check cached vs uncached scores on N items, then exit")
    args = ap.parse_args()

    device = args.device or pick_device()
    items = load_mmlu_subset(n_per_subject=args.n_per_subject, seed=args.seed)
    if args.limit:
        items = items[: args.limit]
    print(f"{len(items)} MMLU items; protocols={args.protocols}; device={device}", flush=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    for model_id in args.models:
        print(f"\nLoading {model_id} ...", flush=True)
        model, tok = load_model(model_id, device)

        if args.verify_cache:
            worst, fb = verify_cache(model, tok, device, items[: args.verify_cache], args.protocols)
            print(f"max |cached - uncached| logprob diff = {worst:.2e}; cache fallbacks = {fb}")
            continue

        items_path = RESULTS_DIR / f"exp1_{slug(model_id)}_items.jsonl"
        done = {}
        if items_path.exists():
            for line in items_path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    done[(r["protocol"], r["uid"])] = r
            print(f"Resuming: {len(done)} scored items on disk.", flush=True)

        scorer = PrefixCachedScorer(model, tok, device)
        t0, n_new = time.time(), 0
        todo = len(items) * len(args.protocols) - sum(
            (p, it.uid) in done for p in args.protocols for it in items)
        with items_path.open("a") as fout:
            # Group by subject so each prefix cache is built once per protocol.
            for p in args.protocols:
                pending = [(idx, it) for idx, it in enumerate(items) if (p, it.uid) not in done]
                batchable = PROTOCOLS[p]["style"] == "letter" and args.batch_size > 1
                if batchable:
                    # Length-sort within each subject to minimise padding.
                    first = {}
                    for _, it in pending:
                        first.setdefault(it.subject, len(first))
                    pending.sort(key=lambda x: (first[x[1].subject],
                                                len(tok(mmlu_query(x[1], PROTOCOLS[p]["style"]))["input_ids"])))
                chunks, cur = [], []
                for idx, it in pending:
                    same = cur and mmlu_prefix(cur[0][1], PROTOCOLS[p]["k_shot"], PROTOCOLS[p]["style"]) \
                        == mmlu_prefix(it, PROTOCOLS[p]["k_shot"], PROTOCOLS[p]["style"])
                    if cur and (not batchable or not same or len(cur) >= args.batch_size):
                        chunks.append(cur)
                        cur = []
                    cur.append((idx, it))
                if cur:
                    chunks.append(cur)
                for chunk in chunks:
                    if batchable:
                        results = score_letter_batch(scorer, [it for _, it in chunk], p)
                    else:
                        results = [score_item(scorer, it, p) for _, it in chunk]
                    for (idx, item), res in zip(chunk, results):
                        rec = {"protocol": p, "item_idx": idx, "uid": item.uid, "subject": item.subject,
                               "gold": item.answer_idx, **res}
                        fout.write(json.dumps(rec) + "\n")
                        done[(p, item.uid)] = rec
                        n_new += 1
                        if n_new % 200 == 0:
                            el = time.time() - t0
                            print(f"[{n_new}/{todo}] {p} {el:.0f}s elapsed, "
                                  f"eta {el / n_new * (todo - n_new):.0f}s", flush=True)
                    fout.flush()

        # Resume is keyed on uid, so full-set and subset runs can share one items
        # file. letter_5shot records for these items are kept as the McNemar
        # reference even if this run did not score letter_5shot.
        uids = {it.uid for it in items}
        records = [r for (p, u), r in done.items()
                   if u in uids and (p in args.protocols or p == "letter_5shot")]
        summary = summarize(model_id, records, len(items), args.protocols)
        run_meta = {"n_per_subject": args.n_per_subject, "seed": args.seed,
                    "total_items": len(items), "limit": args.limit, "device": device,
                    "dtype": str(next(model.parameters()).dtype),
                    "cache_fallbacks": scorer.fallbacks, "batch_size": args.batch_size,
                    "runtime_seconds_this_session": time.time() - t0}
        for e in summary["protocols"].values():
            e["n_per_subject"] = args.n_per_subject
        out_path = RESULTS_DIR / f"exp1_{slug(model_id)}.json"
        # Merge with protocols from earlier runs (e.g. full-set letter_5shot
        # followed by subset-only letter_0shot/cloze).
        if out_path.exists():
            prev = json.loads(out_path.read_text())
            summary["protocols"] = {**prev.get("protocols", {}), **summary["protocols"]}
        summary.update(run_meta)
        out_path.write_text(json.dumps(summary, indent=2))

        print(f"\n=== {model_id} ===")
        for c in OFFICIAL.get(model_id, []):
            print(f"  official {c['value']:.2f} ({c['agg'] or 'agg unstated'}) -- {c['source']}")
        for name, e in summary["protocols"].items():  # includes merged earlier runs
            extra = " ".join(f"d={v['delta_pp']:+.2f}pp({v['agg_used']})" for v in e["vs_official"])
            print(f"{name:18s} micro={e['accuracy']:.2%} [{e['ci95'][0]:.2%}, {e['ci95'][1]:.2%}] "
                  f"macro={e['macro_accuracy']:.2%} n={e['total']} {extra}")
        print(f"Saved {out_path}")
        # HF modules hold reference cycles; without gc.collect() the previous
        # model stays resident and the next one pushes the machine into swap.
        del model, tok, scorer
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
