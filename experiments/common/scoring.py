"""Two scoring strategies for multiple-choice benchmarks like MMLU.

1. loglikelihood_choice: for HF `transformers` models (typically base/
   pretrained checkpoints). Scores each answer letter directly via the
   model's next-token logits -- this is the standard way vendors report
   base-model MMLU numbers, and it needs no free-form generation.

2. extract_letter_answer: for chat/instruct models queried through Ollama,
   where the model free-generates and we parse out its final answer letter.
"""
import re

import torch

LETTERS = ["A", "B", "C", "D"]


def loglikelihood_choice(model, tokenizer, prompt, device):
    """Returns (predicted_letter_idx, logprobs[4]) by comparing the model's
    next-token log-probability for ' A', ' B', ' C', ' D' after `prompt`.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]  # last-token logits
    logprobs = torch.log_softmax(logits, dim=-1)

    scores = []
    for letter in LETTERS:
        # Leading space matters for most BPE tokenizers (" A" != "A").
        token_ids = tokenizer.encode(" " + letter, add_special_tokens=False)
        token_id = token_ids[-1]
        scores.append(logprobs[token_id].item())

    pred_idx = int(torch.tensor(scores).argmax())
    return pred_idx, scores


_THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.IGNORECASE | re.DOTALL)
_ANSWER_RE = re.compile(
    r"answer(?:\s+is)?\s*[:\-]?\s*\**\s*[\(\[]?([ABCD])(?![A-Za-z])", re.IGNORECASE
)
# A lone letter that is the whole response, optionally wrapped/punctuated:
# "B", "(B)", "B.", "**B**", "B) text"
_LEADING_LETTER_RE = re.compile(r"^\W*([ABCD])(?:[\.\):\*\]\n,]|$)")
# Letter explicitly marked as an option, e.g. "(C)" or "C)" -- never a bare
# capital "A", which is usually the English article.
_MARKED_LETTER_RE = re.compile(r"\(([ABCD])\)|\b([ABCD])\)")


def extract_letter_answer(text):
    """Parses a free-form generation for its final multiple-choice answer.

    Returns the letter, or None if no answer can be identified (callers
    should count None as incorrect *and* report the unparsed rate).
    Priority: explicit "Answer: X" (last occurrence) > response that starts
    with a lone option letter > last explicitly marked option "(X)"/"X)".
    Deliberately does NOT fall back to any standalone capital letter, which
    misreads the article "A" as an answer.
    """
    if not text:
        return None
    text = _THINK_RE.sub("", text).strip()
    m = _ANSWER_RE.findall(text)
    if m:
        return m[-1].upper()
    m = _LEADING_LETTER_RE.match(text)
    if m:
        return m.group(1).upper()
    m = _MARKED_LETTER_RE.findall(text)
    if m:
        a, b = m[-1]
        return (a or b).upper()
    return None


_LETTER_LIST_RE = re.compile(r"(?:\(?[A-D]\)?[\s.,;:]*)+")


def extract_letter_answer_lenient(text):
    """Strict parse first; otherwise, if the whole response is only a list of
    option letters (e.g. the model re-answers the few-shot exemplars as
    "A\nB\nC\nA\nA\nD"), take the last letter. Sensitivity analysis only."""
    pred = extract_letter_answer(text)
    if pred is not None or not text:
        return pred
    body = _THINK_RE.sub("", text).strip()
    if _LETTER_LIST_RE.fullmatch(body):
        return re.findall(r"[A-D]", body)[-1]
    return None


def mcnemar_exact(b, c):
    """Exact two-sided McNemar test p-value for paired binary outcomes.
    b = #items correct under condition 1 only, c = #items correct under
    condition 2 only.
    """
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def wilson_ci(k, n, z=1.96):
    """Wilson score confidence interval for a binomial proportion -- used
    for error bars on accuracy estimates from a finite sample.
    """
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = (z * ((p * (1 - p) / n + z ** 2 / (4 * n ** 2)) ** 0.5)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))
