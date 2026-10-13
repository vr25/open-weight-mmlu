"""ARC-Challenge items with Meta's released 25-shot prompts.

Meta's Llama-3.1-8B eval details (gated HF dataset meta-llama/Llama-3.1-8B-evals)
contain, for each of the 1,165 four-option ARC-Challenge test items, the exact
prompt used for the model-card number (25 exemplars sampled per item from the
train split). We parse those exemplars so every protocol uses the same ones.
"""
import re
from dataclasses import dataclass

import pandas as pd
from huggingface_hub import hf_hub_download

META_REPO = "meta-llama/Llama-3.1-8B-evals"
META_FILE = "Llama-3.1-8B-evals/Details_arc_challenge_2024-07-22T14-58-13.476219.parquet.gzip"
LETTERS = "ABCD"
_BLOCK = re.compile(r"Question: (.*?)\nA\. (.*?)\nB\. (.*?)\nC\. (.*?)\nD\. (.*?)\nAnswer: ([ABCD])\n\n", re.S)


@dataclass
class ARCItem:
    uid: str
    question: str
    choices: list
    answer_idx: int
    shots: list            # [(question, [4 choices], answer_idx)] x 25
    meta_prompt: str       # Meta's prompt up to and including "Answer:"
    meta_pred: int
    meta_correct: bool


def _q(question, choices):
    return f"Question: {question}\n" + "".join(f"{L}. {c}\n" for L, c in zip(LETTERS, choices))


def letter_prefix(it):
    return "".join(_q(q, c) + f"Answer: {LETTERS[a]}\n\n" for q, c, a in it.shots)


def letter_query(it):
    return _q(it.question, it.choices) + "Answer:"


def cloze_prefix(it):
    return "".join(f"Question: {q}\nAnswer: {c[a]}\n\n" for q, c, a in it.shots)


def cloze_query(it):
    return f"Question: {it.question}\nAnswer:"


def load_arc():
    df = pd.read_parquet(hf_hub_download(META_REPO, META_FILE, repo_type="dataset"))
    items = []
    for r in df.itertuples():
        prompt = r.input_final_prompts[0]
        assert prompt.endswith("Answer: A"), prompt[-30:]
        head = prompt[: -len(" A")]
        shots = [(q, [a, b, c, d], LETTERS.index(L)) for q, a, b, c, d, L in _BLOCK.findall(head)]
        ch = [r.input_choice_list[L] for L in LETTERS]
        it = ARCItem(uid=r.input_question_hash[:16], question=r.input_question, choices=ch,
                     answer_idx=LETTERS.index(r.input_correct_responses[0][-1]), shots=shots,
                     meta_prompt=head, meta_pred=LETTERS.index(r.output_parsed_answer),
                     meta_correct=bool(r.is_correct))
        assert len(shots) == 25 and letter_prefix(it) + letter_query(it) == head, r.Index
        items.append(it)
    return items
