"""Loading and prompt-formatting utilities for the MMLU benchmark.

Reproduces the standard 5-shot MMLU protocol (Hendrycks et al., 2021):
each test question is preceded by 5 worked examples from the same subject's
`dev` split, formatted as multiple-choice questions with lettered answers.
"""
import random
from dataclasses import dataclass, field

from datasets import load_dataset

LETTERS = ["A", "B", "C", "D"]

SUBJECT_DISPLAY = lambda s: s.replace("_", " ")


@dataclass
class MMLUItem:
    subject: str
    question: str
    choices: list
    answer_idx: int  # 0-3
    few_shot: list = field(default_factory=list)  # list of MMLUItem-like dev examples
    uid: str = ""  # stable id: "<subject>:<row index in the cais/mmlu 'all' test split>"


def _format_question(question, choices):
    lines = [question.strip()]
    for letter, choice in zip(LETTERS, choices):
        lines.append(f"{letter}. {choice}")
    return "\n".join(lines)


def format_mmlu_prompt(item: MMLUItem) -> str:
    """Standard 5-shot MMLU prompt ending right before the answer letter."""
    parts = [f"The following are multiple choice questions (with answers) about {SUBJECT_DISPLAY(item.subject)}.\n"]
    for ex in item.few_shot:
        parts.append(_format_question(ex["question"], ex["choices"]))
        parts.append(f"Answer: {LETTERS[ex['answer']]}\n")
    parts.append(_format_question(item.question, item.choices))
    parts.append("Answer:")
    return "\n".join(parts)


def mmlu_prefix(item: MMLUItem, k_shot=5, style="letter") -> str:
    """Shared per-subject context (header + k exemplars). For style="letter",
    mmlu_prefix(item) + mmlu_query(item) == format_mmlu_prompt(item).

    style="cloze" omits the lettered options and shows exemplar answers as
    the answer text itself (the "cloze formulation", CF, used e.g. by OLMES).
    """
    parts = [f"The following are multiple choice questions (with answers) about {SUBJECT_DISPLAY(item.subject)}.\n"]
    for ex in item.few_shot[:k_shot]:
        if style == "letter":
            parts.append(_format_question(ex["question"], ex["choices"]))
            parts.append(f"Answer: {LETTERS[ex['answer']]}\n")
        else:
            parts.append(f"Question: {ex['question'].strip()}")
            parts.append(f"Answer: {str(ex['choices'][ex['answer']]).strip()}\n")
    return "\n".join(parts) + "\n"


def mmlu_query(item: MMLUItem, style="letter") -> str:
    if style == "letter":
        return _format_question(item.question, item.choices) + "\nAnswer:"
    return f"Question: {item.question.strip()}\nAnswer:"


def load_mmlu_subset(n_per_subject=20, seed=0, subjects=None):
    """Loads a stratified subset of MMLU test items with 5-shot dev exemplars attached.

    Returns a list of MMLUItem. With n_per_subject=20 across all 57 subjects
    this is ~1,140 items. n_per_subject=None (or 0) returns the full
    14,042-item test set.
    """
    ds = load_dataset("cais/mmlu", "all")
    test_df = ds["test"].to_pandas()
    test_df["_row"] = range(len(test_df))
    dev_df = ds["dev"].to_pandas()

    all_subjects = sorted(test_df["subject"].unique())
    if subjects:
        all_subjects = [s for s in all_subjects if s in subjects]

    rng = random.Random(seed)
    items = []
    for subject in all_subjects:
        test_rows = test_df[test_df["subject"] == subject].to_dict("records")
        dev_rows = dev_df[dev_df["subject"] == subject].to_dict("records")
        rng.shuffle(test_rows)
        few_shot = dev_rows[:5]
        for row in test_rows[: (n_per_subject or None)]:
            items.append(
                MMLUItem(
                    subject=subject,
                    question=row["question"],
                    choices=list(row["choices"]),
                    answer_idx=int(row["answer"]),
                    few_shot=few_shot,
                    uid=f"{subject}:{row['_row']}",
                )
            )
    return items
