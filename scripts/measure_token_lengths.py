"""Μετράμε το token-length distribution των Q+A pairs στο cleaned train set,
για κάθε συνδυασμό (model tokenizer x input_fmt). Στόχος: να διαλέξουμε
max_length που κρατάει ~95% των παραδειγμάτων χωρίς truncation.

Run: python scripts/measure_token_lengths.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import clean_data, load_clarity  # noqa: E402

MODELS = [
    "distilbert-base-uncased",
    "bert-base-uncased",
    "microsoft/deberta-v3-base",
]
INPUT_FMTS = ["two_segment", "concat_sep"]
PERCENTILES = [50, 75, 90, 95, 99, 100]


def token_lengths(tokenizer, questions, answers, fmt):
    if fmt == "two_segment":
        enc = tokenizer(questions, answers, truncation=False, padding=False)
    elif fmt == "concat_sep":
        sep = tokenizer.sep_token or "[SEP]"
        texts = [f"{q} {sep} {a}" for q, a in zip(questions, answers)]
        enc = tokenizer(texts, truncation=False, padding=False)
    else:
        raise ValueError(fmt)
    return np.array([len(ids) for ids in enc["input_ids"]])


def main():
    train, _ = load_clarity()
    cleaned = clean_data(train, verbose=True)
    questions = cleaned["question"].astype(str).tolist()
    answers = cleaned["answer"].astype(str).tolist()

    header = f"{'model':<32} {'fmt':<14} " + " ".join(f"p{p:<4}" for p in PERCENTILES)
    print(header)
    print("-" * len(header))

    for model_name in MODELS:
        tok = AutoTokenizer.from_pretrained(model_name)
        for fmt in INPUT_FMTS:
            lens = token_lengths(tok, questions, answers, fmt)
            pcts = np.percentile(lens, PERCENTILES).astype(int)
            cells = " ".join(f"{v:<5}" for v in pcts)
            print(f"{model_name:<32} {fmt:<14} {cells}")

    print("\n[info] Candidate max_length values: look at p95 column.")
    print("[info] Common choices: 128, 192, 256. Round up from p95 to nearest power/standard.")


if __name__ == "__main__":
    main()
