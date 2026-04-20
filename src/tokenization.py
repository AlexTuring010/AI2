import pandas as pd
import torch
from torch.utils.data import TensorDataset
from transformers import PreTrainedTokenizerBase


def tokenize_pairs(
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    input_fmt: str = "two_segment",
) -> TensorDataset:
    """Tokenize Q+A pairs. Returns TensorDataset of (input_ids, attention_mask, labels)."""
    questions = df["question"].astype(str).tolist()
    answers = df["answer"].astype(str).tolist()

    if input_fmt == "two_segment":
        enc = tokenizer(
            questions,
            answers,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
    elif input_fmt == "concat_sep":
        sep = tokenizer.sep_token or "[SEP]"
        texts = [f"{q} {sep} {a}" for q, a in zip(questions, answers)]
        enc = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
    else:
        raise ValueError(f"Unknown input_fmt: {input_fmt!r}")

    labels = torch.tensor(df["label_id"].values, dtype=torch.long)
    return TensorDataset(enc["input_ids"], enc["attention_mask"], labels)
