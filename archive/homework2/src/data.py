from typing import Optional, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Clarity level (3 classes) - το required output του assignment
CLARITY_LABELS = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]
CLARITY2ID = {l: i for i, l in enumerate(CLARITY_LABELS)}
ID2CLARITY = {i: l for l, i in CLARITY2ID.items()}
NUM_CLARITY = 3

# Evasion level (9 classes) - fine-grained taxonomy από το paper
EVASION_LABELS = [
    "Explicit",
    "Implicit",
    "General",
    "Partial/half-answer",
    "Dodging",
    "Deflection",
    "Declining to answer",
    "Claims ignorance",
    "Clarification",
]
EVASION2ID = {l: i for i, l in enumerate(EVASION_LABELS)}
ID2EVASION = {i: l for l, i in EVASION2ID.items()}
NUM_EVASION = 9

# Deterministic mapping από το dataset paper (arxiv 2409.13879)
EVASION_TO_CLARITY = {
    "Explicit": "Clear Reply",
    "Implicit": "Ambivalent",
    "General": "Ambivalent",
    "Partial/half-answer": "Ambivalent",
    "Dodging": "Ambivalent",
    "Deflection": "Ambivalent",
    "Declining to answer": "Clear Non-Reply",
    "Claims ignorance": "Clear Non-Reply",
    "Clarification": "Clear Non-Reply",
}
EVASION_ID_TO_CLARITY_ID = {
    EVASION2ID[e]: CLARITY2ID[EVASION_TO_CLARITY[e]] for e in EVASION_LABELS
}

# Backward compat - default είναι clarity
LABEL2ID = CLARITY2ID
ID2LABEL = ID2CLARITY
NUM_LABELS = NUM_CLARITY

HF_DATASET = "ailsntua/QEvasion"


def load_clarity(data_dir: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load QEvasion από Hugging Face (όπως στο HW1). Κάνω rename στο boundary:
    `interview_answer` -> `answer`, `clarity_label` -> `label`. Το `question` και
    το `evasion_label` μένουν ως είναι (το evasion_label χρειάζεται για το
    evasion-based training experiment).
    """
    ds = load_dataset(HF_DATASET)
    train = ds["train"].to_pandas().copy()
    test = ds["test"].to_pandas().copy()
    for df in (train, test):
        df.rename(
            columns={"interview_answer": "answer", "clarity_label": "label"},
            inplace=True,
        )
    return train, test


# Rows με visibly corrupted answer text, εντοπισμένα στην ανάλυση του HW1
# μέσω OOV-based inspection. Αφαιρούμε by `index` column του dataset.
CORRUPTED_TRAIN_INDEX_VALUES = list(range(1870, 1884))  # 1870..1883 inclusive (14 rows)


def clean_data(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Forum-literal ordering:
    1. Conflicts detected στο RAW set — drop ΟΛΑ τα rows κάθε conflicting (q,a) pair
       (και τα δύο σκέλη, ακόμα κι αν το ένα τυχαίνει να είναι corrupted).
    2. Drop τα εναπομείναντα corrupted rows (index ∈ [1870, 1883]) από το HW1 OOV analysis.
    Expected: 3448 -> 3424 (−24 conflicts) -> 3410 (−14 corrupted, αν δεν υπάρχει overlap
    με τα ήδη-dropped conflict rows).
    """
    raw_n = len(df)
    qa_cols = ["question", "answer"]

    dup_mask = df.duplicated(subset=qa_cols, keep=False)
    conflicting_ids = (
        df[dup_mask]
        .groupby(qa_cols)["label"]
        .nunique()
        .pipe(lambda s: s[s > 1])
        .index
    )
    multi_idx = pd.MultiIndex.from_frame(df[qa_cols])
    conflict_multi = pd.MultiIndex.from_tuples(conflicting_ids)
    drop_mask = multi_idx.isin(conflict_multi)
    cleaned = df[~drop_mask].copy()
    after_conflicts = len(cleaned)

    cleaned = cleaned.loc[~cleaned["index"].isin(CORRUPTED_TRAIN_INDEX_VALUES)].reset_index(drop=True)
    after_corrupted = len(cleaned)

    if verbose:
        print(
            f"[clean_data] raw={raw_n} "
            f"-> after_conflicts={after_conflicts} (−{raw_n - after_conflicts}) "
            f"-> after_corrupted={after_corrupted} (−{after_conflicts - after_corrupted})"
        )
    return cleaned


def encode_labels(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """Κάνω encode και τα clarity labels (`label` -> `label_id`) και τα evasion
    labels (`evasion_label` -> `evasion_id`), αν υπάρχουν. Το evasion_id το
    χρειάζομαι για το evasion-based training experiment όπου κάνω train σε 9
    classes και mapping πίσω σε 3 στο eval time.
    """
    df = df.copy()
    df["label_id"] = df[label_col].map(LABEL2ID)
    if "evasion_label" in df.columns:
        df["evasion_id"] = df["evasion_label"].map(EVASION2ID)
    return df


def create_split(
    df: pd.DataFrame,
    val_size: float = 0.1,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Επιστρέφει (train_indices, val_indices) — fixed split για fair comparison."""
    idx = np.arange(len(df))
    train_idx, val_idx = train_test_split(
        idx,
        test_size=val_size,
        random_state=seed,
        stratify=df["label_id"].values,
    )
    return train_idx, val_idx


def subgroup_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Προσθέτει q_len_bin και a_len_bin για subgroup analysis."""
    df = df.copy()
    df["q_len"] = df["question"].str.split().str.len()
    df["a_len"] = df["answer"].str.split().str.len()
    df["q_len_bin"] = pd.qcut(df["q_len"], q=3, labels=["short", "medium", "long"])
    df["a_len_bin"] = pd.qcut(df["a_len"], q=3, labels=["short", "medium", "long"])
    return df


def smoke_subset(df: pd.DataFrame, n_per_class: int = 50, seed: int = 42) -> pd.DataFrame:
    """Μικρό subset για smoke testing — n δείγματα ανά κλάση."""
    parts = [
        g.sample(min(n_per_class, len(g)), random_state=seed)
        for _, g in df.groupby("label_id")
    ]
    return pd.concat(parts, ignore_index=True)
