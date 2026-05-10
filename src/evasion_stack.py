"""Predicted evasion-probability side features.

Gold evasion labels are unavailable at test time, so this module trains a small
TF-IDF logistic-regression evasion classifier and appends predicted probabilities.
For train rows it uses out-of-fold probabilities to avoid target leakage.
"""

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

from .data import NUM_EVASION


def _texts(df: pd.DataFrame) -> list:
    return (
        df["question"].astype(str) + " [SEP] " + df["answer"].astype(str)
    ).tolist()


def _make_model(seed: int):
    return make_pipeline(
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=30000,
            sublinear_tf=True,
        ),
        LogisticRegression(
            max_iter=1000,
            C=2.0,
            class_weight="balanced",
            solver="liblinear",
            random_state=seed,
        ),
    )


def _aligned_proba(model, texts: list) -> np.ndarray:
    probs = model.predict_proba(texts)
    aligned = np.zeros((len(texts), NUM_EVASION), dtype=np.float32)
    classes = model.named_steps["logisticregression"].classes_
    for j, cls in enumerate(classes):
        aligned[:, int(cls)] = probs[:, j]
    return aligned


def fit_evasion_stack_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    seed: int,
    n_splits: int = 5,
) -> Tuple[np.ndarray, np.ndarray, object]:
    y = train_df["evasion_id"].astype(int).values
    train_texts = _texts(train_df)
    val_texts = _texts(val_df)
    oof = np.zeros((len(train_df), NUM_EVASION), dtype=np.float32)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold_train, fold_holdout in skf.split(train_texts, y):
        model = _make_model(seed)
        model.fit([train_texts[i] for i in fold_train], y[fold_train])
        holdout_texts = [train_texts[i] for i in fold_holdout]
        oof[fold_holdout] = _aligned_proba(model, holdout_texts)
    final_model = _make_model(seed)
    final_model.fit(train_texts, y)
    val_probs = _aligned_proba(final_model, val_texts)
    return oof, val_probs, final_model


def transform_evasion_stack_features(df: pd.DataFrame, model) -> np.ndarray:
    return _aligned_proba(model, _texts(df))


def save_evasion_stack_model(model, path) -> None:
    joblib.dump(model, Path(path))


def load_evasion_stack_model(path):
    return joblib.load(Path(path))
