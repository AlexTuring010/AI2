"""Local CPU inference για 3 DeBERTa seeds -> test_logits + α-weighted submission.

Loads best_model.pt from kaggle/models_archive/ for seeds 42, 0, 1, runs inference
στο held-out Kaggle test set (308 rows), saves:
- test_logits_seed{seed}.npy (308, 3)   raw logits per seed
- submission.csv                         α-weighted ensemble (0.8·seed42 + 0.0·seed0 + 0.2·seed1)
                                         using the same weights found optimal on val (F1=0.7070)

CPU inference is slow (~1-2 min per seed @ 256 tokens × 308 examples on modern CPU),
but this is a one-shot.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data import ID2LABEL, NUM_LABELS, load_clarity
from src.tokenization import tokenize_pairs

ARCHIVE = PROJECT_ROOT / "kaggle" / "models_archive"
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 256
INPUT_FMT = "two_segment"
BATCH_SIZE = 8  # smaller on CPU
SEEDS = [42, 0, 1]
ALPHA = {42: 0.8, 0: 0.0, 1: 0.2}  # from α-grid on val (F1=0.7070)
RUN_TEMPLATE = "deberta-v3-base_two_segment_lr2e-05_bs16_ep3_ml256_seed{seed}"
OUT_DIR = PROJECT_ROOT / "kaggle" / "ensemble_artifacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def infer_one_seed(seed: int, test_df: pd.DataFrame, tokenizer) -> np.ndarray:
    """Load best_model.pt for this seed and return test logits (N, 3)."""
    run_dir = ARCHIVE / RUN_TEMPLATE.format(seed=seed)
    ckpt = run_dir / "best_model.pt"
    print(f"[seed={seed}] loading {ckpt}")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    state = torch.load(ckpt, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [warn] missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  [warn] unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    model.eval()

    # tokenize_pairs expects label_id column; add dummy
    df = test_df.copy()
    df["label_id"] = 0
    ds = tokenize_pairs(df, tokenizer, MAX_LENGTH, INPUT_FMT)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, sampler=SequentialSampler(ds))

    all_logits = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            input_ids, attention_mask, _ = batch
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(out.logits.cpu().numpy())
            if (i + 1) % 5 == 0 or (i + 1) == len(loader):
                print(f"  [seed={seed}] batch {i+1}/{len(loader)}")
    logits = np.concatenate(all_logits, axis=0)
    out_path = OUT_DIR / f"test_logits_seed{seed}.npy"
    np.save(out_path, logits)
    print(f"  [seed={seed}] saved {out_path}  shape={logits.shape}")
    del model
    return logits


def logits_to_probs(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def main():
    print(f"[info] device: CPU | max_length={MAX_LENGTH} | fmt={INPUT_FMT}")
    _, test_df = load_clarity()
    print(f"[info] test rows: {len(test_df)}  columns: {list(test_df.columns)[:8]}...")
    assert "index" in test_df.columns, "test_df missing 'index' column — submission Id will break"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    seed_probs = {}
    for seed in SEEDS:
        logits = infer_one_seed(seed, test_df, tokenizer)
        seed_probs[seed] = logits_to_probs(logits)

    # α-weighted ensemble
    probs_ens = sum(ALPHA[s] * seed_probs[s] for s in SEEDS)
    preds = probs_ens.argmax(axis=1)
    print(f"\n[ensemble] weights {ALPHA}  pred distribution: "
          f"{np.bincount(preds, minlength=3).tolist()}")

    submission = pd.DataFrame({
        "Id": test_df["index"].values,
        "Predicted": [ID2LABEL[int(p)] for p in preds],
    })
    out_csv = OUT_DIR / "submission.csv"
    submission.to_csv(out_csv, index=False)
    print(f"\n[done] wrote {out_csv}")
    print(submission["Predicted"].value_counts())

    # also save equal-weight for comparison
    probs_eq = np.mean([seed_probs[s] for s in SEEDS], axis=0)
    preds_eq = probs_eq.argmax(axis=1)
    sub_eq = pd.DataFrame({
        "Id": test_df["index"].values,
        "Predicted": [ID2LABEL[int(p)] for p in preds_eq],
    })
    out_eq = OUT_DIR / "submission_equal_weight.csv"
    sub_eq.to_csv(out_eq, index=False)
    print(f"[done] wrote {out_eq} (for comparison; val F1 was 0.6712)")

    # sanity: how often do the two submissions disagree?
    disagree = (preds != preds_eq).sum()
    print(f"[info] α-weighted vs equal-weight disagreements: {disagree}/{len(preds)}")


if __name__ == "__main__":
    main()
