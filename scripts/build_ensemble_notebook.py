"""Build a single Kaggle notebook that trains 9 models (3 architectures x 3 seeds)
and averages their softmax probabilities on the test set to produce an ensemble
submission. Leaderboard-only; not part of the 3-per-model TA deliverable.

Re-uses the install + library cells from kaggle_experiment.ipynb so the inlined
src/ stays in sync with the dev notebook.
"""
import json
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_NB = PROJECT_ROOT / "notebooks" / "kaggle_experiment.ipynb"
OUT_PATH = PROJECT_ROOT / "notebooks" / "kaggle_ensemble.ipynb"


def new_id() -> str:
    return str(uuid.uuid4())


def _as_lines(text: str) -> list[str]:
    lines = text.splitlines()
    return [ln + "\n" for ln in lines[:-1]] + lines[-1:] if lines else []


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": new_id(),
        "metadata": {},
        "source": _as_lines(source.strip()),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "id": new_id(),
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": _as_lines(source.strip("\n")),
    }


with open(SOURCE_NB) as f:
    dev_nb = json.load(f)

LIBRARY_SRC = None
INSTALL_SRC = None
for cell in dev_nb["cells"]:
    src = "".join(cell.get("source", []))
    if cell["cell_type"] == "code" and "# Cell 1" in src and "Install" in src:
        INSTALL_SRC = src
    if cell["cell_type"] == "code" and "# Cell 2" in src and "Library" in src:
        LIBRARY_SRC = src

assert INSTALL_SRC and LIBRARY_SRC, "could not find install/library cells in source notebook"


INTRO_MD = """
# AI2 - Homework 2: Ensemble των 3 μοντέλων (supplementary)

Αυτό το notebook είναι **εκτός των 3 model-specific deliverables**. Είναι ένα
συμπληρωματικό πείραμα για τον leaderboard score: τρέχω fine-tuning και στα 3 required
μοντέλα (DistilBERT, BERT, DeBERTa-v3) με 3 seeds το καθένα (42, 0, 1), και μετά
μέσο όρο των softmax probabilities για να πάρω ensemble predictions.

**Γιατί**: κάθε μοντέλο/seed συγκλίνει σε διαφορετικό local minimum και κάνει ελαφρώς
διαφορετικά λάθη. Ο μέσος όρος 9 probability distributions συνήθως δίνει +0.02-0.05
macro-F1 πάνω από το καλύτερο single model. Είναι classic variance-reduction τεχνική.

**Single-model best (confirmed)**:
- DistilBERT ml=512 seed=42 wd=0.0: val_f1_macro = 0.6405
- BERT ml=256 seed=1 wd=0.0: val_f1_macro = 0.6417
- DeBERTa ml=256 seed=42 wd=0.0: val_f1_macro = 0.6962

Στόχος του ensemble: να ξεπεράσω το 0.6962 στο val set και να βελτιώσω το leaderboard
score.

**Runtime estimate**: ~3-3.5 ώρες σε Kaggle T4 (9 trainings back-to-back).
"""

CONFIG_MD = """
## Βήμα 3 - Configs για τα 9 training runs

Χρησιμοποιώ το best config κάθε μοντέλου (από τα single-model notebooks) και τρέχω 3
seeds: 42, 0, 1. Όλα με explicit `weight_decay=0.0` (βλ. το DistilBERT notebook για το
rationale - το PyTorch AdamW default είναι 0.01 οπότε πρέπει να το θέσω explicit).
"""

CONFIGS_CODE = """
# 9 configs: 3 architectures x 3 seeds, all with wd=0.0, mode='final'
ensemble_configs = []

# DistilBERT: ml=512 gave the best single-seed F1
for seed in [42, 0, 1]:
    ensemble_configs.append(ExperimentConfig(
        model_name="distilbert-base-uncased",
        input_fmt="two_segment",
        max_length=512,
        lr=3e-5,
        batch_size=16,
        epochs=3,
        warmup_steps=100,
        mode="final",
        seed=seed,
        weight_decay=0.0,
    ))

# BERT: ml=256, ep=5
for seed in [42, 0, 1]:
    ensemble_configs.append(ExperimentConfig(
        model_name="bert-base-uncased",
        input_fmt="two_segment",
        max_length=256,
        lr=2e-5,
        batch_size=16,
        epochs=5,
        warmup_steps=100,
        mode="final",
        seed=seed,
        weight_decay=0.0,
    ))

# DeBERTa-v3: ml=256, ep=3
for seed in [42, 0, 1]:
    ensemble_configs.append(ExperimentConfig(
        model_name="microsoft/deberta-v3-base",
        input_fmt="two_segment",
        max_length=256,
        lr=2e-5,
        batch_size=16,
        epochs=3,
        warmup_steps=100,
        mode="final",
        seed=seed,
        weight_decay=0.0,
    ))

print(f"Total configs: {len(ensemble_configs)}")
for i, cfg in enumerate(ensemble_configs):
    print(f"  [{i+1}] {cfg.run_id}")
"""

TRAIN_MD = """
## Βήμα 4 - Training loop + test inference

Για κάθε config:
1. Τρέχω `run_experiment` (κάνει fine-tuning, σώζει το best checkpoint με βάση val F1).
2. Φορτώνω ξανά το best checkpoint, προβλέπω softmax probs πάνω στο **val** και στο **test**.
3. Αποθηκεύω τα probs για το ensemble.

Το val split είναι fixed (`split_id=0`, stratified), άρα τα val labels είναι ίδια
για όλα τα 9 runs - μπορώ να υπολογίσω ensemble val F1 συνεκτικά.
"""

TRAIN_CODE = """
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from scipy.special import softmax

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# pre-load test data once (load_clarity returns train, test)
_, test_df_raw = load_clarity()
test_df_enc = test_df_raw.copy()
test_df_enc["label_id"] = 0  # dummy labels for tokenizer pipeline

individual_val_f1s = []
test_probs_per_model = []   # each entry: [N_test, 3] softmax array
val_probs_per_model = []    # each entry: [N_val,  3] softmax array
val_labels_ref = None       # same for all runs (fixed split)

for i, cfg in enumerate(ensemble_configs):
    print(f"\\n{'='*70}")
    print(f"[{i+1}/{len(ensemble_configs)}] {cfg.run_id}")
    print('='*70)

    # train (saves best_model.pt to models/<run_id>/)
    history, *_ = run_experiment(cfg)

    best_epoch = max(history, key=lambda h: h["f1_macro"])
    individual_val_f1s.append(best_epoch["f1_macro"])
    print(f"  best val_f1_macro: {best_epoch['f1_macro']:.4f}")

    # reload best checkpoint + tokenizer for inference
    run_dir = Path(cfg.models_dir) / cfg.run_id
    reload_model, reload_tok = load_model_and_tokenizer(cfg.model_name, num_labels=NUM_LABELS)
    reload_model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
    reload_model.to(device).eval()

    # test softmax probs
    test_ds = tokenize_pairs(test_df_enc, reload_tok, cfg.max_length, cfg.input_fmt)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, sampler=SequentialSampler(test_ds))

    test_probs_chunks = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, _ = [b.to(device) for b in batch]
            out = reload_model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(out.logits, dim=-1).cpu().numpy()
            test_probs_chunks.append(probs)
    test_probs = np.concatenate(test_probs_chunks, axis=0)
    test_probs_per_model.append(test_probs)

    # val softmax probs (reuse saved val_logits, convert to probs)
    val_logits = np.load(run_dir / "val_preds.npy")
    val_labels = np.load(run_dir / "val_labels.npy")
    val_probs = softmax(val_logits, axis=-1)
    val_probs_per_model.append(val_probs)
    if val_labels_ref is None:
        val_labels_ref = val_labels
    else:
        # sanity: fixed split means identical val labels every time
        assert np.array_equal(val_labels_ref, val_labels), "val split drift!"

    # free GPU memory before next model
    del reload_model, reload_tok
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

print(f"\\nAll 9 trainings done.")
print(f"Individual val F1s: {[f'{x:.4f}' for x in individual_val_f1s]}")
print(f"Mean individual F1: {np.mean(individual_val_f1s):.4f}")
"""

ENSEMBLE_MD = """
## Βήμα 5 - Ensemble computation

Μέσος όρος των 9 softmax probability matrices (stacked κατά μοντέλο), argmax για τις
τελικές προβλέψεις. Υπολογίζω και το val F1 του ensemble για comparison με τα
individual μοντέλα.
"""

ENSEMBLE_CODE = """
from sklearn.metrics import f1_score, accuracy_score, classification_report

# stack and average
test_probs_stack = np.stack(test_probs_per_model, axis=0)  # [9, N_test, 3]
val_probs_stack  = np.stack(val_probs_per_model,  axis=0)  # [9, N_val,  3]

ens_test_probs = test_probs_stack.mean(axis=0)
ens_val_probs  = val_probs_stack.mean(axis=0)

ens_test_preds = ens_test_probs.argmax(axis=1)
ens_val_preds  = ens_val_probs.argmax(axis=1)

# val metrics for the ensemble
ens_val_f1_macro = f1_score(val_labels_ref, ens_val_preds, average="macro")
ens_val_acc      = accuracy_score(val_labels_ref, ens_val_preds)

print(f"Ensemble val F1 (macro): {ens_val_f1_macro:.4f}")
print(f"Ensemble val accuracy:   {ens_val_acc:.4f}")
print(f"Best single val F1:      {max(individual_val_f1s):.4f}")
print(f"Improvement over best single: +{ens_val_f1_macro - max(individual_val_f1s):+.4f}")
print()

LABEL_ORDER = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]
print(classification_report(val_labels_ref, ens_val_preds, target_names=LABEL_ORDER, digits=3))
"""

SUBMISSION_MD = """
## Βήμα 6 - Submission CSV

Κρατάω τις ensemble προβλέψεις στη μορφή `Id, Predicted` που απαιτεί ο διαγωνισμός.
"""

SUBMISSION_CODE = """
import pandas as pd

submission = pd.DataFrame({
    "Id": test_df_raw["index"] if "index" in test_df_raw.columns else range(len(test_df_raw)),
    "Predicted": [ID2LABEL[p] for p in ens_test_preds],
})

submission.to_csv("/kaggle/working/submission_ensemble.csv", index=False)
submission.to_csv("/kaggle/working/submission.csv", index=False)
print(f"Wrote submission_ensemble.csv and submission.csv ({len(submission)} rows)")
display(submission.head())
print(submission["Predicted"].value_counts())
"""

DIAGNOSTICS_MD = """
## Βήμα 7 - Ανά-μοντέλο diagnostics

Individual val F1 ανά run, για να δω αν κάποιο μοντέλο υστερεί σημαντικά και ενδεχομένως
να το αποκλείσω από το ensemble σε μελλοντική επανάληψη.
"""

DIAGNOSTICS_CODE = """
diag_rows = []
for cfg, f1 in zip(ensemble_configs, individual_val_f1s):
    diag_rows.append({
        "run_id": cfg.run_id,
        "model": cfg.model_name.split("/")[-1],
        "seed": cfg.seed,
        "val_f1_macro": round(f1, 4),
    })

diag_df = pd.DataFrame(diag_rows)
display(diag_df)
print(f"\\nEnsemble val F1: {ens_val_f1_macro:.4f}")
print(f"Best individual:  {diag_df['val_f1_macro'].max():.4f}  ({diag_df.loc[diag_df['val_f1_macro'].idxmax(), 'run_id']})")
print(f"Worst individual: {diag_df['val_f1_macro'].min():.4f}  ({diag_df.loc[diag_df['val_f1_macro'].idxmin(), 'run_id']})")
"""

DOWNLOAD_MD = """
## Βήμα 8 - Download submission

Link για να κατεβάσω τοπικά το submission.csv.
"""

DOWNLOAD_CODE = """
from IPython.display import FileLink
display(FileLink("/kaggle/working/submission_ensemble.csv"))
display(FileLink("/kaggle/working/submission.csv"))
"""


def build() -> dict:
    cells = []
    cells.append(md(INTRO_MD))

    cells.append(md("## Βήμα 1 - Εγκατάσταση dependencies\n\nPin `transformers==4.44.0` για reproducibility (DeBERTa-v3 crashes με 5.0.0)."))
    cells.append(code(INSTALL_SRC.strip()))

    cells.append(md(
        "## Βήμα 2 - Library (inlined src/)\n\n"
        "Ολόκληρο το training pipeline σε ένα cell, όπως στα model-specific notebooks."
    ))
    cells.append(code(LIBRARY_SRC.strip()))

    cells.append(md(CONFIG_MD))
    cells.append(code(CONFIGS_CODE))

    cells.append(md(TRAIN_MD))
    cells.append(code(TRAIN_CODE))

    cells.append(md(ENSEMBLE_MD))
    cells.append(code(ENSEMBLE_CODE))

    cells.append(md(SUBMISSION_MD))
    cells.append(code(SUBMISSION_CODE))

    cells.append(md(DIAGNOSTICS_MD))
    cells.append(code(DIAGNOSTICS_CODE))

    cells.append(md(DOWNLOAD_MD))
    cells.append(code(DOWNLOAD_CODE))

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    nb = build()
    with open(OUT_PATH, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Built: {OUT_PATH}  ({len(nb['cells'])} cells)")
