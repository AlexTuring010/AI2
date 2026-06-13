"""Generate all report figures from saved kaggle/models_archive runs.

Outputs to reports/figures/:
  - learning_curves.png       — val_f1_macro per epoch, one line per confirm run, grouped by model
  - confusion_matrices.png    — best seed per model
  - roc_curves.png            — one-vs-rest ROC per class, best seed per model
  - subgroup_analysis.png     — f1_macro by q_len_bin and a_len_bin (DistilBERT best)
  - error_analysis.txt        — worst-predicted val examples (DistilBERT best)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datasets import load_dataset
from sklearn.metrics import (
    confusion_matrix, f1_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = PROJECT_ROOT / "kaggle" / "models_archive"
OUT = PROJECT_ROOT / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

LABEL_NAMES = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]
LABEL2ID = {"Clear Reply": 0, "Ambivalent": 1, "Clear Non-Reply": 2}
CORRUPTED = list(range(1870, 1884))

# ── helpers ────────────────────────────────────────────────────────────────

def load_run(run_dir: Path):
    metrics = json.loads((run_dir / "metrics.json").read_text())
    val_logits = np.load(run_dir / "val_preds.npy")
    val_labels = np.load(run_dir / "val_labels.npy")
    config = json.loads((run_dir / "config.json").read_text())
    return metrics, val_logits, val_labels, config


def short_model(model_name: str) -> str:
    return model_name.split("/")[-1]


def best_seed_run(runs: list[dict]) -> dict:
    return max(runs, key=lambda r: max(e["f1_macro"] for e in r["metrics"]))


def load_val_df():
    ds = load_dataset("ailsntua/QEvasion")
    train = ds["train"].to_pandas().copy()
    train.rename(columns={"interview_answer": "answer", "clarity_label": "label"}, inplace=True)
    # clean
    qa_cols = ["question", "answer"]
    dup_mask = train.duplicated(subset=qa_cols, keep=False)
    conflict_idx = (
        train[dup_mask].groupby(qa_cols)["label"].nunique()
        .pipe(lambda s: s[s > 1]).index
    )
    drop_mask = pd.MultiIndex.from_frame(train[qa_cols]).isin(
        pd.MultiIndex.from_tuples(conflict_idx)
    )
    train = train[~drop_mask]
    train = train.loc[~train["index"].isin(CORRUPTED)].reset_index(drop=True)
    train["label_id"] = train["label"].map(LABEL2ID)
    # reproduce the same val split (split_id=0, val_size=0.1)
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(train))
    _, val_idx = train_test_split(idx, test_size=0.1, random_state=0,
                                   stratify=train["label_id"].values)
    val_df = train.iloc[val_idx].reset_index(drop=True)
    val_df["q_len"] = val_df["question"].str.split().str.len()
    val_df["a_len"] = val_df["answer"].str.split().str.len()
    val_df["q_len_bin"] = pd.qcut(val_df["q_len"], q=3, labels=["short", "medium", "long"])
    val_df["a_len_bin"] = pd.qcut(val_df["a_len"], q=3, labels=["short", "medium", "long"])
    return val_df


# ── collect runs ───────────────────────────────────────────────────────────

CONFIRM_PREFIXES = {
    "distilbert-base-uncased": "distilbert-base-uncased_two_segment_lr3e-05_bs16_ep3_ml256",
    "bert-base-uncased":       "bert-base-uncased_two_segment_lr2e-05_bs16_ep5_ml256",
    "deberta-v3-base":         "deberta-v3-base_two_segment_lr2e-05_bs16_ep3_ml256",
}

all_runs = {}   # model_short -> list of run dicts
for model_short, prefix in CONFIRM_PREFIXES.items():
    all_runs[model_short] = []
    for seed in (42, 0, 1):
        run_dir = ARCHIVE / f"{prefix}_seed{seed}"
        if not run_dir.exists():
            print(f"[warn] missing: {run_dir.name}")
            continue
        metrics, val_logits, val_labels, config = load_run(run_dir)
        best_f1 = max(e["f1_macro"] for e in metrics)
        all_runs[model_short].append({
            "run_dir": run_dir,
            "metrics": metrics,
            "val_logits": val_logits,
            "val_labels": val_labels,
            "config": config,
            "best_f1": best_f1,
            "seed": seed,
        })


# ── 1. Learning curves ─────────────────────────────────────────────────────

print("Generating learning_curves.png ...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

for ax, (model_short, runs) in zip(axes, all_runs.items()):
    for run in runs:
        epochs = [e["epoch"] for e in run["metrics"]]
        f1s    = [e["f1_macro"] for e in run["metrics"]]
        ax.plot(epochs, f1s, marker="o", color=colors[run["seed"] % 3 if run["seed"] != 42 else 0],
                label=f"seed={run['seed']}", alpha=0.8)
    ax.set_title(model_short)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("val F1-macro")
    ax.legend(fontsize=8)
    ax.set_xticks(epochs)
    ax.grid(True, alpha=0.3)

plt.suptitle("Learning curves — val F1-macro per epoch", y=1.02)
plt.tight_layout()
plt.savefig(OUT / "learning_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  saved -> {OUT / 'learning_curves.png'}")


# ── 2. Confusion matrices ──────────────────────────────────────────────────

print("Generating confusion_matrices.png ...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, (model_short, runs) in zip(axes, all_runs.items()):
    if not runs:
        continue
    best = best_seed_run(runs)
    preds = best["val_logits"].argmax(axis=1)
    cm = confusion_matrix(best["val_labels"], preds, labels=[0, 1, 2])
    print(f"  {model_short} seed={best['seed']} f1={best['best_f1']:.4f} cm={cm.tolist()}")
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=LABEL_NAMES,
                yticklabels=LABEL_NAMES, cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{model_short}\nseed={best['seed']}  F1={best['best_f1']:.4f}")
    ax.tick_params(axis="x", rotation=20)

plt.tight_layout()
plt.savefig(OUT / "confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  saved -> {OUT / 'confusion_matrices.png'}")


# ── 3. ROC curves ──────────────────────────────────────────────────────────

print("Generating roc_curves.png ...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
line_colors = ["#e41a1c", "#377eb8", "#4daf4a"]

for ax, (model_short, runs) in zip(axes, all_runs.items()):
    if not runs:
        continue
    best = best_seed_run(runs)
    y_true_bin = label_binarize(best["val_labels"], classes=[0, 1, 2])
    probs = np.exp(best["val_logits"]) / np.exp(best["val_logits"]).sum(axis=1, keepdims=True)
    for i, (cls_name, color) in enumerate(zip(LABEL_NAMES, line_colors)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, label=f"{cls_name} (AUC={roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"{model_short}\nseed={best['seed']}")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.suptitle("ROC curves (one-vs-rest)", y=1.02)
plt.tight_layout()
plt.savefig(OUT / "roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  saved -> {OUT / 'roc_curves.png'}")


# ── 4. Subgroup analysis ───────────────────────────────────────────────────

print("Generating subgroup_analysis.png ...")
try:
    val_df = load_val_df()
    distilbert_best = best_seed_run(all_runs["distilbert-base-uncased"])
    preds = distilbert_best["val_logits"].argmax(axis=1)
    val_df = val_df.iloc[:len(preds)].copy()
    val_df["pred"] = preds

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, bin_col in zip(axes, ["q_len_bin", "a_len_bin"]):
        rows = []
        for grp_name, grp in val_df.groupby(bin_col, observed=True):
            f1 = f1_score(grp["label_id"], grp["pred"], average="macro", zero_division=0)
            rows.append({"bin": grp_name, "f1_macro": f1, "n": len(grp)})
        grp_df = pd.DataFrame(rows)
        ax.bar(grp_df["bin"].astype(str), grp_df["f1_macro"], color="#4c72b0")
        for _, row in grp_df.iterrows():
            ax.text(str(row["bin"]), row["f1_macro"] + 0.005, f"n={row['n']}", ha="center", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_xlabel(bin_col)
        ax.set_ylabel("val F1-macro")
        ax.set_title(f"DistilBERT — subgroup by {bin_col}")
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / "subgroup_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved -> {OUT / 'subgroup_analysis.png'}")
except Exception as e:
    print(f"  [warn] subgroup analysis failed: {e}")


# ── 5. Error analysis ──────────────────────────────────────────────────────

print("Generating error_analysis.txt ...")
try:
    val_df_err = load_val_df()
    distilbert_best = best_seed_run(all_runs["distilbert-base-uncased"])
    preds = distilbert_best["val_logits"].argmax(axis=1)
    probs = np.exp(distilbert_best["val_logits"]) / np.exp(distilbert_best["val_logits"]).sum(axis=1, keepdims=True)
    val_df_err = val_df_err.iloc[:len(preds)].copy()
    val_df_err["pred"] = preds
    val_df_err["confidence"] = probs.max(axis=1)
    val_df_err["correct"] = val_df_err["pred"] == val_df_err["label_id"]

    errors = val_df_err[~val_df_err["correct"]].sort_values("confidence", ascending=False)
    id2label = {0: "Clear Reply", 1: "Ambivalent", 2: "Clear Non-Reply"}

    lines = [f"Error analysis — DistilBERT best run (seed={distilbert_best['seed']})",
             f"Total errors: {len(errors)} / {len(val_df_err)} ({len(errors)/len(val_df_err)*100:.1f}%)\n"]

    for _, row in errors.head(20).iterrows():
        lines.append(f"TRUE: {id2label[int(row['label_id'])]}  |  PRED: {id2label[int(row['pred'])]}  |  conf={row['confidence']:.3f}")
        lines.append(f"  Q: {str(row['question'])[:120]}")
        lines.append(f"  A: {str(row['answer'])[:120]}")
        lines.append("")

    (OUT / "error_analysis.txt").write_text("\n".join(lines))
    print(f"  saved -> {OUT / 'error_analysis.txt'}")
except Exception as e:
    print(f"  [warn] error analysis failed: {e}")


print("\nDone. All figures saved to reports/figures/")
