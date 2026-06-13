"""DeBERTa 3-seed val logit-ensemble.

Loads val_preds (log-probs of clarity classes) from seeds 42, 0, 1 and reports:
- per-seed val F1_macro (sanity check against EXPERIMENTS.md)
- 3-seed average-of-probabilities ensemble F1
- per-class F1 for best vs ensemble
- α-weighted ensembles over grid (find best combination if one seed dominates)
"""
import itertools
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report

ARCHIVE = Path("/home/alexg/projects/AI2/kaggle/models_archive")
SEEDS = [42, 0, 1]
DEB_TEMPLATE = "deberta-v3-base_two_segment_lr2e-05_bs16_ep3_ml256_seed{seed}"


def load_run(seed: int):
    d = ARCHIVE / DEB_TEMPLATE.format(seed=seed)
    logp = np.load(d / "val_preds.npy")  # (342, 3) log-probs
    y = np.load(d / "val_labels.npy")    # (342,)
    probs = np.exp(logp)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs, y


per_seed = {}
for s in SEEDS:
    probs, y = load_run(s)
    preds = probs.argmax(1)
    f1 = f1_score(y, preds, average="macro")
    acc = accuracy_score(y, preds)
    f1_pc = f1_score(y, preds, average=None)
    per_seed[s] = (probs, y, f1, acc, f1_pc)
    print(f"seed={s:>3}  f1_macro={f1:.4f}  acc={acc:.4f}  per_class={[f'{x:.3f}' for x in f1_pc]}")

# sanity check: labels identical across seeds (same val split)
y_ref = per_seed[42][1]
for s in SEEDS:
    assert np.array_equal(per_seed[s][1], y_ref), f"seed={s} has different val labels!"
print("\n[ok] all seeds share the same val split (342 examples)")

# 3-seed averaged ensemble
probs_mean = np.mean([per_seed[s][0] for s in SEEDS], axis=0)
preds_mean = probs_mean.argmax(1)
f1_mean = f1_score(y_ref, preds_mean, average="macro")
acc_mean = accuracy_score(y_ref, preds_mean)
f1_pc_mean = f1_score(y_ref, preds_mean, average=None)

best_seed = max(SEEDS, key=lambda s: per_seed[s][2])
best_f1 = per_seed[best_seed][2]
print(f"\nbest single seed: seed={best_seed}  f1_macro={best_f1:.4f}")
print(f"3-seed ensemble: f1_macro={f1_mean:.4f}  acc={acc_mean:.4f}  per_class={[f'{x:.3f}' for x in f1_pc_mean]}")
print(f"Δ vs best single seed: {f1_mean - best_f1:+.4f}")

# α-grid search over 3 weights (sum to 1)
print("\n=== α-grid search (increments of 0.05) ===")
best_grid = (best_f1, None)
for a in np.arange(0.0, 1.01, 0.05):
    for b in np.arange(0.0, 1.01 - a, 0.05):
        c = 1.0 - a - b
        if c < 0:
            continue
        p = a * per_seed[SEEDS[0]][0] + b * per_seed[SEEDS[1]][0] + c * per_seed[SEEDS[2]][0]
        f1 = f1_score(y_ref, p.argmax(1), average="macro")
        if f1 > best_grid[0]:
            best_grid = (f1, (round(a, 2), round(b, 2), round(c, 2)))

print(f"best α-weighted ensemble: f1_macro={best_grid[0]:.4f}  weights(seed42, seed0, seed1)={best_grid[1]}")
print(f"Δ vs best single seed (0.6962):  {best_grid[0] - best_f1:+.4f}")

# classification report on the ensemble
print("\n=== 3-seed equal-weight ensemble classification report ===")
print(classification_report(y_ref, preds_mean, target_names=["CR", "AMB", "CNR"], digits=4))
