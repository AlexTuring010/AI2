"""Local α-sweep ensemble of Phase 5 DistilBERT runs.

clarity runs saved raw logits; evasion runs saved log(clarity_probs). Both
become valid clarity-prob distributions after softmax (softmax(logp)==p when
logp = log(prob_distribution)).
"""

from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "kaggle" / "session_models"

RUNS = {
    "clarity_baseline": "distilbert-base-uncased_two_segment_lr3e-05_bs16_ep3_ml512_seed42",
    "neg_winner": "distilbert-base-uncased_two_segment_lr3e-05_bs16_ep3_ml512_seed42_neg",
    "qtoks": "distilbert-base-uncased_two_segment_lr3e-05_bs16_ep3_ml512_seed42_qtoks",
    "focal": "distilbert-base-uncased_two_segment_lr3e-05_bs16_ep3_ml512_seed42_focal2.0",
    "evasion_s42": "distilbert-base-uncased_two_segment_lr3e-05_bs16_ep3_ml512_seed42_tgtevasion",
    "evasion_s0": "distilbert-base-uncased_two_segment_lr3e-05_bs16_ep3_ml512_seed0_tgtevasion",
    "evasion_s1": "distilbert-base-uncased_two_segment_lr3e-05_bs16_ep3_ml512_seed1_tgtevasion",
}


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def load_probs(run_dir: str):
    d = BASE / run_dir
    logits = np.load(d / "val_preds.npy")
    labels = np.load(d / "val_labels.npy")
    return softmax(logits), labels


def macro_f1(probs, labels):
    return f1_score(labels, probs.argmax(axis=1), average="macro", zero_division=0)


def main():
    probs = {name: load_probs(path)[0] for name, path in RUNS.items()}
    labels = load_probs(RUNS["clarity_baseline"])[1]

    print(f"Val set size: {len(labels)}, class dist: {np.bincount(labels).tolist()}\n")

    print("=== Standalone F1 (sanity check) ===")
    for name, p in probs.items():
        print(f"  {name:20s} F1={macro_f1(p, labels):.4f}")
    print()

    evasion_mean = (probs["evasion_s42"] + probs["evasion_s0"] + probs["evasion_s1"]) / 3
    print(f"  evasion_3seed_mean     F1={macro_f1(evasion_mean, labels):.4f}")
    print()

    # --- Pair sweeps ---
    alphas = np.arange(0.0, 1.0001, 0.05)

    pairs = [
        ("clarity_baseline", "evasion_s42"),
        ("clarity_baseline", "evasion_3seed_mean"),
        ("clarity_baseline", "neg_winner"),
        ("neg_winner", "evasion_s42"),
        ("neg_winner", "evasion_3seed_mean"),
        ("clarity_baseline", "qtoks"),
        ("clarity_baseline", "focal"),
    ]
    evasion_mean_map = {"evasion_3seed_mean": evasion_mean}

    print("=== Pair α-sweep: P = α·A + (1-α)·B ===")
    print(f"{'A':25s} {'B':25s} {'α*':>6s} {'F1*':>7s} {'ΔvsA':>8s} {'ΔvsB':>8s}")
    best_overall = (0.0, None, None, None)
    for a, b in pairs:
        pa = probs.get(a, evasion_mean_map.get(a))
        pb = probs.get(b, evasion_mean_map.get(b))
        scores = []
        for alpha in alphas:
            mix = alpha * pa + (1 - alpha) * pb
            scores.append((alpha, macro_f1(mix, labels)))
        astar, f1star = max(scores, key=lambda x: x[1])
        f1a = macro_f1(pa, labels)
        f1b = macro_f1(pb, labels)
        print(
            f"{a:25s} {b:25s} {astar:>6.2f} {f1star:>7.4f} {f1star-f1a:>+8.4f} {f1star-f1b:>+8.4f}"
        )
        if f1star > best_overall[0]:
            best_overall = (f1star, astar, a, b)

    print()
    print(f"Best pair ensemble: {best_overall[2]} ⊕ {best_overall[3]} @ α={best_overall[1]:.2f}  F1={best_overall[0]:.4f}")

    # --- Triple: clarity + neg + evasion_mean ---
    print("\n=== Triple α,β grid: clarity·α + neg·β + evasion_mean·(1-α-β) ===")
    best = (0.0, None, None)
    step = 0.05
    for a in np.arange(0.0, 1.0001, step):
        for b in np.arange(0.0, 1.0001 - a + 1e-9, step):
            c = 1.0 - a - b
            if c < -1e-9:
                continue
            mix = a * probs["clarity_baseline"] + b * probs["neg_winner"] + c * evasion_mean
            f = macro_f1(mix, labels)
            if f > best[0]:
                best = (f, a, b)
    a, b = best[1], best[2]
    c = 1.0 - a - b
    print(f"  best: clarity={a:.2f} neg={b:.2f} evasion_mean={c:.2f}  F1={best[0]:.4f}")

    # --- Per-class diagnostics for best ensemble ---
    print("\n=== Per-class F1 of top candidates ===")

    def per_class(p):
        preds = p.argmax(1)
        return f1_score(labels, preds, average=None, labels=[0, 1, 2], zero_division=0)

    print(f"  clarity_baseline     {per_class(probs['clarity_baseline'])}")
    print(f"  neg_winner           {per_class(probs['neg_winner'])}")
    print(f"  evasion_3seed_mean   {per_class(evasion_mean)}")
    if best_overall[2] and best_overall[3]:
        pa = probs.get(best_overall[2], evasion_mean_map.get(best_overall[2]))
        pb = probs.get(best_overall[3], evasion_mean_map.get(best_overall[3]))
        best_mix = best_overall[1] * pa + (1 - best_overall[1]) * pb
        print(f"  best pair            {per_class(best_mix)}")
    best_triple = a * probs["clarity_baseline"] + b * probs["neg_winner"] + c * evasion_mean
    print(f"  best triple          {per_class(best_triple)}")


if __name__ == "__main__":
    main()
