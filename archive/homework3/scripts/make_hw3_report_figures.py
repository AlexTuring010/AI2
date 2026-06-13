"""Generate all figures for the HW3 report into overleaf_report_package_hw3/figures/."""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

OUT = r"C:\Users\alexg\AI2\overleaf_report_package_hw3\figures"
os.makedirs(OUT, exist_ok=True)

LABELS = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]
SHORT_LABELS = ["CR", "AMB", "CNR"]
SIZES = ["0.8B", "2B", "4B"]
STRATS = ["zero-shot", "few-shot", "CoT"]
GRID = np.array([
    [0.298, 0.345, 0.419],
    [0.393, 0.381, 0.488],
    [0.318, 0.414, 0.548],
])


def fig_grid_heatmap():
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    im = ax.imshow(GRID, cmap="YlGnBu", vmin=0.25, vmax=0.60, aspect="auto")
    ax.set_xticks(range(len(STRATS)))
    ax.set_xticklabels(STRATS)
    ax.set_yticks(range(len(SIZES)))
    ax.set_yticklabels(SIZES)
    ax.set_xlabel("Στρατηγική prompting")
    ax.set_ylabel("Μέγεθος Qwen3.5")
    ax.set_title("Phase 4 grid: macro-F1 (dev N=500, greedy, no-think)")
    for i in range(3):
        for j in range(3):
            v = GRID[i, j]
            txt_color = "white" if v > 0.45 else "black"
            label = f"{v:.3f}"
            if (i, j) == (2, 2):
                label += "\n(best)"
            ax.text(j, i, label, ha="center", va="center",
                    color=txt_color, fontweight="bold" if (i, j) == (2, 2) else "normal")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("macro-F1")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "grid_heatmap.png"))
    plt.close(fig)


def fig_grid_bars():
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    x = np.arange(len(SIZES))
    w = 0.26
    colors = ["#88a4d4", "#f5b06b", "#5fae6a"]
    for k, strat in enumerate(STRATS):
        vals = GRID[:, k]
        ax.bar(x + (k - 1) * w, vals, width=w, label=strat, color=colors[k],
               edgecolor="black", linewidth=0.5)
        for xi, v in zip(x + (k - 1) * w, vals):
            ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(0.25, ls="--", color="gray", lw=1, label="dummy AMB-only (0.25)")
    ax.set_xticks(x)
    ax.set_xticklabels(SIZES)
    ax.set_xlabel("Μέγεθος Qwen3.5")
    ax.set_ylabel("macro-F1 (dev N=500)")
    ax.set_title("Phase 4: 3 στρατηγικές × 3 μεγέθη")
    ax.set_ylim(0.20, 0.62)
    ax.legend(loc="upper left", ncol=2, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "grid_bars.png"))
    plt.close(fig)


def fig_cot_scaling():
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    cot = [0.419, 0.488, 0.548]
    zero = [0.298, 0.393, 0.318]
    few = [0.345, 0.381, 0.414]
    x_labels = SIZES
    x = np.arange(len(SIZES))
    ax.plot(x, cot, "o-", color="#2a7f3f", lw=2.2, ms=8, label="CoT")
    ax.plot(x, few, "s--", color="#cc7a00", lw=1.8, ms=7, label="few-shot")
    ax.plot(x, zero, "^--", color="#3357a8", lw=1.8, ms=7, label="zero-shot")
    for xi, v in zip(x, cot):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=9, color="#2a7f3f", fontweight="bold")
    for xi, v in zip(x, zero):
        ax.text(xi, v - 0.020, f"{v:.3f}", ha="center", fontsize=9, color="#3357a8")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Μέγεθος Qwen3.5")
    ax.set_ylabel("macro-F1 (dev N=500)")
    ax.set_title("Scaling με το μέγεθος ανά στρατηγική")
    ax.set_ylim(0.25, 0.60)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "cot_scaling.png"))
    plt.close(fig)


def _cm_heatmap(ax, cm, title, vmax=None):
    if vmax is None:
        vmax = cm.max()
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=vmax)
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(SHORT_LABELS)
    ax.set_yticklabels(SHORT_LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    ax.set_title(title)
    for i in range(3):
        for j in range(3):
            v = cm[i, j]
            color = "white" if v > vmax * 0.55 else "black"
            ax.text(j, i, str(int(v)), ha="center", va="center", color=color, fontsize=11)
    return im


def fig_confusion_4b():
    cm_zero = np.array([
        [54,  4, 94],
        [88, 16, 192],
        [11,  2, 39],
    ])
    cm_cot = np.array([
        [73, 76, 3],
        [37, 211, 48],
        [1, 25, 26],
    ])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    _cm_heatmap(axes[0], cm_zero, "4B zero-shot — collapses to CNR\n(macro-F1 = 0.318)")
    _cm_heatmap(axes[1], cm_cot, "4B + CoT — the working system\n(macro-F1 = 0.548)")
    fig.suptitle("Πώς το CoT διορθώνει το collapse στο 4B (rows = gold, cols = pred, N=500)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "confusion_4b.png"))
    plt.close(fig)


def fig_prompt_evolution():
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    variants = ["v0\nnaive", "v1\n+ορισμοί", "v2\n+anti-collapse", "v3\n+rubric",
                "v4\nsharpened", "v4\n+ CoT"]
    macros = [0.209, 0.222, 0.239, 0.291, 0.322, 0.417]
    accs = [0.202, 0.330, 0.354, 0.384, 0.368, 0.490]
    x = np.arange(len(variants))
    w = 0.4
    b1 = ax.bar(x - w / 2, macros, width=w, color="#3357a8", label="macro-F1", edgecolor="black", lw=0.5)
    b2 = ax.bar(x + w / 2, accs, width=w, color="#cc7a00", label="accuracy", edgecolor="black", lw=0.5)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008,
                    f"{b.get_height():.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.set_ylabel("score (Qwen3.5-0.8B, dev N=500)")
    ax.set_title("Phase 2–3: η εξέλιξη του prompt στο 0.8B")
    ax.set_ylim(0, 0.62)
    ax.axhline(0.25, ls="--", color="gray", lw=1, label="dummy macro-F1 (0.25)")
    ax.axhline(0.59, ls=":", color="gray", lw=1, label="dummy accuracy (0.59)")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "prompt_evolution.png"))
    plt.close(fig)


def fig_subgroup_answer_length():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bins = ["short\n(<60 λέξεις)", "medium\n(60–180 λέξεις)", "long\n(>180 λέξεις)"]
    cot_4b = [0.563, 0.583, 0.455]
    rubric_08b = [0.34, 0.30, 0.17]
    x = np.arange(len(bins))
    w = 0.4
    b1 = ax.bar(x - w / 2, rubric_08b, width=w, color="#cc7a00",
                label="0.8B + v3_rubric (Phase 2, zero-shot)", edgecolor="black", lw=0.5)
    b2 = ax.bar(x + w / 2, cot_4b, width=w, color="#2a7f3f",
                label="4B + CoT (best system, Phase 4)", edgecolor="black", lw=0.5)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008,
                    f"{b.get_height():.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.set_ylabel("macro-F1 ανά subgroup")
    ax.set_title("Πώς το CoT σώζει το long-answer subgroup")
    ax.set_ylim(0, 0.72)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "subgroup_answer_length.png"))
    plt.close(fig)


def fig_question_length_subgroup():
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    bins = ["short\n(<10 λέξεις)", "medium\n(10–20)", "long\n(>20)"]
    vals = [0.607, 0.535, 0.486]
    x = np.arange(len(bins))
    bars = ax.bar(x, vals, color="#5fae6a", edgecolor="black", lw=0.5, width=0.5)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008,
                f"{b.get_height():.3f}", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.set_ylabel("macro-F1 (4B + CoT, dev N=500)")
    ax.set_title("Subgroup ανά μήκος ερώτησης")
    ax.set_ylim(0, 0.75)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "subgroup_question_length.png"))
    plt.close(fig)


def fig_self_consistency():
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ns = [1, 2, 3, 4, 5, 6, 7, 8]
    macro = [0.553, 0.581, 0.561, 0.544, 0.550, 0.542, 0.547, 0.538]
    weighted = [0.670, 0.692, 0.674, 0.672, 0.667, 0.669, 0.664, 0.664]
    ax.plot(ns, macro, "o-", color="#3357a8", lw=2, ms=7, label="macro-F1 (cot_v5c + SC)")
    ax.plot(ns, weighted, "s--", color="#2a7f3f", lw=2, ms=7, label="weighted-F1 (cot_v5c + SC)")
    ax.axhline(0.575, ls=":", color="#cc7a00", lw=2, label="cot_v4 greedy test macro-F1 (0.575)")
    ax.axhline(0.725, ls=":", color="#a63333", lw=2, label="cot_v4 Kaggle LB weighted-F1 (0.725)")
    for n, m in zip(ns, macro):
        ax.text(n, m + 0.008, f"{m:.3f}", ha="center", fontsize=8, color="#3357a8")
    ax.set_xlabel("N (πλήθος sampled passes, majority vote)")
    ax.set_ylabel("F1 (test N=308)")
    ax.set_title("Self-consistency στο cot_v5c: peak στο N=2, μετά μονότονη πτώση")
    ax.set_xticks(ns)
    ax.set_ylim(0.50, 0.74)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "self_consistency.png"))
    plt.close(fig)


def fig_class_distribution():
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    counts = [2040, 1052, 356]
    colors = ["#5fae6a", "#cc7a00", "#a63333"]
    x = np.arange(3)
    bars = ax.bar(x, counts, color=colors, edgecolor="black", lw=0.5, width=0.55)
    pcts = [c / sum(counts) * 100 for c in counts]
    for b, p in zip(bars, pcts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 30,
                f"{int(b.get_height())} ({p:.1f}%)", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylabel("Αριθμός παραδειγμάτων (train, N=3448)")
    ax.set_title("Class imbalance στο QEvasion train split")
    ax.set_ylim(0, 2400)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "class_distribution.png"))
    plt.close(fig)


def fig_phase5_tradeoff():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    variants = ["cot_v4\n(reference)", "cot_v5a\n(sharpened)", "cot_v5b\n(+ mistakes)", "cot_v5c\n(best-of-both)"]
    cr = [0.557, 0.603, 0.613, 0.625]
    amb = [0.693, 0.688, 0.702, 0.679]
    cnr = [0.394, 0.327, 0.289, 0.387]
    macro = [0.548, 0.539, 0.535, 0.563]
    x = np.arange(len(variants))
    w = 0.20
    ax.bar(x - 1.5 * w, cr, width=w, label="CR F1", color="#3357a8", edgecolor="black", lw=0.4)
    ax.bar(x - 0.5 * w, amb, width=w, label="AMB F1", color="#5fae6a", edgecolor="black", lw=0.4)
    ax.bar(x + 0.5 * w, cnr, width=w, label="CNR F1", color="#cc7a00", edgecolor="black", lw=0.4)
    ax.bar(x + 1.5 * w, macro, width=w, label="macro-F1", color="#a63333", edgecolor="black", lw=0.4)
    for xi, m in zip(x, macro):
        ax.text(xi + 1.5 * w, m + 0.012, f"{m:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.set_ylabel("F1 (4B, dev N=500)")
    ax.set_title("Phase 5 — το prompt-refinement έγινε precision/recall tradeoff")
    ax.set_ylim(0.20, 0.78)
    ax.legend(loc="lower right", ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "phase5_tradeoff.png"))
    plt.close(fig)


def fig_hw_comparison():
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    systems = [
        "HW1\nLR + TF-IDF\n(macro-F1 val)",
        "HW2\nDeBERTa-base\nfine-tuned (Kaggle)",
        "HW3\nQwen3.5-4B + CoT\n(macro-F1 test)",
        "HW3\nQwen3.5-4B + CoT\n(Kaggle weighted-F1)",
    ]
    vals = [0.6017, 0.70, 0.575, 0.725]
    colors = ["#888888", "#3357a8", "#5fae6a", "#a63333"]
    x = np.arange(len(systems))
    bars = ax.bar(x, vals, color=colors, edgecolor="black", lw=0.5, width=0.55)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                f"{b.get_height():.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(systems)
    ax.set_ylabel("F1 score")
    ax.set_title("Σύγκριση: HW1 (vector) → HW2 (encoder fine-tune) → HW3 (prompting)")
    ax.set_ylim(0, 0.85)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "hw_comparison.png"))
    plt.close(fig)


def fig_4b_cot_per_class():
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    classes = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]
    prec = [0.66, 0.68, 0.33]
    rec = [0.48, 0.71, 0.50]
    f1 = [0.557, 0.693, 0.394]
    x = np.arange(len(classes))
    w = 0.25
    ax.bar(x - w, prec, width=w, label="Precision", color="#5fae6a", edgecolor="black", lw=0.4)
    ax.bar(x, rec, width=w, label="Recall", color="#3357a8", edgecolor="black", lw=0.4)
    ax.bar(x + w, f1, width=w, label="F1", color="#a63333", edgecolor="black", lw=0.4)
    for xi, v in zip(x + w, f1):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("score")
    ax.set_title("4B + CoT — per-class breakdown (dev N=500)")
    ax.set_ylim(0, 0.85)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "per_class_4b_cot.png"))
    plt.close(fig)


def main():
    fig_grid_heatmap()
    fig_grid_bars()
    fig_cot_scaling()
    fig_confusion_4b()
    fig_prompt_evolution()
    fig_subgroup_answer_length()
    fig_question_length_subgroup()
    fig_self_consistency()
    fig_class_distribution()
    fig_phase5_tradeoff()
    fig_hw_comparison()
    fig_4b_cot_per_class()
    print("All figures saved to", OUT)


if __name__ == "__main__":
    main()
