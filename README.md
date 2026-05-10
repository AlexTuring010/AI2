# ai-class-hw2

A multi-phase **NLP experimentation project** for the second homework of the **Artificial Intelligence II** course at the University of Athens (Department of Informatics & Telecommunications). Solo project.

The task: 3-class classification on the **CLARITY** dataset of question/answer pairs — label each as **Clear Reply**, **Ambivalent**, or **Clear Non-Reply**. Required models: `bert-base-uncased`, `distilbert-base-uncased`, `microsoft/deberta-v3-base`. Required training style: explicit PyTorch loop, no HuggingFace `Trainer`.

> **Final public Kaggle macro-F1: 0.70.** ~38th-place leaderboard position at submission time.

## What's in the repo

### Process artefacts (the "how" of the work)

This project was run end-to-end with disciplined process documentation. The artefacts are useful both as portfolio-of-process and as a how-to for similar experiment-heavy NLP work:

| File | Purpose |
|---|---|
| **`PLAN.md`** | The phase plan: source materials, environment setup, run modes (`smoke` / `dev` / `confirm` / `final`), dev vs. deliverable split, anti-rerun caching rules |
| **`CLAUDE.md`** | The hard rules for the project (assignment constraints, model list, source-priority order when sources conflict, what should and shouldn't be rerun) |
| **`PROJECT_STATE.md`** | The live "current state" doc — refreshed each session; tracks the active experiment, latest results, the open decision points |
| **`PROJECT_LOG.md`** | Chronological archive — entries get demoted from `PROJECT_STATE.md` here once they go stale, so the active doc stays small and scannable |
| **`EXPERIMENTS.md`** | The structured experiment log — one row per run, parameters in, metrics out |

### Notebooks

```
notebooks/
├── kaggle_final_bert.ipynb           # final-mode BERT submission notebook
├── kaggle_final_distilbert.ipynb     # final-mode DistilBERT submission notebook
├── kaggle_final_deberta.ipynb        # final-mode DeBERTa-v3 submission notebook
├── kaggle_final_deberta_large.ipynb  # exploratory: DeBERTa-large (out of scope but tested)
├── kaggle_ensemble.ipynb             # alpha-ensemble combining the survivors
└── kaggle_experiment.ipynb           # scratch / variant ablations
```

The three required models each get a self-contained Kaggle-ready final notebook. `kaggle_ensemble.ipynb` produces the alpha-weighted submission that scored 0.70.

### Report

LaTeX source for the formal report ships in `overleaf_report_package/` and a check-build copy in `overleaf_report_package_check/`. The packed `.zip` (`ai2_report_overleaf.zip`) is the version uploaded to Overleaf for compilation. Figures (`confusion_matrices.png`, `learning_curves.png`, `roc_curves.png`, `subgroup_analysis.png`) are the report-ready plots.

## Highlights from the experimentation

The full chronology lives in `PROJECT_LOG.md` and `EXPERIMENTS.md`; the highlights an external reader would care about:

### Phase 4 — the hidden `weight_decay` default

PyTorch's `AdamW` defaults to `weight_decay=0.01`, which silently regularized every prior run. Setting `weight_decay=0.0` explicitly was a single-line change that moved DistilBERT from 0.61 → 0.63 macro-F1 on a single seed. Worth documenting because it's the kind of default-induced bug that's invisible until you ablate.

### DeBERTa-v3 debugging — multi-step root-cause

DeBERTa-v3 initially produced `NaN` training losses with default optimizer settings. Setting AdamW `eps=1e-6` fixed the NaNs but the model then predicted all-Ambivalent regardless of learning rate. Diagnostics showed gradient norms much larger than the default `max_grad_norm=10.8`, getting clipped to ~1.0 at the head — DeBERTa-v3's disentangled attention produces large gradients, and the randomly-initialized pooler+classifier weren't getting enough signal.

**Fix:** differential learning rate (backbone at `cfg.lr`, head at `cfg.lr * 10`). Single-seed val F1 jumped from "stuck at chance" to **0.6962**.

### Phase 12 — numeric side-channel architecture

A Phase 11 attempt to inject hand-engineered features by adding artificial tokens like `[A_HEDGE]`, `[OV_LOW]` to the text underperformed — DeBERTa had to learn embeddings for these from scratch.

Phase 12 instead kept the Q/A text untouched and added a **side branch**: deterministic features → standardize on the train split → small MLP → concat with DeBERTa's pooled vector → classifier. This was the breakthrough: best-single seed val F1 **0.7007**, beating the prior DeBERTa baseline (0.6962). Clear Reply F1 specifically jumped to 0.6603.

The 3-seed alpha ensemble of this configuration was the **0.70 Kaggle submission**.

### Phases 13–17 — what got rejected

Honesty matters: many subsequent attempts didn't help.

- **Phase 13** — more epochs + label smoothing: rejected. Training paths shifted, didn't generalize.
- **Phase 14** — `max_length=512`: rejected for DeBERTa (helped DistilBERT in earlier phases, didn't transfer).
- **Phase 15** — richer numeric features (metadata flags, IDF-weighted overlap, TF-IDF cosine): rejected. Added too many feature families at once; over-noised the side channel.
- **Phase 16** — controlled coverage features + evasion-label probabilities (out-of-fold to avoid leakage): rejected. Valid but didn't help downstream.
- **Phase 17** — word normalization preprocessing variants + alpha ensembling: small win on local val (0.7084), uncertain on Kaggle. Default `submission.csv` reflects this.

## Run modes

Every notebook supports four modes (defined in `CLAUDE.md`):

| Mode | Purpose |
|------|---------|
| `smoke` | Tiny subset, 1 epoch — pipeline verification |
| `dev` | Fixed validation split, DistilBERT first, normal epoch budget |
| `confirm` | Full data, 2–3 seeds, report mean ± std |
| `final` | Clean packaging into Kaggle submission notebooks |

Expensive training is never the default. Ideas get screened on DistilBERT before being promoted to BERT/DeBERTa — avoids wasting compute on dead ends.

## Stack

- **PyTorch** for the training loops
- **HuggingFace Transformers** for the model checkpoints (`bert-base-uncased`, `distilbert-base-uncased`, `microsoft/deberta-v3-base`)
- **Kaggle** for compute (the assignment ran on a Kaggle competition)
- **scikit-learn** for evaluation metrics
- **LaTeX** for the report (Overleaf-compatible)

## License

[MIT](LICENSE) — applies to my own code and process documents in this repo. The CLARITY dataset and assignment-distributed materials retain their original course copyright.
