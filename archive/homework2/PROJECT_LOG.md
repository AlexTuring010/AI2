# PROJECT_LOG.md

Ιστορικό αρχείο - μετακινούνται εδώ στοιχεία από PROJECT_STATE.md όταν γίνουν stale.

---

## Phase 4 archive (2026-04-22) - wd=0.0 discovery + confirm sweep

### Key finding
PyTorch's `AdamW` defaults to `weight_decay=0.01`. Όλα τα prior runs είχαν αυτό το hidden regularizer. Θέτοντας `weight_decay=0.0` explicit:
- DistilBERT (ml=256): 0.6085 -> 0.6285 single-seed (+0.020). ml=512 extra +0.012 = 0.6405.
- BERT: 0.6181 -> 0.6072 mean (-0.011) - wd=0.01 ήταν slightly better στο average
- DeBERTa: 0.6585 -> 0.6674 mean (+0.009)

Takeaway: wd=0.0 defensible default αλλά not uniformly optimal. BERT έχει high seed-variance.

### Hyperparameter sweep results (ml=256 baseline)
| run | val_f1_macro | val_acc | f1_CR | f1_AMB | f1_CNR | notes |
|-----|-------------|---------|-------|--------|--------|-------|
| DistilBERT lr=2e-5 ep=3 ml=256 | 0.6059 | 0.6784 | 0.43 | 0.77 | 0.61 | dev baseline |
| DistilBERT lr=2e-5 ep=5 ml=256 | 0.6138 | 0.6520 | 0.54 | 0.72 | 0.58 | best@ep4; overfits after |
| DistilBERT lr=3e-5 ep=3 ml=256 | **0.6193** | 0.6901 | 0.51 | 0.77 | 0.58 | **best DistilBERT ml=256** |
| BERT lr=2e-5 ep=3 ml=256 | 0.6110 | 0.6784 | 0.49 | 0.76 | 0.58 | still improving at ep3 |
| BERT lr=3e-5 ep=3 ml=256 | 0.6024 | 0.6901 | 0.44 | 0.78 | 0.59 | best@ep2 then drops |
| BERT lr=2e-5 ep=5 ml=256 | 0.6186 | 0.6725 | 0.51 | 0.75 | 0.60 | best@ep4, noisy |

### Original confirm runs (wd=0.01 implicit - old default)
| model | val_f1_macro | val_acc | notes |
|-------|-------------|---------|-------|
| DistilBERT lr=3e-5 ep=3 ml=256 | 0.6085 ± 0.007 | 0.676 ± 0.009 | stable across seeds |
| BERT lr=2e-5 ep=5 ml=256 | 0.6181 ± 0.021 | 0.679 ± 0.014 | high variance; best@ep3-4 |
| DeBERTa lr=2e-5 ep=3 ml=256 | 0.6585 ± 0.030 | 0.705 ± 0.025 | seed=0 notably weaker |

### DeBERTa-v3 debugging history (για το report)
1. Initial run: `train_loss=nan` -> fixed με `eps=1e-6` στο AdamW
2. After fix: loss not NaN αλλά barely moves (0.94->0.91), model predicts all Ambivalent regardless of lr (1e-5 through 5e-5)
3. Diagnostic: model architecture OK (loss=1.07 σε toy input, 202 params με grad, classifier grad=5.6), αλλά max_grad_norm=10.8 κάνει clipped σε 1.0 -> 10x reduction
4. Root cause: DeBERTa-v3's disentangled attention παράγει large grad norms - randomly-init pooler+classifier δεν παίρνουν αρκετό signal με uniform lr μετά από clipping
5. Fix: differential lr (backbone: cfg.lr, head: cfg.lr * 10) -> SUCCEEDED, val F1 0.6962 single best, 0.6674 mean
6. Δεν χρειάστηκαν οι fallback options (freeze backbone, deberta-base)

Όλα αυτά είναι valuable "modifications tried" content για το report.

### First Kaggle submission (before Phase 5)
- DeBERTa seed=42 wd=0.0 ml=256 val F1 = 0.6962
- Leaderboard position: 38th
- Ensemble notebook built (9 models × 3 seeds averaged) αλλά δεν έτρεξε ακόμα

---
<!-- Εγγραφές προστίθενται χρονολογικά από κάτω -->
