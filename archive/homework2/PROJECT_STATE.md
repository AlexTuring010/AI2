# PROJECT_STATE.md

## Report draft

Drafted the Greek LaTeX report in `reports/main.tex` using the provided template and bibliography in `reports/refs.bib`. It includes the requested student-style thought process, educational explanations of BERT/DistilBERT/DeBERTa, the main failed experiments, final public Kaggle score **0.70**, plus the existing evaluation figures from `reports/figures/`.

Local PDF compilation was not run because no LaTeX executable (`pdflatex`, `xelatex`, `lualatex`, `latexmk`, or `tectonic`) is installed/in PATH on this machine.

Added the longer rewritten report as `overleaf_report_package/main_v2.tex` and mirrored it to `reports/main_v2.tex`. The new draft is about **103k Unicode characters** (145k UTF-8 bytes), keeps the original `main.tex` untouched, and is included in `ai2_report_overleaf.zip` alongside the original `main.tex`, `refs.bib`, `fphw.cls`, and figures.

## Active result (2026-04-30) - numeric-feature DeBERTa breakthrough

### Why this matters
The Phase 11 tag run injected new artificial tokens into the text, so DeBERTa had to learn random token embeddings for `[A_HEDGE]`, `[OV_LOW]`, etc. That was not faithful to the Assignment 1 feature-engineering approach and it hurt.

Phase 12 kept Q/A text unchanged and added numeric features through a side branch: deterministic features -> train-split standardization -> small MLP -> concat with DeBERTa pooled vector -> classifier.

### Results
| config | best epoch | val macro-F1 | val acc | f1[CR, AMB, CNR] | note |
|---|---:|---:|---:|---|---|
| seed 42 numfeat | 4 | **0.7007** | 0.7281 | [0.6603, 0.7750, 0.6667] | best single, beats old DeBERTa seed42 0.6962 |
| seed 0 numfeat | 3 | 0.6462 | 0.6959 | [0.5380, 0.7689, 0.6316] | weaker, overfits at ep4 |
| seed 1 numfeat | 3 | 0.6684 | 0.6959 | [0.6154, 0.7463, 0.6437] | useful but weaker, overfits at ep4 |
| equal 3-seed | — | 0.6869 | — | — | worse than seed42 because seed0/1 drag it down |
| alpha 3-seed | — | **0.7057** | — | — | weights = [0.9, 0.0, 0.1], default `submission.csv` |

Best single confusion matrix: `[[69,34,2], [34,155,13], [1,9,25]]`. Compared with old DeBERTa seed42 baseline, the main gain is Clear Reply: CR F1 reaches **0.6603**.

### Submission recommendation
Submitted Phase 12 alpha numeric-feature ensemble on Kaggle: **0.70**. This is +0.01 over the previous DeBERTa alpha ensemble (**0.68**) but still not enough.

### Phase 13 result: more epochs rejected
`notebooks/kaggle_experiment.ipynb` tested:
- seed42 numeric-feature DeBERTa, epochs=6: val **0.6876**, best@ep4
- seed42 numeric-feature DeBERTa, epochs=6 + label_smoothing=0.05: val **0.6810**, best@ep4
- two-run alpha ensemble: val **0.6958**, Kaggle **0.67**

Conclusion: the Phase 12 ep4 result was not simply undertrained. Extending the schedule changed the training path and overfit; label smoothing did not help. Stop this branch.

### Phase 14 result: longer context rejected
`notebooks/kaggle_experiment.ipynb` tested DeBERTa + numeric side-channel with `max_length=512`, seed42, batch size 8, epochs 4.

Result: val **0.6615**, acc **0.6930**, f1=[0.6161, 0.7469, 0.6216]. This is far below the ml=256 numeric-feature seed42 run (**0.7007**) and below the original DeBERTa seed42 baseline (**0.6962**).

Conclusion: full context did not help this architecture. The smaller batch and longer sequence likely made optimization noisier, and DeBERTa did not show the same ml=512 gain that DistilBERT showed.

### Phase 15 result: richer HW1 features rejected
`notebooks/kaggle_experiment.ipynb` tested richer numeric features (metadata flags, question order, sentence/punctuation structure, IDF-weighted overlap, TF-IDF-style cosine) with feature hidden size 128.

Result: val **0.6593**, acc **0.6959**, f1=[0.5591, 0.7617, 0.6571], Kaggle **0.64**.

Conclusion: adding many feature families at once over-noised the numeric side branch. The model shifted heavily toward Ambivalent on test (`221/308`) and lost the Clear Reply gain that made Phase 12 useful. This was too aggressive, not a bug in CSV formatting or label mapping.

### Phase 16 result: controlled coverage + evasion probabilities rejected
`notebooks/kaggle_experiment.ipynb` tested Phase 16:
- back to the winning ml=256, bs=16, ep=4, seed42 recipe,
- numeric profile = `coverage_evidence` (base Phase 12 features + IDF/TF-IDF coverage/evidence features, no noisy metadata block),
- append 9 predicted evasion-label probabilities from a separate TF-IDF logistic-regression evasion model,
- train rows use out-of-fold evasion predictions to avoid gold-label leakage; val/test use a final evasion model trained on the train split.

Result: val **0.6503**, acc **0.6871**, f1=[0.6216, 0.7411, 0.5882]. This is far below Phase 12 (**0.7007**) and confirms the combined coverage/evidence + evasion-probability package is not useful.

Interpretation: the separate evasion predictor was a valid way to use `evasion_label` without leakage, but its probabilities did not help the clarity classifier. It preserved some Clear Reply signal but hurt Ambivalent and Clear Non-Reply. Do not spend more time on evasion stacking unless doing a much slower, careful ablation after the report.

### Phase 17 result: real-word preprocessing ensemble is best local val
`notebooks/kaggle_experiment.ipynb` tested three controlled runs, all using the Phase 12 recipe (`ml=256`, `bs=16`, `ep=4`, seed42, base numeric side-channel):

| config | val macro-F1 | val acc | f1[CR, AMB, CNR] | note |
|---|---:|---:|---|---|
| raw Phase 12 rerun | 0.6951 | 0.7222 | [0.6476, 0.7711, 0.6667] | close to historical 0.7007; pipeline sane |
| word normalization | 0.6965 | 0.7251 | [0.6231, 0.7807, 0.6857] | tiny single-model gain vs same-session raw; AMB/CNR up, CR down |
| word normalization + evidence prefix | 0.6578 | 0.6901 | [0.5911, 0.7506, 0.6316] | weak alone, but complementary |
| alpha ensemble | **0.7084** | — | — | weights = [0.0 raw, 0.65 wordnorm, 0.35 wordnorm+evidence] |

The best single model is word normalization at **0.6965**, which is not a real breakthrough compared with historical Phase 12 **0.7007**. The useful discovery is complementarity: the evidence-prefix model is too weak alone but changes the prediction distribution enough to improve an alpha ensemble. The default `submission.csv` is now the Phase 17 alpha ensemble with test distribution: Ambivalent=183, Clear Reply=93, Clear Non-Reply=32.

### Current final recommendation
Submit the Phase 17 alpha ensemble if there is a submission slot. If it improves Kaggle, keep it. If it does not, final leaderboard best remains Phase 12 alpha at **0.70**. For one more experiment, prefer a **submission-only ensemble expansion** rather than another feature block: add the historical Phase 12 seed42/seed1 artifacts if available, or run one more `wordnorm` seed only if time allows. Avoid more evidence prefixes as single models; they are useful only as ensemble diversity.

### Active next experiment: Phase 18 focused long-answer preprocessing
`notebooks/kaggle_experiment.ipynb` tested a final serious sweep targeting the validation error-analysis pattern: long answers with mixed evidence, hedging, causal explanation, and deflection.

Queued runs:
1. raw Phase 12 rerun: full answer, base numeric features;
2. Phase 17 word normalization: full answer, base numeric features;
3. focused answer view: opening + question-overlap sentences + closing, base numeric features;
4. focused answer view + word normalization, base numeric features;
5. focused answer view + word normalization + `long_mixed` numeric profile.

The `long_mixed` profile adds long/very-long answer flags, many-sentence flag, long-low-overlap flag, evidence-vs-hedge/deflection contrast, evidence-after-deflection, and evidence density.

Result:

| config | val macro-F1 | val acc | f1[CR, AMB, CNR] | note |
|---|---:|---:|---|---|
| raw full answer | 0.6936 | 0.7193 | [0.6604, 0.7626, 0.6579] | baseline rerun, best@ep3 |
| wordnorm full answer | **0.6980** | 0.7339 | [0.6359, 0.7914, 0.6667] | best single in this session |
| focused answer | 0.6796 | 0.7222 | [0.6495, 0.7794, 0.6098] | weak alone, complementary |
| focused + wordnorm | 0.5634 | 0.6462 | [0.2778, 0.7563, 0.6562] | rejected, CR collapse |
| focused + wordnorm + long_mixed | 0.6284 | 0.6754 | [0.5029, 0.7523, 0.6301] | rejected, still too low |
| alpha ensemble | **0.7136** | — | — | top-three weights = 0.20 wordnorm + 0.55 raw + 0.25 focused |
| calibrated alpha | **0.7154** | — | — | class multipliers [0.70, 1.10, 0.70], riskier Ambivalent-heavy test distribution |

Interpretation: the focused-answer idea did not solve long answers as a single-model preprocessing, and combining it with word normalization was destructive. But focused raw answers are complementary enough to help the alpha ensemble. The most reliable submission is `/kaggle/working/submission.csv` (Phase 18 alpha, test distribution Ambivalent=187, Clear Reply=91, Clear Non-Reply=30). The highest local-val submission is `/kaggle/working/submission_calibrated.csv` (val 0.7154) but its test distribution is Ambivalent-heavy: 216/308.

### Active next experiment: Phase 19 full-context help + dual-view model
`notebooks/kaggle_experiment.ipynb` tested the user's requested final extension:

1. full raw Q/A + `long_mixed` numeric features;
2. full word-normalized Q/A + `long_mixed` numeric features;
3. focused Q/A + `long_mixed` numeric features as the simple separate-view ensemble member;
4. dual-view architecture: shared DeBERTa over full Q/A and focused Q/A, concat both pooled vectors with `long_mixed` numeric features;
5. same dual-view architecture with word normalization.

The dual-view model is implemented in `src/features_head.py` as `DualViewFeatureConcatClassifier`. It is heavier, so notebook configs used `batch_size=8` for dual-view runs. This was the “keep full context and add help” test, plus the more complex architecture the user suggested.

Result:

| config | val macro-F1 | val acc | f1[CR, AMB, CNR] | note |
|---|---:|---:|---|---|
| full raw + long_mixed | 0.6799 | 0.7105 | [0.6030, 0.7700, 0.6667] | best Phase 19 single, but below Phase 18/12 |
| full wordnorm + long_mixed | 0.6718 | 0.7018 | [0.5941, 0.7635, 0.6579] | long_mixed erases wordnorm gain |
| focused + long_mixed | 0.6497 | 0.6930 | [0.5574, 0.7617, 0.6301] | rejected |
| dual-view raw + long_mixed | 0.6665 | 0.6988 | [0.6520, 0.7448, 0.6027] | overfits after ep3, CNR weak |
| dual-view wordnorm + long_mixed | 0.6404 | 0.6637 | [0.6094, 0.7090, 0.6027] | rejected |
| alpha ensemble | 0.6888 | — | — | far below Phase 18 alpha 0.7136 |
| calibrated alpha | 0.6913 | — | — | still far below Phase 18 |

Conclusion: Phase 19 is rejected. The dual-view architecture ran cleanly but did not improve; with ~3k examples and two DeBERTa passes, it overfits/under-optimizes. The `long_mixed` numeric profile is also too noisy: when added to full raw/wordnorm text it consistently lowers performance. Keep Phase 18 alpha as the best local model.

Packaging note: Cell 8 failed while zipping run directories because Kaggle ran out of disk space (`No space left on device`). This happened after training/submission generation and does not invalidate the metrics or CSVs.

### Deliverable notebooks packaged
Regenerated the three final Kaggle deliverable notebooks:

- `notebooks/kaggle_final_distilbert.ipynb`: best DistilBERT single model (`ml=512`, `lr=3e-5`, `ep=3`, seed42), writes `submission.csv`.
- `notebooks/kaggle_final_bert.ipynb`: best BERT single model (`ml=256`, `lr=2e-5`, `ep=5`, seed1), writes `submission.csv`.
- `notebooks/kaggle_final_deberta.ipynb`: submitted DeBERTa path that produced the public Kaggle **0.70** score: Phase 12 numeric-feature ensemble (raw full Q/A, seeds 42/0/1, fixed alpha weights `[0.9, 0.0, 0.1]`), writes `submission.csv`.
- `notebooks/kaggle_final_deberta_large.ipynb`: experimental DeBERTa-v3-large variant of the Phase 18 setup (`batch_size=2`, `lr=1e-5`, `ep=3`) for a possible final high-risk run. This is not yet validated/submitted.

All three are self-contained with the current inlined project library. The builder is `scripts/build_deliverable_notebooks.js`.

Updated the final notebooks' markdown cells to Greek, in the same more student-voiced style as the report. The code cells/configs are still generated by the same builder; this was a presentation/documentation update, not a modeling change.

## Previous result (2026-04-30) - feature-token DeBERTa attempt rejected

### What ran
`notebooks/kaggle_experiment.ipynb` ran two final high-upside variants:

1. `microsoft/deberta-v3-base`, ml=256, seed=42, ep=4, `use_engineered_cue_tokens=True`
2. same config + `use_negation_markers=True`

Both used explicit `weight_decay=0.0`, matching the best DeBERTa setup. The cue tokens encoded question intent, answer style/evidence, Q-A lexical overlap, and answer length.

### Results
| config | best epoch | val macro-F1 | val acc | f1[CR, AMB, CNR] | verdict |
|---|---:|---:|---:|---|---|
| DeBERTa + cues | 3 | 0.6191 | 0.6696 | [0.4940, 0.7460, 0.6173] | rejected |
| DeBERTa + NEG + cues | 4 | 0.6077 | 0.6696 | [0.3867, 0.7603, 0.6761] | rejected |

Calibration on the cue-only run improved val from **0.6191** to **0.6499** with multipliers `[1.20, 1.00, 0.65]`, but this is still far below the best single DeBERTa baseline (**0.6962**) and current DeBERTa val ensemble (**0.7070**). Do **not** submit these unless every other submission slot is expendable.

### Interpretation for report
This is useful negative evidence: feature engineering that helped the HW1 linear/TF-IDF system did **not** transfer when injected as special tokens into a strong DeBERTa model. The likely reason is that the cues are too coarse and compete with DeBERTa's learned semantic representation. Adding `[NEG]` on top made CR worse, suggesting stacked handcrafted markers can distort the decision boundary.

### Interpretation at the time
The feature-token branch is dead. The new active branch above tests the same intuition with the correct numeric-feature architecture.

## Session end snapshot (2026-04-23)

### Kaggle leaderboard status
- **#29 @ 0.68 test F1** (DeBERTa α-weighted 3-seed ensemble, submitted 2026-04-23).
- Previous: #38 (DeBERTa seed=42 solo).
- Top-1 = 0.75, top-10 cutoff ~ 0.71, 0.72 cluster to break through. Gap from us to top-10 ~ +0.03.
- Deadline 2026-05-03 23:55 (~10 days).

### What shipped today
1. **Fixed multi-task head architecture** (`src/multitask.py`): added `pre_classifier + ReLU + Dropout` between pooler and heads so multi-task model matches `DistilBertForSequenceClassification`'s 2-layer structure. Removed Phase 6's confound.
2. **DistilBERT aux=0.3 3-seed confirmed winner** (0.6367 ± 0.012 vs baseline 0.6167 ± 0.021, +0.020 paired, variance halved).
3. **DeBERTa multi-task aux retired**: aux hurts strong backbones (CR +0.009 / AMB -0.041 / CNR -0.114). BERT tied. Capacity-inverse finding.
4. **DeBERTa 3-seed α-weighted ensemble**: val 0.7070 (+0.011 vs best single seed @ 0.6962), α = 0.8·seed42 + 0.0·seed0 + 0.2·seed1. Local CPU test inference (`scripts/deberta_ensemble_test.py`) → `kaggle/ensemble_artifacts/submission.csv` → submitted → #29.
5. **Cross-model ensemble probed and rejected**: on DeBERTa's 90 val errors, DistilBERT-aux right 21/90 (23%), BERT 17/90 (19%) — below chance. α-grid always collapses to 100% DeBERTa. No cross-family diversity to exploit.

### Next-session plan (priority-ordered)

**Track A — scale DeBERTa-internal diversity** (my original proposal, model-side):
- A1. **DeBERTa ml=512 seed=42** (queued in `notebooks/kaggle_experiment.ipynb` Cell 3; ~25-30 min T4, bs=8). Pass bar = val > 0.6962. If pass → seeds 0, 1 next.
- A2. **DeBERTa + NEG 3 seeds** (~50 min). Orthogonal input preprocessing; never tried on DeBERTa.
- A3. **DeBERTa two-stage pretrain+finetune** (~25 min gate). Pretrain on evasion 9-class, finetune on clarity 3-class. Riskier but biggest potential single-model lift.

**Track B — engineered features** (user intuition — signals WE compute from Q/A, NOT dataset-provided columns like gpt3.5_*):
- B1. **Lexical cue counts** as special-token prefixes: hedge density (maybe/might/probably), negation count (not/never/no), definiteness (yes/no/never as first word), question-mark count in answer. Bucket each into 3-4 buckets → `[HEDGE_HIGH]` `[NEG_2]` etc. Inject as prefix tokens on the answer.
- B2. **Structural features** concatenated to [CLS] pooled vector before classifier: `len(A)`, `len(A)/len(Q)`, sentence count in A, punctuation density. Adds ~5 dims to the head input; requires custom classifier (wrap AutoModel, route numeric features around the pooler).
- B3. **Q-A alignment signal**: token overlap ratio (question content words present in answer), cosine sim of (Q, A) sentence embeddings from a frozen mini-encoder (e.g. all-MiniLM-L6). Low overlap → Ambivalent / Non-Reply bias. Inject as bucketed token or numeric feature.
- B4. **Discourse markers** as prefix tokens: `[PUSHBACK]` for but/however/well-clause openings, `[CONFIRM]` for so/therefore/exactly, `[DEFLECT]` for "let me just" / "the real issue is", on top of existing `[NEG]`.
- B5. **Input reformulation**: entailment-style framing `"Does '{question}' get answered by '{answer}'? yes/maybe/no"` → 3-class classification becomes an NLI-style task. Or: contrastive pairs during training (same Q, multiple answers with different clarity labels).
- B6. **Threshold tuning / calibration** (post-hoc, cheap): DeBERTa-α has CR recall = 0.56 (weakest class). Tune per-class decision thresholds on val instead of argmax.

**Injection methods to implement in `src/` once we pick one:**
- (a) Prefix special tokens: extend `apply_question_prefix` / `apply_negation_markers` pattern. Minimal plumbing; works with existing training loop.
- (b) Numeric features concatenated to pooled [CLS]: requires a small custom classifier head (new module `src/features_head.py`: DeBERTa backbone → pooler → concat(numeric_features) → Linear → class logits).

**Track C — cheap wins available now**:
- C1. Upload more DeBERTa seeds (add seed=2, seed=3 at ml=256). Pure ensemble-diversity scaling. ~10 min/seed on T4.
- C2. Locally infer `submission_equal_weight.csv` on test (already generated) — submit as ablation baseline to measure "disagreement cost" between α-weighted and equal-weight submissions.

**User belief to remember**: real gains likely come from features / smart preprocessing, not more seeds. Track B is weighted higher than model size scaling.

### Handoff notes
- `/tmp/ai2venv/` has torch+transformers==4.44.0 for local CPU inference. Reusable.
- `kaggle/ensemble_artifacts/` holds 3 DeBERTa seed logits + 2 submission CSVs. Safe to add more checkpoints' logits and re-sweep α locally.
- `scripts/deberta_ensemble_val.py` = α-grid on val. `scripts/deberta_ensemble_test.py` = CPU inference + submission.
- Cell 3 of `notebooks/kaggle_experiment.ipynb` is ready to run: DeBERTa ml=512 seed=42.

---

## Τρέχουσα φάση
**Phase 7b DONE** (2026-04-23, Kaggle T4). Multi-task με **fixed 2-layer head** wins on DistilBERT.
- **aux=0.3 fixed head 3-seed: mean 0.6367 ± 0.012 vs baseline 0.6167 ± 0.021 (+0.020 paired) vs NEG 0.6235 ± 0.022 (+0.013 paired)**. Per-seed: [0.6454, 0.6231, 0.6415].
- **aux also halves seed variance** (0.012 vs 0.021/0.022) - evasion aux acts ως regularizer.
- aux beats NEG 2-of-3 seeds (seed=0: +0.016, seed=1: +0.027, seed=42: -0.003 near-tie).
- Phase 6 aux failures (max 0.6355) were due to head-architecture confound, not the aux loss.
- Best@ep2 σε 2 of 3 seeds (mild overfit στο ep3).
- **Multi-task aux=0.3 is the new DistilBERT winner** and the new candidate to port σε BERT + DeBERTa.

**Deadline: 2026-05-03, 23:55**. Σήμερα: 2026-04-23. ~10 μέρες remaining.

**Deadline: 2026-05-03, 23:55**. Σήμερα: 2026-04-22. ~11 μέρες remaining.

## Phase 7 results (DistilBERT ml=512, aux=0.3 με fixed 2-layer head)

| seed | baseline | NEG | **aux=0.3 fixed** | Δ aux vs baseline |
|------|---------:|----:|-------------------:|------------------:|
| 42 | 0.6405 | 0.6484 | **0.6454** | +0.0049 |
| 0 | 0.6049 | 0.6074 | **0.6231** | +0.0182 |
| 1 | 0.6046 | 0.6146 | **0.6415** | +0.0369 |
| **mean ± std** | 0.6167 ± 0.021 | 0.6235 ± 0.022 | **0.6367 ± 0.012** | **+0.0200** |

- aux beats baseline σε 3/3 seeds, beats NEG σε 2/3 seeds.
- aux 3-seed std = 0.012 (vs 0.021/0.022) - auxiliary evasion signal sharpens σε μικρότερο variance.
- Best@ep2 σε 2/3 seeds (ep3 slight overfit: val_loss plateau while train drops).
- **Phase 6 aux failures ήταν head-architecture confound, όχι aux loss**.

Ensemble triple local val (Phase 5): 0.6561 με weighted 0.50·clarity + 0.45·neg + 0.05·evasion_mean. Phase 7 aux single run (0.6454) είναι λιγότερο από ensemble, αλλά είναι ένα model-artifact (not a post-hoc combination) που μπορεί να portάρει σε BERT/DeBERTa.

## Phase 6 multi-task aux results (thin-head confound, Phase 5→6 table context)

| config | seed | val F1 | verdict |
|--------|------|-------:|---------|
| aux λ=0.1 (thin) | 42 | 0.6146 | -0.026 vs baseline |
| **aux λ=0.3 (thin)** | 42 | **0.6355** | -0.005 best thin-head aux |
| aux λ=0.5 (thin) | 42 | 0.6177 | -0.023 |
| neg + aux 0.3 (thin) | 42 | 0.6183 | no stacking |
| **aux λ=0.3 (fixed head, Phase 7a)** | 42 | **0.6454** | **+0.010 vs thin, +0.005 vs baseline** |

## Phase 5 screening (context)

## Phase 5 screening results (DistilBERT ml=512, best-epoch val F1-macro)

| config | seed | val F1 | val acc | f1[CR,AMB,CNR] | Δ vs baseline | verdict |
|--------|------|-------:|--------:|----------------|--------------:|---------|
| **neg** (spaCy [NEG]) | 42 | **0.6484** | 0.7076 | [0.481, 0.789, 0.676] | **+0.0079** | **winner** |
| baseline (clarity, CE) | 42 | 0.6405 | 0.6959 | [0.497, 0.777, 0.648] | - | anchor |
| qtoks ([AFFIRM_Q]/[MULTI_Q]) | 42 | 0.6385 | 0.6988 | [0.497, 0.779, 0.640] | -0.0020 | tie, drop |
| focal γ=2.0 | 42 | 0.6284 | 0.6901 | [0.465, 0.773, 0.648] | -0.0121 | hurts CR |
| tgt=evasion | 42 | 0.6042 | 0.6754 | [0.423, 0.767, 0.623] | -0.0363 | rejected |
| tgt=evasion | 0 | 0.5982 | 0.6784 | [0.408, 0.771, 0.615] | -0.0423 | rejected |
| tgt=evasion | 1 | 0.5658 | 0.6520 | [0.379, 0.752, 0.567] | -0.0747 | rejected |
| combined (evasion+focal+qtoks+neg) | 42 | 0.5299 | 0.6520 | [0.240, 0.765, 0.585] | -0.1106 | rejected |
| cw + focal γ=2.0 | 42 | 0.4383 | 0.4152 | [0.493, 0.254, 0.568] | -0.2022 | **catastrophe** (AMB collapses) |

**Evasion 3-seed: mean = 0.5894 ± 0.017** (tight spread -> hypothesis solidly rejected ΩΣ STANDALONE, not noise).

### Ensemble rescue (local α-sweep, 2026-04-22)
Το evasion run δεν ήταν wasted. Weighted ensemble στο val:

| ensemble | weights | val F1 | Δ vs best standalone (neg=0.6484) |
|----------|---------|-------:|----------------------------------:|
| clarity + neg (pair) | 0.05 / 0.95 | 0.6523 | +0.0039 |
| **clarity + neg + evasion_3seed_mean (triple)** | **0.50 / 0.45 / 0.05** | **0.6561** | **+0.0077** |

Per-class του best triple: CR=0.500 AMB=0.792 CNR=0.676. Το 5% evasion nudge tips a few borderline Reply predictions (CR 0.490 -> 0.500 σε σχέση με clarity+neg pair).

**Insight για το report**: οι fine-grained evasion labels έχουν auxiliary signal που το plain 3-class CE loss δεν μπορεί να εκμεταλλευτεί — όταν δοθούν στο model ως primary target υπάρχει probability diffusion που σπάει το Reply class, αλλά όταν combined ως weighted auxiliary με πολύ μικρό βάρος διορθώνουν borderline predictions.

**Caveat**: weights val-set-tuned σε 342 examples με 3 free parameters - μικρό overfitting risk. Cross-model ensemble (DistilBERT+BERT+DeBERTa) πιθανόν dominates to triple DistilBERT-only.

Script: `scripts/ensemble_alpha_sweep.py`.

## Phase 8 result (DeBERTa aux=0.3, 2026-04-23) — RETIRED for DeBERTa

| epochs | val F1 | CR | AMB | CNR | val_loss@ep3 | val_loss@best |
|-------:|-------:|---:|----:|----:|-------------:|--------------:|
| ep=3 (8a) | 0.6033 | 0.413 | 0.758 | 0.639 | 0.750 | 0.750 |
| ep=5 (8b) | 0.6473 | 0.620 | 0.742 | 0.580 | 0.730 (min) | 0.794 |
| baseline | **0.6962** | 0.611 | 0.783 | 0.694 | — | — |

Gap vs baseline (ep=5): CR +0.009, AMB -0.041, CNR **-0.114**. val_loss bottoms @ep3 then rises -> not undertraining, overfitting με task confusion. **Hypothesis**: evasion taxonomy splits AMB into 6 sub-classes και CNR into 3, οπότε το aux loss fragment-άρει αυτά που DeBERTa's strong backbone ήδη σωστά grouped.

**Multi-task δεν κάνει transfer σε DeBERTa**. Key insight για το report: auxiliary granularity hurts when the backbone is already sufficient.

## Ship plan (revised post Phase 9a)

| model | winning config | 3-seed mean | best single-seed | Kaggle notebook |
|-------|----------------|------------:|-----------------:|-----------------|
| DistilBERT | **aux=0.3 fixed head** | **0.6367 ± 0.012** | 0.6454 (seed=42) | regen needed |
| BERT | **baseline** OR NEG (TBD 30 min) | 0.6072 ± 0.030 | 0.6417 (seed=1) | regen needed |
| DeBERTa | **baseline** (multi-task retired) | 0.6674 ± 0.032 | 0.6962 (seed=42) | **already submitted, 38th place** |

## Phase 9a result (BERT aux=0.3 seed=42, 2026-04-23)
**0.5905 @ ep4** (baseline seed=42 = 0.5927; Δ = -0.002, tied). Per-class: CR +0.061, AMB -0.037, CNR -0.030. **Same task-confusion pattern as DeBERTa** — aux shifts decision boundary toward CR at cost of AMB/CNR. On BERT the gains/losses cancel to wash; on DeBERTa losses win.

**Aux transfer degrades with backbone capacity**: DistilBERT +0.020 / BERT -0.002 / DeBERTa -0.049. Clean report finding: auxiliary granularity helps weak representations (DistilBERT), is neutral for medium (BERT), hurts strong (DeBERTa).

Multi-task retired for BERT as well.

## IMMEDIATE NEXT ACTION — freeze & ship or test BERT NEG

Decision point. BERT aux failed (wash). Options:

**Option 1 (ship now, fastest): freeze configs, regenerate 3 final notebooks** with DistilBERT aux=0.3, BERT baseline (seed=1 = 0.6417), DeBERTa baseline. Submit, write report. Story is already good (aux-capacity inverse relationship is a real finding).

**Option 2 (one more BERT test, ~30 min): BERT + NEG 3 seeds**. Complete the ablation matrix — we'd know whether NEG transfers to BERT (it barely helped DistilBERT at +0.007). Predict marginal. If it wins, BERT notebook uses NEG; if not, ship baseline.

Recommendation: **Option 2** — it's cheap, closes the story, gives us one more report row. If BERT NEG lifts by ≥0.01 it goes into the BERT notebook; otherwise BERT ships baseline.

After that: regenerate 3 final Kaggle notebooks + submit + report.

## Key findings για το report

### Τι κέρδισε: negation markers
- CR 0.497 -> 0.481 (-0.016): λίγο χειρότερο στο Clear Reply
- AMB 0.777 -> 0.789 (+0.012): μικρό lift
- **CNR 0.648 -> 0.676 (+0.028)**: μεγάλο lift στο Clear Non-Reply
- Interpretation: τα Clear Non-Reply και Ambivalent responses συχνά περιέχουν negated verbs ("I can't say...", "that's not my area...") - το [NEG] prefix δίνει explicit syntactic cue που το transformer δεν βγάζει από μόνο του με τόσο μικρό train set.

### Τι απορρίφθηκε: evasion-based training
- Μέση ρίψη -0.051 F1 κατά από clarity baseline (3 seeds, tight variance).
- Root cause hypothesis: 3412 train rows / 9 evasion classes ≈ 379/class. Κάποιες evasion classes (π.χ. Claims ignorance, Clarification) έχουν πολύ λίγα examples. Ο classifier δεν μαθαίνει good representations και το collapse-to-clarity step δεν μπορεί να επανακτήσει.
- Ambivalent F1 stays at 0.77 (unchanged) αλλά **Clear Reply F1 κατρακυλά από 0.497 σε 0.42**: η μοναδική Reply evasion class ("Explicit") μένει ίσως με πολύ λίγη probability mass όταν το softmax διασκορπίζεται σε 6 Ambivalent sub-classes.
- **Test split έχει evasion_label = '' σε όλα τα 308 rows** (verified 2026-04-22), οπότε use-as-feature δεν ήταν εναλλακτική.

### Τι ήταν ουδέτερο: question tokens (qtoks)
- Essentially tied με baseline (-0.002). Το multiple_questions/affirmative_questions signal είναι πιθανόν ήδη captured στο raw text (π.χ. question marks, "Do you..." patterns).

### Τι χτύπησε: focal alone
- -0.012 F1. Focal γ=2.0 down-weights easy examples. Το Reply class είναι "harder than Ambivalent" σε αυτό το dataset (F1 0.497 vs 0.777), οπότε focal loss δεν επικεντρώνεται στο σωστό failure mode.

### Catastrophic: cw + focal combined
- -0.202 F1, **Ambivalent F1 από 0.777 σε 0.254**. Class weights ήδη up-weight minorities (CR, CNR). Focal επιπλέον down-weights confident predictions (όλα τα easy Ambivalent). Το double-penalty σπρώχνει το model να προβλέπει παντού Reply/Non-Reply → AMB collapse.

## Delivery model
1. **Three Kaggle notebooks** (DistilBERT, BERT, DeBERTa), executed με outputs, shared με TA `despinap`. **Winning config: `use_negation_markers=True`**.
2. **One PDF report** στο e-class, `[full-id].pdf`, Greek, με notebook links.

## Known best results (post Phase 5 screening, pending 3-seed confirm)

| model | baseline config | val_f1_macro (mean ± std, 3 seeds) | best_single_seed | best_single_f1 | Phase 5 neg single seed=42 |
|-------|-----------------|-------------------------------------|------------------|----------------|----------------------------|
| DistilBERT | lr=3e-5 ep=3 ml=512 wd=0.0 | 0.6167 ± 0.021 | 42 | 0.6405 | **0.6484** (+0.008) |
| BERT | lr=2e-5 ep=5 ml=256 wd=0.0 | 0.6072 ± 0.030 | 1 | 0.6417 | TBD |
| DeBERTa | lr=2e-5 ep=3 ml=256 wd=0.0 + diff lr | 0.6674 ± 0.032 | 42 | **0.6962** | TBD |

HW1 baseline: 0.6017. Kaggle leaderboard baseline: 0.56. Πρώτη submission (DeBERTa 0.6962): 38th place.

## Phase 6 code state (multi-task wiring, 2026-04-22)
- `src/multitask.py` (**new**): `MultiTaskClassifier` dual-head wrapper (AutoModel backbone + clarity_head 3-class + evasion_head 9-class), `load_multitask_model` helper. Συμβατό με resize_token_embeddings για qtoks/neg.
- `src/config.py`: new `aux_evasion_weight: float = 0.0`. run_id suffix `_aux{weight}`. Mutually exclusive με `train_target="evasion"` (raises ValueError).
- `src/tokenization.py`: **fixed pre-existing bug** όπου `tokenize_pairs` αγνοούσε το `label_col` param (χρησιμοποιούσε πάντα `df["label_id"]`). Added `tokenize_pairs_multitask` επιστρέφει 4-tensor (ids, mask, clarity, evasion).
- `src/train.py`: `train_one_epoch` τώρα δέχεται 4-tensor batches όταν `aux_evasion_weight>0`, combined loss `CE_clarity + λ * CE_evasion`.
- `src/evaluate.py`: handles 4-tensor batches (unpacks first 3), tolerates None loss.
- `src/experiment.py`: branches on `multitask` flag, loads MultiTaskClassifier, uses multitask tokenization, passes `aux_evasion_weight` στο train loop. DeBERTa differential lr aware των multi-task heads (`clarity_head + evasion_head` αντί για `pooler + classifier`).
- `scripts/build_library_cell.py` (**new**): auto-regenerates notebook Cell 2 από src/ modules (dependency order, strips intra-package imports). Run after src/ changes.
- `notebooks/kaggle_experiment.ipynb` Cell 2 regenerated (953 lines). Cell 3 replaced με Phase 6 sweep (6 configs).
- Smoke test passed end-to-end (src/ + notebook library cell both).

## Phase 6 sweep queued (6 configs, ~40 min T4)
- Block A (neg confirm): seeds 0, 1 με `use_negation_markers=True`
- Block B (aux evasion λ sweep): seed=42 με λ ∈ {0.1, 0.3, 0.5}
- Block C (combined): seed=42 `use_negation_markers=True + aux_evasion_weight=0.3`

## Phase 7 code state (head-architecture fix, 2026-04-23)
- `src/multitask.py`: added `pre_classifier = nn.Linear(hidden, hidden)` + `ReLU` + `Dropout(0.2)` between pooler and both heads. Matches `DistilBertForSequenceClassification`'s 2-layer head (was single `nn.Linear` before). Fixes confound that made Phase 6 aux runs structurally thinner than baseline.
- `src/experiment.py`: DeBERTa differential-lr head group now includes `pre_classifier.parameters()` (10× backbone lr).
- `notebooks/kaggle_experiment.ipynb`: Cell 0 banner → "Phase 7 hypothesis test". Cell 2 regenerated (967 lines). Cell 3 → single run `distilbert aux=0.3 seed=42`.
- Artifacts: `kaggle/session_models3/..._seed42_aux0.3` (Phase 7 fixed-head) vs `kaggle/session_models2/..._seed42_aux0.3` (Phase 6 thin-head).

### Phase 7 result (2026-04-23)
**Hypothesis confirmed.** aux=0.3 seed=42 με fixed head = **0.6454** (best@ep2), +0.0099 vs Phase 6 thin head (0.6355), +0.0049 vs baseline (0.6405). Per-class f1=[0.497, 0.783, 0.657] - CR flat, AMB +0.006, CNR +0.009 vs baseline. val_loss plateau 0.734→0.734 while train drops 1.29→1.12 στο ep3 (slight overfit; best@ep2 is the real result).

**Caveat**: still single seed - +0.005 είναι smaller than across-seed std (0.021). Need 3 seeds.

## Code state (Phase 5 wiring, still current)
- `src/config.py`: Phase 5 flags wired. run_id suffixes `_cw`, `_tgtevasion`, `_focal{g}`, `_qtoks`, `_neg`.
- `src/data.py`: EVASION taxonomy constants. **Empirical train = 3412 rows**.
- `src/tokenization.py`: `ensure_special_tokens`, `apply_question_prefix`, `mark_negations` (spaCy lazy-loaded).
- `src/evaluate.py`: optional `evasion_to_clarity` collapse; saves log(clarity_probs) για downstream softmax compat.
- `src/train.py`: `focal_loss` helper, `loss_type`/`focal_gamma`/`class_weights` pass-through.
- `src/experiment.py`: branches on train_target, applies prefix+negation transforms.
- `notebooks/kaggle_experiment.ipynb`: Phase 5 screening sweep executed σήμερα, artifacts στο `kaggle/session_models/`.
- `kaggle/session_models/`: 9 directories, κάθε ένα best_model.pt + metrics.json + config.json + val_preds.npy + val_labels.npy.

## Report evidence tracker
- [ ] Class distribution (cite από HW1)
- [x] Hyperparameter ablation: lr (2e-5 vs 3e-5), epochs (3 vs 5) - DistilBERT done
- [x] Input formulation: concat_sep vs two_segment (inconclusive για DistilBERT χωρίς token_type_ids)
- [x] Model comparison (DistilBERT vs BERT vs DeBERTa) με mean±std - DONE
- [x] **Phase 5 ablations** (evasion / focal / qtoks / neg / combined) - screening DONE, needs 3-seed confirm για neg
- [ ] Subgroup analysis (q_len_bin, a_len_bin)
- [ ] Error analysis με concrete examples
- [x] Confusion matrices: DistilBERT + BERT + DeBERTa best - saved
- [x] Learning curves: DistilBERT (both lr), BERT ep=5 trajectory, DeBERTa
- [ ] "Modifications tried": wd discovery, DeBERTa grad-clip+differential lr, Phase 5 ablations
- [ ] **Why evasion-based training failed** - small-train / granularity argument, with 3-seed evidence

## Remaining tasks
- [ ] 3-seed confirm του neg στο DistilBERT (seeds 0 + 1)
- [ ] Port winning config σε BERT + DeBERTa (3 seeds each)
- [ ] Regenerate `scripts/build_final_notebooks.py` MODELS dict με `use_negation_markers=True`
- [ ] Build 3 final Kaggle notebooks
- [ ] Submit neg-variant DeBERTa στο Kaggle - compare με 38th place
- [ ] Ensemble notebook: decide αν παραμένει στο delivery ή dev-only
- [ ] Generate report figures locally: learning curves, CM, ROC
- [ ] Subgroup + error analysis
- [ ] Write Greek PDF report

## Open questions
1. **Θα κρατήσει το +0.008 του neg σε 3 seeds;** (αν όχι, δεν ports σε BERT/DeBERTa)
2. Θα transfer-άρει το neg signal σε DeBERTa-v3 που ήδη έχει stronger syntax awareness; (πιθανόν μικρότερο gain)
3. Submit μόνο neg-variant DeBERTa στο Kaggle ή ensemble baseline + neg;
4. Ensemble - worth το effort για τον report ή να μείνει dev-only;
