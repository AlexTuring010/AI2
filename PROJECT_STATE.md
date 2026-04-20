# PROJECT_STATE.md

## Τρέχουσα φάση
Phase 3 — Architecture αποφασισμένη, ready to implement src/ skeleton

## Ολοκληρωμένα
- [x] Phase 0: Reference materials συλλεγμένα σε refs/
- [x] Phase 1: Project directory initialized, .gitignore, git remote
- [x] Phase 2: /init run, CLAUDE.md created
- [x] Phase 3 kickoff: spec διαβάστηκε, clarifications συνοψίστηκαν, architecture αποφασίστηκε

## Delivery model (αποφασισμένο)
Τα μόνα αντικείμενα που βαθμολογούνται είναι:
1. **Τρία Kaggle notebooks** (ένα ανά μοντέλο), executed με outputs visible, shared με TA `despinap`, not public. Κάθε ένα εξάγει `submission_<model>.csv`.
2. **Ένα PDF report** στο e-class, όνομα `[full-id].pdf`, γραμμένο στα Ελληνικά, με links στα τρία notebooks.

Τίποτα άλλο δεν παραδίδεται. Το `src/`, τα `scripts/`, τα dev notebooks, τα tracking files (`PROJECT_STATE.md`, `EXPERIMENTS.md`, `PROJECT_LOG.md`) είναι dev-only scaffolding — δεν φεύγουν από το machine μας.

**Deadline:** 2026-04-24, 23:55.

## Packaging strategy (αποφασισμένο)
**Inline-at-build**: ένα packaging script θα διαβάζει τα `src/*.py` και θα τα κάνει inject σε μια "library" cell στην κορυφή κάθε Kaggle notebook template. Τα τρία final notebooks είναι build outputs.

**Γιατί όχι "src/ as Kaggle dataset"**: Kaggle competition notebooks συχνά τρέχουν με internet off ή ασταθώς· επιπλέον, το dataset-sharing προσθέτει TA-facing friction (πρέπει να attach-άρουν private dataset για να γίνει grade). Τα inlined notebooks είναι self-contained — ο TA πατάει "Run All" και δεν χρειάζεται τίποτα άλλο.

**Τελικό run constraint**: κάθε Kaggle notebook πρέπει να τρέξει end-to-end στο Kaggle compute (≤12h GPU) με cell outputs saved. Απαιτείται actual Kaggle execution κοντά στο τέλος — όχι μόνο local testing.

## Αποφάσεις που κλείδωσαν (πρόσφατες)
- **HW1 preprocessing δεν μεταφέρεται** (direct read στο `refs/assignment1_report.ipynb`):
  - HW1 έκανε: lowercase, Unicode NFKC, contraction expansion, surface-pattern → OOV tokens (`date_tok`, `year_tok`, …), punctuation tokens (`qmark`, `emark`, `ellipsis`), negation `neg_X` prefix via spaCy, lemmatization, `[SEP]` concat, 120 engineered features.
  - Για transformers: **όλα off**. Οι OOV tokens σπάνε τα pretrained embeddings, το lemmatization καταστρέφει το subword signal, τα cased/uncased tokenizers χειρίζονται normalization μόνα τους.
  - Κρατάμε ΜΟΝΟ: (α) τα 2 data cleaning steps, (β) `class_weight='balanced'` → weighted CE ως ablation, (γ) HW1 finding «Ambivalent↔Clear Reply το δυσκολότερο boundary» για το error analysis, (δ) HW1 best macro-F1=0.6017 ως baseline στο report.

- **input_fmt δεν επηρεάζει length**: Το `two_segment` (`text=q, text_pair=a`) και το `concat_sep` (`q [SEP] a`) δίνουν ΙΔΙΟ token count — το `[SEP]` κοστίζει ένα token και στις δύο περιπτώσεις. Η επιλογή αφορά τα segment embeddings (token_type_ids), όχι το truncation. Default: `two_segment`, αλλά θα κάνουμε ablation στο report.
- **max_length tiered strategy** (από `scripts/measure_token_lengths.py` στο cleaned set):
  - Measured percentiles (BERT/DistilBERT tokenizer): p50=288, p75=578, p90=883, p95=1122, p99=1644, max=2595. DeBERTa-v3 ελαφρώς πιο συμπαγές (~2% κοντύτερα tokens).
  - **Hard cap = 512** για όλα τα models (position embeddings). Άρα το p95=1122 είναι εκτός. Δεν θα τρέξουμε sliding-window/chunking — θα αναφέρουμε το truncation ως design choice στο report.
  - **Tiered**: smoke/hyperparam screening → `max_length=128`; dev baseline → `256`; confirm runs → `512`. Screen cheap, σπαταλάμε GPU μόνο στο confirm.

## Open questions / αποφάσεις εκκρεμούν
1. **Class imbalance strategy**: Weighted CrossEntropyLoss ή plain? → Start plain, add weighted version ως hyperparameter experiment.
2. **Kaggle data path**: Πού expose-άρει το Kaggle το CLARITY dataset; Να επιβεβαιωθεί input path στο Kaggle environment.
3. ~~Cleaning count discrepancy~~ **Resolved**: Το 3412 προέκυπτε από το ordering — dropping corrupted πρώτα έκανε invisible 1 conflict-pair με overlap στο corrupted range (1 πλευρά corrupted → survivor δεν φαίνεται πια ως duplicate). Γυρίσαμε σε **forum-literal ordering** (conflicts πρώτα, σε raw data, drop και τα 24 rows — μετά τα 14 remaining corrupted). Τελικό cleaned = **3410** όπως περίμενε η memory/HW1.

## Επόμενη ενέργεια
Υλοποίηση `src/config.py`, `src/data.py`, `src/cache.py` → smoke test για να επιβεβαιωθεί ότι το 24-row deduplication και το split creation λειτουργούν πριν αγγίξουμε το training.

## Report template gaps (αναγνωρισμένο)
Το template (`refs/AI2_Template.zip/main.tex`) είναι γενικό course scaffold — ΔΕΝ έχει sections για τα παρακάτω που ζητάει το HW2 spec και που πρέπει να προσθέσουμε:
- **Error Analysis** (spec: "central component")
- Model-by-model comparison (BERT/DistilBERT/DeBERTa)
- Modifications tried και τα effects τους
- Subgroup analysis (question/answer length bins)
- Input representation choice (explicit statement)
- Kaggle notebook links

Επιπλέον housekeeping πριν το submit:
- Strip/comment out όλα τα `\instructornote{...}` (είναι guidance για τον φοιτητή)
- Populate `\fillin{<first-name last-name>}` και `\fillin{<sdiYYZZZZZ>}`
- Αντικατάσταση placeholder refs στο `refs.bib` με πραγματικά ML papers (BERT, DistilBERT, DeBERTa-v3, HF Transformers)
- Περιεχόμενο γραμμένο στα Ελληνικά μέσα στο English LaTeX scaffold

## Report evidence tracker
- [ ] Class distribution (από HW1 — απλώς cite)
- [ ] Input formulation comparison results
- [ ] Hyperparameter ablation table
- [ ] Model comparison (DistilBERT vs BERT vs DeBERTa) με mean ± std
- [ ] Subgroup analysis (question length bins, answer length bins)
- [ ] Error analysis με concrete examples
- [ ] Confusion matrices για κάθε model
- [ ] Learning curves (train loss και val F1 vs epoch)

### Content για «Preprocessing & modifications» section του report (πρέπει να γραφτεί)
- [ ] **Data cleaning rules** που εφαρμόσαμε: 14 corrupted rows (index 1870–1883, HW1 OOV analysis) + 24 conflicting-label duplicates (forum instruction) → 3410 training rows. Ordering (conflicts first, corrupted second) για να matchάρει την forum οδηγία «drop all 24».
- [ ] **Input representation choice** (spec το ζητάει explicit): επιλέξαμε native two-segment tokenizer input (`tokenizer(question, answer)`). Ablation report vs manual `[SEP]` concat. Token-length analysis (p50=288, p95=1122, hard cap 512) και decision για truncation αντί για sliding-window.
- [ ] **max_length tiered strategy**: 128 για screening, 256 για dev, 512 για final confirm. Justification: p95=1122 εκτός του 512 ceiling, sliding-window out-of-scope.
- [ ] **HW1 preprocessing που ΔΕΝ μεταφέραμε και γιατί**: lowercase (uncased tokenizers το κάνουν· σπάει DeBERTa-v3), contraction expansion (subword tokenizers handle "don't" natively), surface-pattern OOV tokens (`date_tok`, `year_tok`, κλπ. — δεν υπάρχουν στα pretrained embeddings), punctuation tokens (`qmark`, `emark`, `ellipsis` — ίδιο OOV issue), negation `neg_X` prefix (transformers χειρίζονται negation contextually), lemmatization (καταστρέφει το subword morphological signal).
- [ ] **120 engineered features από HW1 που δεν μεταφέραμε**: hybrid model (concat features στο [CLS]) θα απαιτούσε porting ~800 lines spaCy/regex pipeline σε κάθε Kaggle notebook, με μικρό expected payoff (HW1 features = +0.02 macro-F1 πάνω από TF-IDF text-only). Οι περισσότερες features δουλεύουν ως lexical cues που το transformer attention τις πιάνει contextually. Αναφέρεται ως «considered and rejected modification».
- [ ] **Class imbalance strategy**: HW1 χρησιμοποίησε `class_weight='balanced'` στο LogReg. Για transformers ξεκινάμε plain CrossEntropyLoss, προσθέτουμε weighted CE ως ablation.
- [ ] **Comparison with HW1**: HW1 best = 0.6017 val macro-F1 (qa_dual_tfidf + text_plus_relational_core, LogReg C=1.0, balanced). Transformer results να συγκριθούν direct.
