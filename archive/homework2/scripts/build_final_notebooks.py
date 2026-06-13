"""Build the 3 final Kaggle notebooks (one per required model).

Structure of each notebook:
  - Title + abstract (Greek)
  - Install dependencies
  - Library (inlined src/, pulled from kaggle_experiment.ipynb so it stays in sync)
  - One markdown + commented-out config cell per experiment tried
  - Final active config
  - Run cell
  - Results / learning curve / confusion matrix
  - Submission CSV (saves both submission_<model>.csv AND submission.csv for Kaggle)
  - Download helper

The assignment (via forum) says: keep all experiments in the notebook but comment out
the ones not part of the final solution. So every experiment we tried appears as a
markdown cell describing it + a commented-out config cell.
"""
import json
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_NB = PROJECT_ROOT / "notebooks" / "kaggle_experiment.ipynb"
OUT_DIR = PROJECT_ROOT / "notebooks"


def new_id() -> str:
    return str(uuid.uuid4())


def _as_lines(text: str) -> list[str]:
    # nbformat expects each list element to end with \n except the last one
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


# pull the library cell from the dev notebook so it stays in sync
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


# ─────────────────────────────────────────────────────────────────────────────
# Per-model content
# ─────────────────────────────────────────────────────────────────────────────

MODELS = {
    "distilbert": {
        "full_name": "distilbert-base-uncased",
        "display": "DistilBERT",
        "short": "distilbert-base-uncased",
        "final": dict(lr=3e-5, epochs=3, max_length=512, input_fmt="two_segment",
                      warmup_steps=100, batch_size=16, seed=42),
        "intro": """
# AI2 - Homework 2: Fine-tuning DistilBERT σε CLARITY (QEvasion)

Αυτό είναι το final notebook για το DistilBERT μοντέλο.

Το πρόβλημα είναι 3-class classification (Clear Reply / Ambivalent / Clear Non-Reply)
πάνω σε question-answer pairs από τo CLARITY dataset (HuggingFace: `ailsntua/QEvasion`).

Ο στόχος: fine-tuning του `distilbert-base-uncased` με explicit PyTorch training loop
(χωρίς HF Trainer, όπως ζητάει η εκφώνηση).

**HW1 baseline (LogReg + TF-IDF): val macro-F1 = 0.6017**.

Όλα τα πειράματα που δοκίμασα εμφανίζονται παρακάτω ως markdown + commented-out
config, και μόνο το final config είναι ενεργό.
        """,
        "experiments": [
            {
                "title": "Πείραμα 1 - Baseline (lr=2e-5, ep=3)",
                "md": """
Πρώτη δοκιμή: τα "safe" defaults για BERT-family fine-tuning - lr=2e-5, ep=3, max_length=256.
Αυτή η διαμόρφωση έδωσε val_f1_macro ≈ 0.6059, δηλαδή μόλις πάνω από το HW1 baseline.
                """,
                "code": """
# sweep_configs = [
#     ExperimentConfig(model_name="distilbert-base-uncased", input_fmt="two_segment",
#                      max_length=256, lr=2e-5, batch_size=16, epochs=3,
#                      warmup_steps=100, mode="dev", seed=42),
# ]
                """,
            },
            {
                "title": "Πείραμα 2 - LR sweep (lr=3e-5)",
                "md": """
Το training loss έπεφτε ακόμα στο ep3, οπότε δοκίμασα ελαφρώς μεγαλύτερο lr=3e-5.
Αποτέλεσμα: val_f1_macro = **0.6193** (καλύτερο). Αυτό έγινε το best DistilBERT config.
                """,
                "code": """
# sweep_configs = [
#     ExperimentConfig(model_name="distilbert-base-uncased", input_fmt="two_segment",
#                      max_length=256, lr=3e-5, batch_size=16, epochs=3,
#                      warmup_steps=100, mode="dev", seed=42),
# ]
                """,
            },
            {
                "title": "Πείραμα 3 - Epochs sweep (ep=5)",
                "md": """
Δοκίμασα να επεκτείνω σε ep=5 για να δω αν συνεχίζει να βελτιώνεται.
Αποτέλεσμα: best@ep4 με val_f1_macro = 0.6138, αλλά overfits μετά (train_loss 0.546->0.462
ενώ val_loss 0.771->0.782). Άρα ep=3 είναι καλύτερο.
                """,
                "code": """
# sweep_configs = [
#     ExperimentConfig(model_name="distilbert-base-uncased", input_fmt="two_segment",
#                      max_length=256, lr=2e-5, batch_size=16, epochs=5,
#                      warmup_steps=100, mode="dev", seed=42),
# ]
                """,
            },
            {
                "title": "Πείραμα 4 - Input format ablation (concat_sep)",
                "md": """
Δοκίμασα `concat_sep` (question + [SEP] + answer ως single string) αντί του
native `two_segment` (question, answer ως δύο ξεχωριστά tokenizer inputs).

**Interesting finding**: τα δύο formats δίνουν **ταυτόσημα αποτελέσματα** στο DistilBERT,
γιατί το DistilBERT δεν χρησιμοποιεί token_type_ids (segment embeddings). Οπότε ο
tokenizer παράγει το ίδιο ακριβώς token sequence και στις δύο περιπτώσεις.
Για BERT/DeBERTa που έχουν segment embeddings, το distinction θα είχε σημασία.
                """,
                "code": """
# sweep_configs = [
#     ExperimentConfig(model_name="distilbert-base-uncased", input_fmt="concat_sep",
#                      max_length=256, lr=3e-5, batch_size=16, epochs=3,
#                      warmup_steps=100, mode="dev", seed=42),
# ]
                """,
            },
            {
                "title": "Πείραμα 5 - Weighted CrossEntropyLoss",
                "md": """
Η Clear Reply ήταν consistently η πιο αδύναμη κλάση (~0.47-0.62). Δοκίμασα
balanced class weights (CR=1.09, AMB=0.56, CNR=3.24).

Αποτέλεσμα: CR βελτιώθηκε λίγο (+0.02), CNR επίσης (+0.03), αλλά Ambivalent
έπεσε αρκετά (-0.06), οπότε το overall macro-F1 μειώθηκε (0.5955 vs 0.6016).
Δεν αξίζει - το κρατάω αναφερόμενο αλλά δεν το χρησιμοποιώ στο final.
                """,
                "code": """
# sweep_configs = [
#     ExperimentConfig(model_name="distilbert-base-uncased", input_fmt="two_segment",
#                      max_length=256, lr=3e-5, batch_size=16, epochs=3,
#                      warmup_steps=100, mode="dev", seed=42,
#                      use_class_weights=True),
# ]
                """,
            },
            {
                "title": "Πείραμα 6 - Confirm runs ml=256 (3 seeds)",
                "md": """
Για να βγάλω μέτρηση με uncertainty, έτρεξα το ml=256 config με 3 seeds (42, 0, 1).

**Αποτέλεσμα: val_f1_macro = 0.6085 ± 0.007** (ml=256, implicit wd=0.01)

Η variance είναι μικρή - stable training.
                """,
                "code": """
# sweep_configs = [
#     ExperimentConfig(model_name="distilbert-base-uncased", input_fmt="two_segment",
#                      max_length=256, lr=3e-5, batch_size=16, epochs=3,
#                      warmup_steps=100, mode="confirm", seed=s)
#     for s in (42, 0, 1)
# ]
                """,
            },
            {
                "title": "Πείραμα 7 - max_length ablation (128 / 256 / 512)",
                "md": """
Πολλά political answers είναι μεγαλύτερα από 256 tokens. Δοκίμασα:
- ml=128: val_f1_macro = 0.6260
- ml=256: val_f1_macro = 0.6285
- **ml=512: val_f1_macro = 0.6405** (+0.012 vs ml=256)

Το ml=512 captures τα full answers που truncate-άρονταν στο 256. Έγινε το νέο best DistilBERT config.
                """,
                "code": """
# sweep_configs = [
#     ExperimentConfig(model_name="distilbert-base-uncased", input_fmt="two_segment",
#                      max_length=ml, lr=3e-5, batch_size=16, epochs=3,
#                      warmup_steps=100, mode="dev", seed=42)
#     for ml in (128, 256, 512)
# ]
                """,
            },
            {
                "title": "Πείραμα 8 - weight_decay ablation (και μια έκπληξη!)",
                "md": """
Δοκίμασα weight_decay ∈ {0.0, 0.01, 0.1} στο AdamW optimizer:
- wd=0.0: val_f1_macro = 0.6285
- wd=0.01: val_f1_macro = 0.6016
- wd=0.1: val_f1_macro = 0.6120

**Surprise finding**: το PyTorch `AdamW` έχει default weight_decay=0.01 (δεν είναι 0!).
Όλα τα προηγούμενα πειράματά μου έτρεχαν implicitly με wd=0.01 - η "baseline" μέτρηση
των 0.6016 ήταν ακριβώς αυτή η ρύθμιση. Κάνοντας explicit το wd=0.0, κερδίζω +0.027
στο DistilBERT σχεδόν χωρίς κόστος.

Άρα το final config χρησιμοποιεί wd=0.0 (το νέο library default μετά το fix).
                """,
                "code": """
# sweep_configs = [
#     ExperimentConfig(model_name="distilbert-base-uncased", input_fmt="two_segment",
#                      max_length=256, lr=3e-5, batch_size=16, epochs=3,
#                      warmup_steps=100, mode="dev", seed=42, weight_decay=wd)
#     for wd in (0.0, 0.01, 0.1)
# ]
                """,
            },
            {
                "title": "Πείραμα 9 - Final confirm runs (ml=512, wd=0.0, 3 seeds)",
                "md": """
Confirm runs για το νέο best config:

**Αποτέλεσμα: val_f1_macro = 0.6167 ± 0.021**

Best single seed: seed=42 με val_f1_macro = **0.6405** (το config που ships στο Kaggle).
Η variance ανέβηκε σε σχέση με το ml=256 wd=0.01 (από 0.007 σε 0.021), αλλά το mean
ανέβηκε επίσης (από 0.6085 σε 0.6167).
                """,
                "code": """
# sweep_configs = [
#     ExperimentConfig(model_name="distilbert-base-uncased", input_fmt="two_segment",
#                      max_length=512, lr=3e-5, batch_size=16, epochs=3,
#                      warmup_steps=100, mode="confirm", seed=s)
#     for s in (42, 0, 1)
# ]
                """,
            },
        ],
    },

    "bert": {
        "full_name": "bert-base-uncased",
        "display": "BERT",
        "short": "bert-base-uncased",
        "final": dict(lr=2e-5, epochs=5, max_length=256, input_fmt="two_segment",
                      warmup_steps=100, batch_size=16, seed=1),
        "intro": """
# AI2 - Homework 2: Fine-tuning BERT σε CLARITY (QEvasion)

Αυτό είναι το final notebook για το BERT μοντέλο (`bert-base-uncased`).

3-class classification (Clear Reply / Ambivalent / Clear Non-Reply) πάνω σε Q-A pairs
από το CLARITY dataset. Explicit PyTorch training loop, όχι HF Trainer.

**HW1 baseline (LogReg + TF-IDF): val macro-F1 = 0.6017**.

Όλα τα πειράματα που δοκίμασα εμφανίζονται παρακάτω με σχόλια στον κώδικα, και
μόνο το final config είναι ενεργό.
        """,
        "experiments": [
            {
                "title": "Πείραμα 1 - Baseline (lr=2e-5, ep=3)",
                "md": """
Με τα defaults για BERT fine-tuning (lr=2e-5, ep=3, ml=256) πήρα val_f1_macro = 0.6110.
Το training loss έπεφτε ακόμα στο ep3 - έμοιαζε να μπορεί να τρέξει περισσότερο.
                """,
                "code": """
# sweep_configs = [
#     ExperimentConfig(model_name="bert-base-uncased", input_fmt="two_segment",
#                      max_length=256, lr=2e-5, batch_size=16, epochs=3,
#                      warmup_steps=100, mode="dev", seed=42),
# ]
                """,
            },
            {
                "title": "Πείραμα 2 - Higher LR (lr=3e-5)",
                "md": """
Δοκίμασα lr=3e-5 αναλογικά με ό,τι βοήθησε το DistilBERT.
Για το BERT όμως αυτό overfits πιο γρήγορα: val_f1_macro = 0.6024, best@ep2,
και μετά πέφτει (0.602->0.594). Άρα το BERT προτιμά μικρότερο lr από το DistilBERT.
                """,
                "code": """
# sweep_configs = [
#     ExperimentConfig(model_name="bert-base-uncased", input_fmt="two_segment",
#                      max_length=256, lr=3e-5, batch_size=16, epochs=3,
#                      warmup_steps=100, mode="dev", seed=42),
# ]
                """,
            },
            {
                "title": "Πείραμα 3 - Περισσότερα epochs (ep=5) - best config",
                "md": """
Με ep=5 στο BERT παίρνω val_f1_macro = 0.6186, με best checkpoint στο ep4
(noisy trajectory: ep3=0.590, ep4=0.619, ep5=0.579). Το val_loss αρχίζει να ανεβαίνει
από το ep3, αλλά το best-checkpoint saving το πιάνει.

Αυτό έγινε το **best BERT config**.
                """,
                "code": """
# sweep_configs = [
#     ExperimentConfig(model_name="bert-base-uncased", input_fmt="two_segment",
#                      max_length=256, lr=2e-5, batch_size=16, epochs=5,
#                      warmup_steps=100, mode="dev", seed=42),
# ]
                """,
            },
            {
                "title": "Πείραμα 4 - Confirm runs με wd=0.01 (3 seeds)",
                "md": """
Confirm runs με 3 seeds (42, 0, 1) στο best config - με το implicit PyTorch default
weight_decay=0.01.

**Αποτέλεσμα: val_f1_macro = 0.6181 ± 0.021**

Η variance είναι αρκετά μεγαλύτερη από το DistilBERT - το BERT είναι πιο "θορυβώδες"
στο training, best epoch διαφέρει μεταξύ seeds (3 ή 4).
                """,
                "code": """
# sweep_configs = [
#     ExperimentConfig(model_name="bert-base-uncased", input_fmt="two_segment",
#                      max_length=256, lr=2e-5, batch_size=16, epochs=5,
#                      warmup_steps=100, mode="confirm", seed=s)
#     for s in (42, 0, 1)
# ]
                """,
            },
            {
                "title": "Πείραμα 5 - weight_decay discovery + re-confirm με wd=0.0",
                "md": """
Στο DistilBERT ablation ανακάλυψα ότι το PyTorch `AdamW` έχει default weight_decay=0.01
(όλα τα πρηγούμενα runs είχαν αυτό το implicit regularization). Έκανα re-confirm και
στο BERT με explicit wd=0.0:

**Αποτέλεσμα: val_f1_macro = 0.6072 ± 0.030**

**Interesting**: στο BERT το wd=0.0 είναι **ελαφρώς χειρότερο** στο mean (0.6072 vs 0.6181).
Αντίθετα με το DistilBERT/DeBERTa που βελτιώθηκαν. Πιθανή εξήγηση: το BERT (110M params)
είναι αρκετά μεγαλύτερο από το DistilBERT (66M) και ωφελείται περισσότερο από implicit L2.

Όμως το **best single seed** βελτιώθηκε οριακά: seed=1 έδωσε 0.6417 (νέο best BERT),
vs το παλιό seed=0 wd=0.01 που έδινε 0.6392. Άρα το single-run που ship-άρω
είναι ελαφρώς καλύτερο με wd=0.0 + seed=1.
                """,
                "code": """
# sweep_configs = [
#     ExperimentConfig(model_name="bert-base-uncased", input_fmt="two_segment",
#                      max_length=256, lr=2e-5, batch_size=16, epochs=5,
#                      warmup_steps=100, mode="confirm", seed=s)
#     for s in (42, 0, 1)   # weight_decay=0.0 via library default
# ]
                """,
            },
        ],
    },

    "deberta": {
        "full_name": "microsoft/deberta-v3-base",
        "display": "DeBERTa-v3",
        "short": "deberta-v3-base",
        "final": dict(lr=2e-5, epochs=3, max_length=256, input_fmt="two_segment",
                      warmup_steps=100, batch_size=16, seed=42),
        "intro": """
# AI2 - Homework 2: Fine-tuning DeBERTa-v3 σε CLARITY (QEvasion)

Αυτό είναι το final notebook για το DeBERTa-v3 μοντέλο (`microsoft/deberta-v3-base`).

3-class classification (Clear Reply / Ambivalent / Clear Non-Reply) πάνω σε Q-A pairs.
Explicit PyTorch training loop.

**HW1 baseline (LogReg + TF-IDF): val macro-F1 = 0.6017**.

Το DeBERTa ήταν σημαντικά πιο challenging στο setup - χρειάστηκαν αρκετές debugging
iterations μέχρι να δουλέψει σωστά. Όλα τα βήματα τεκμηριώνονται παρακάτω.

** ΣΗΜΑΝΤΙΚΟ**: το Cell 1 κάνει pin `transformers==4.44.0`. Η version 5.0.0 του
transformers έχει broken DeBERTa-v3 fine-tuning, το μοντέλο δεν μαθαίνει. Μετά το Cell 1
**πρέπει να γίνει restart του kernel** πριν τρέξετε τα υπόλοιπα cells.
        """,
        "experiments": [
            {
                "title": "Πείραμα 1 - Baseline (NaN loss)",
                "md": """
Πρώτη προσπάθεια με τα ίδια defaults που χρησιμοποίησα για BERT/DistilBERT:
lr=2e-5, ep=3, eps=1e-8 στον AdamW.

Αποτέλεσμα: `train_loss=nan` από το ep1. Το DeBERTa-v3 με disentangled attention
παράγει αριθμητικά unstable gradients στο default epsilon.
                """,
                "code": """
# (Παλιό config με eps=1e-8 - έβγαζε NaN)
# sweep_configs = [
#     ExperimentConfig(model_name="microsoft/deberta-v3-base", input_fmt="two_segment",
#                      max_length=256, lr=2e-5, batch_size=16, epochs=3,
#                      warmup_steps=100, mode="dev", seed=42),
# ]
                """,
            },
            {
                "title": "Πείραμα 2 - Fix: eps=1e-6 στον AdamW",
                "md": """
Αλλάξαμε σε `eps=1e-6` στον AdamW. Το NaN λύθηκε, αλλά το μοντέλο δεν μάθαινε:
val_f1_macro στο 0.2475 σε όλα τα epochs, predicts all Ambivalent.

Διαγνωστικό: max gradient norm = 10.8 (clipped στο 1.0 - 10x reduction). Το uniform
lr δεν φτάνει στο head.
                """,
                "code": """
# (eps=1e-6 fix αλλά ακόμα stuck στο 0.2475 - predicts all Ambivalent)
                """,
            },
            {
                "title": "Πείραμα 3 - Fix: differential learning rate για το head",
                "md": """
Δίνω στο `pooler` + `classifier` (τα randomly-init μέρη) 10x μεγαλύτερο lr από το
pretrained backbone. Αυτό είναι το standard trick για fine-tuning όταν grad clipping
συμπιέζει σημαντικά τα gradients στο head.

Παρόλα αυτά, δεν ήταν αρκετό - το πρόβλημα ήταν αλλού.
                """,
                "code": """
# (Differential lr υλοποιείται μέσα στο run_experiment για όλα τα DeBERTa runs)
# Το configuration ήταν το ίδιο με πριν, η αλλαγή ήταν στον optimizer.
                """,
            },
            {
                "title": "Πείραμα 4 - Root cause: transformers 5.0.0 bug",
                "md": """
Το Kaggle environment είχε transformers 5.0.0, που έχει broken DeBERTa-v3 fine-tuning
σε sequences κοντά στο max_length. Το diagnostic cell έτρεχε forward pass σε 10-token
toy input και δούλευε - στο training όμως με 256-token real data σπάει.

**Fix: pin transformers==4.44.0 + kernel restart**. Αμέσως το μοντέλο μαθαίνει!

Πρώτο successful run με transformers 4.44.0:
- ep1: f1=0.2475 (ακόμα χαμηλά)
- ep2: f1=0.6193 (pops up)
- ep3: **f1=0.6899**

Το είναι το **best overall result** πάνω σε όλα τα μοντέλα.
                """,
                "code": """
# Η λύση είναι το pin στο Cell 1 + kernel restart. Το config παραμένει ίδιο:
# sweep_configs = [
#     ExperimentConfig(model_name="microsoft/deberta-v3-base", input_fmt="two_segment",
#                      max_length=256, lr=2e-5, batch_size=16, epochs=3,
#                      warmup_steps=100, mode="dev", seed=42),
# ]
                """,
            },
            {
                "title": "Πείραμα 5 - Epoch sweep (ep=5)",
                "md": """
Δοκίμασα ep=5 για να δω αν συνεχίζει η βελτίωση πέρα από ep=3.
Αποτέλεσμα: best@ep5 με val_f1_macro=0.6844, αλλά χαμηλότερο από ep3=0.6899 (από ep=3 run).
Το ep=4 είναι overfit (val_f1 πέφτει στο 0.664, val_loss ανεβαίνει 0.663->0.748).

**Conclusion: ep=3 είναι το sweet spot για το DeBERTa.**
                """,
                "code": """
# sweep_configs = [
#     ExperimentConfig(model_name="microsoft/deberta-v3-base", input_fmt="two_segment",
#                      max_length=256, lr=2e-5, batch_size=16, epochs=5,
#                      warmup_steps=100, mode="dev", seed=42),
# ]
                """,
            },
            {
                "title": "Πείραμα 6 - Confirm runs με wd=0.01 (3 seeds)",
                "md": """
Confirm runs με 3 seeds (42, 0, 1) - με το implicit PyTorch default weight_decay=0.01.

**Αποτέλεσμα: val_f1_macro = 0.6585 ± 0.030**

Αξιοσημείωτη variance: seed=42 πήρε 0.6864, seed=1 πήρε 0.6627, seed=0 μόνο 0.6265.
Το DeBERTa είναι αρκετά sensitive στο random initialization.
                """,
                "code": """
# sweep_configs = [
#     ExperimentConfig(model_name="microsoft/deberta-v3-base", input_fmt="two_segment",
#                      max_length=256, lr=2e-5, batch_size=16, epochs=3,
#                      warmup_steps=100, mode="confirm", seed=s)
#     for s in (42, 0, 1)
# ]
                """,
            },
            {
                "title": "Πείραμα 7 - Re-confirm με wd=0.0 (final)",
                "md": """
Μετά την ανακάλυψη του implicit wd=0.01 default στο PyTorch AdamW, έκανα re-confirm
με explicit wd=0.0 (το νέο library default).

**Αποτέλεσμα: val_f1_macro = 0.6674 ± 0.032** (+0.009 mean vs wd=0.01)

Best single seed: seed=42 με val_f1_macro = **0.6962** (best over όλα τα μοντέλα).
Το DeBERTa ωφελείται λίγο από wd=0.0, όπως το DistilBERT αλλά αντίθετα από το BERT.

Αυτό είναι το final config.
                """,
                "code": """
# sweep_configs = [
#     ExperimentConfig(model_name="microsoft/deberta-v3-base", input_fmt="two_segment",
#                      max_length=256, lr=2e-5, batch_size=16, epochs=3,
#                      warmup_steps=100, mode="confirm", seed=s)
#     for s in (42, 0, 1)   # weight_decay=0.0 via library default
# ]
                """,
            },
        ],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Cell builders
# ─────────────────────────────────────────────────────────────────────────────

def final_config_cell(m: dict) -> str:
    f = m["final"]
    return f"""# ── TELIKO CONFIG (το μοναδικό ενεργό) ──
sweep_configs = [
    ExperimentConfig(
        model_name   = "{m['full_name']}",
        input_fmt    = "{f['input_fmt']}",
        max_length   = {f['max_length']},
        lr           = {f['lr']},
        batch_size   = {f['batch_size']},
        epochs       = {f['epochs']},
        warmup_steps = {f['warmup_steps']},
        mode         = "dev",   # κρατάω val split για να φαίνονται metrics
        seed         = {f['seed']},
    ),
]

for cfg in sweep_configs:
    print(cfg.run_id)
"""


RUN_CELL = """# Run training
import gc
from pathlib import Path

sweep_results = []
for cfg in sweep_configs:
    print(f"\\n{'='*60}\\nRUNNING: {cfg.run_id}\\n{'='*60}")
    history, val_logits, val_labels_arr, test_df, tokenizer, model, device = run_experiment(cfg)
    best_epoch = max(history, key=lambda x: x["f1_macro"])
    run_dir = Path(cfg.models_dir) / cfg.run_id
    sweep_results.append({
        "run_id":       cfg.run_id,
        "run_dir":      str(run_dir),
        "best_epoch":   best_epoch["epoch"],
        "val_f1_macro": best_epoch["f1_macro"],
        "val_acc":      best_epoch["accuracy"],
        "f1_per_class": best_epoch["f1_per_class"],
        "history":      history,
        "cfg":          cfg,
    })
    # free GPU before next run (if any)
    model.cpu()
    del model, tokenizer, val_logits, val_labels_arr
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\\nDone.")
"""


RESULTS_CELL = """# Results table
import pandas as pd

summary = pd.DataFrame([{
    "run_id":       r["run_id"],
    "best_epoch":   r["best_epoch"],
    "val_f1_macro": round(r["val_f1_macro"], 4),
    "val_acc":      round(r["val_acc"], 4),
    "f1_CR":        round(r["f1_per_class"][0], 4),
    "f1_AMB":       round(r["f1_per_class"][1], 4),
    "f1_CNR":       round(r["f1_per_class"][2], 4),
} for r in sweep_results])

print("=== Final results (HW1 baseline = 0.6017) ===")
display(summary)
best_run = max(sweep_results, key=lambda r: r["val_f1_macro"])
print(f"\\nBest: {best_run['run_id']}  f1={best_run['val_f1_macro']:.4f}")
"""


LEARNING_CURVE_CELL = """# Learning curve
import matplotlib.pyplot as plt

r = sweep_results[0]
epochs = [e["epoch"] for e in r["history"]]
train_losses = [e["train_loss"] for e in r["history"]]
val_losses   = [e["val_loss"]   for e in r["history"]]
val_f1s      = [e["f1_macro"]   for e in r["history"]]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(epochs, train_losses, "o-", label="train_loss")
ax1.plot(epochs, val_losses,   "o-", label="val_loss")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend(); ax1.grid(True, alpha=0.3)
ax1.set_title("Loss per epoch")

ax2.plot(epochs, val_f1s, "o-", color="green")
ax2.axhline(y=0.6017, color="gray", linestyle="--", label="HW1 baseline")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("val F1-macro"); ax2.legend(); ax2.grid(True, alpha=0.3)
ax2.set_title("Validation F1-macro per epoch")

plt.tight_layout()
plt.savefig("/kaggle/working/learning_curve.png", dpi=150)
plt.show()
"""


CONFUSION_CELL = """# Confusion matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from pathlib import Path

r = sweep_results[0]
val_logits = np.load(Path(r["run_dir"]) / "val_preds.npy")
val_labels = np.load(Path(r["run_dir"]) / "val_labels.npy")
preds = val_logits.argmax(axis=1)
cm = confusion_matrix(val_labels, preds, labels=[0, 1, 2])
label_names = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=label_names, yticklabels=label_names,
            cmap="Blues", ax=ax)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title(f"{r['run_id']}\\nval F1-macro = {r['val_f1_macro']:.4f}")
plt.tight_layout()
plt.savefig("/kaggle/working/confusion_matrix.png", dpi=150)
plt.show()

print(f"cm: {cm.tolist()}")
"""


def submission_cell(m: dict) -> str:
    return f"""# Generate submission CSV (δύο ονόματα: model-specific + generic για το Kaggle uploader)
import pandas as pd
from pathlib import Path

r = sweep_results[0]
reload_model, reload_tok = load_model_and_tokenizer(r["cfg"].model_name, num_labels=NUM_LABELS)
reload_model.load_state_dict(torch.load(
    Path(r["run_dir"]) / "best_model.pt",
    map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reload_model.to(device).eval()

_, test_df = load_clarity()
test_df_enc = test_df.copy()
test_df_enc["label_id"] = 0  # dummy

test_ds = tokenize_pairs(test_df_enc, reload_tok, r["cfg"].max_length, r["cfg"].input_fmt)
test_loader = DataLoader(test_ds, batch_size=r["cfg"].batch_size, sampler=SequentialSampler(test_ds))

all_preds = []
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, _ = [b.to(device) for b in batch]
        out = reload_model(input_ids=input_ids, attention_mask=attention_mask)
        all_preds.extend(out.logits.argmax(dim=1).cpu().numpy())

submission = pd.DataFrame({{
    "Id": test_df["index"] if "index" in test_df.columns else range(len(test_df)),
    "Predicted": [ID2LABEL[p] for p in all_preds],
}})

# Save BOTH names - Kaggle uploader wants "submission.csv", assignment wants model-specific
submission.to_csv("/kaggle/working/submission_{m['short']}.csv", index=False)
submission.to_csv("/kaggle/working/submission.csv", index=False)
print(f"Submissions saved: submission_{m['short']}.csv and submission.csv")
print(f"Val F1-macro = {{r['val_f1_macro']:.4f}}")
display(submission.head())
print(submission["Predicted"].value_counts())
"""


def download_cell(m: dict) -> str:
    return f"""# Download run directory (metrics.json, val_preds.npy, val_labels.npy, best_model.pt)
import shutil
from pathlib import Path
from IPython.display import FileLink, display

src = Path("/kaggle/working/models") / sweep_results[0]["cfg"].run_id
dst_base = "/kaggle/working/run_{m['short']}"
shutil.make_archive(dst_base, "zip", str(src.parent), src.name)
print(f"Zipped: {{dst_base}}.zip")
display(FileLink(f"run_{m['short']}.zip"))
"""


# ─────────────────────────────────────────────────────────────────────────────
# Build notebooks
# ─────────────────────────────────────────────────────────────────────────────

def build_notebook(key: str, m: dict) -> dict:
    cells = []
    cells.append(md(m["intro"]))

    cells.append(md("## Βήμα 1 - Εγκατάσταση dependencies\n\nPin `transformers==4.44.0` για reproducibility."))
    cells.append(code(INSTALL_SRC.strip()))

    cells.append(md(
        "## Βήμα 2 - Library (inlined src/)\n\n"
        "Ολόκληρο το training pipeline μέσα σε ένα cell για self-contained notebook "
        "(config, data loading/cleaning, tokenization, model, training, evaluation, experiment runner)."
    ))
    cells.append(code(LIBRARY_SRC.strip()))

    cells.append(md(
        "## Βήμα 3 - Πειράματα που δοκίμασα\n\n"
        "Εδώ παρουσιάζω όλες τις παραμετροποιήσεις που δοκίμασα, σχολιασμένες "
        "(comment out) ώστε να μην τρέξουν αλλά να είναι τεκμηριωμένες όπως ζητάει η εκφώνηση. "
        "Μόνο το final config στο τέλος είναι ενεργό."
    ))

    for exp in m["experiments"]:
        cells.append(md(f"### {exp['title']}\n\n{exp['md'].strip()}"))
        cells.append(code(exp["code"].strip()))

    cells.append(md(
        f"## Βήμα 4 - Τελικό configuration\n\n"
        f"Best confirmed config για {m['display']}: "
        f"lr={m['final']['lr']}, epochs={m['final']['epochs']}, max_length={m['final']['max_length']}, "
        f"input_fmt={m['final']['input_fmt']}, seed={m['final']['seed']}."
    ))
    cells.append(code(final_config_cell(m)))

    cells.append(md("## Βήμα 5 - Training"))
    cells.append(code(RUN_CELL))

    cells.append(md("## Βήμα 6 - Αποτελέσματα"))
    cells.append(code(RESULTS_CELL))

    cells.append(md("## Βήμα 7 - Learning curve"))
    cells.append(code(LEARNING_CURVE_CELL))

    cells.append(md("## Βήμα 8 - Confusion matrix"))
    cells.append(code(CONFUSION_CELL))

    cells.append(md(
        "## Βήμα 9 - Submission για το Kaggle\n\n"
        "Γενικό `submission.csv` (τι θέλει το Kaggle uploader) + `submission_<model>.csv` "
        "(τι ζητάει η εκφώνηση). Τα δύο αρχεία είναι identical."
    ))
    cells.append(code(submission_cell(m)))

    cells.append(md("## Βήμα 10 - Download artifacts"))
    cells.append(code(download_cell(m)))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return nb


for key, m in MODELS.items():
    nb = build_notebook(key, m)
    out_path = OUT_DIR / f"kaggle_final_{key}.ipynb"
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Built: {out_path}  ({len(nb['cells'])} cells)")
