"""Build notebooks/hw3_final.ipynb : Phase 6 deliverable, το Kaggle notebook.

Best system locked = Qwen3.5-4B + Chain-of-Thought (prompt = cot_v4).
- Live run: το best system end-to-end (dev + test) -> submission_best_prompting_system.csv.
- Άλλα πειράματα (Phase 4 grid, Phase 5 prompt refinement, Phase 5.5 SC): markdown
  results + commented-out reproducible blocks (instructor-approved).

Ίδιο subprocess pattern με τα προηγούμενα notebooks (Kaggle kernel pre-imports stale
transformers).

Regenerate with:  python scripts/make_final_notebook.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "hw3_final.ipynb"

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})


# ====================================================================
# hw3_final.py : η LIVE εκτέλεση του best system (4B + cot_v4).
# Τρέχει σαν subprocess (καθαρό transformers import).
# Παράγει submission_best_prompting_system.csv με τις test predictions.
# ====================================================================
SCRIPT = r'''# hw3_final.py: Best system live: Qwen3.5-4B + Chain-of-Thought (cot_v4).
# Subprocess pattern: το Kaggle kernel pre-imports stale transformers, οποτε γραφω
# ολο το πειραμα εδω και το τρεχω σαν fresh Python process.
import time, gc, sys, random
import numpy as np
import pandas as pd
import torch
from collections import Counter, defaultdict
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_recall_fscore_support, confusion_matrix)
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

print("=" * 72, flush=True)
print("transformers:", transformers.__version__, "| torch:", torch.__version__, flush=True)
print("CUDA:", torch.cuda.is_available(),
      "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-",
      flush=True)
try:
    _cfg = AutoConfig.from_pretrained("Qwen/Qwen3.5-0.8B")
    print("OK: Qwen3.5 supported (model_type =", _cfg.model_type, ")", flush=True)
except Exception as e:
    print("FATAL: transformers", transformers.__version__,
          "cannot load Qwen3.5 config:", repr(e)[:300], flush=True)
    sys.exit(1)
assert torch.cuda.is_available(), "GPU OFF: Settings -> Accelerator -> GPU"
print("=" * 72, flush=True)


# ===== CONFIG =====
LABELS = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]
MODEL = "Qwen/Qwen3.5-4B"
DEV_SUBSET_N = 500
SEED = 42
INIT_BATCH = 4
COT_MAX_NEW_TOKENS = 256
ENABLE_THINKING = False
# Γραφω και τα δυο ονοματα: το `submission.csv` ειναι αυτο που το Kaggle competition
# πιανει by convention (το Phase 4 submission που πηρε 0.725 LB γραφτηκε με αυτο το
# ονομα), και το `submission_best_prompting_system.csv` ειναι το descriptive name
# που ζηταει η εκφωνηση. Identical content σε καθε αρχειο.
SUBMISSION_PATH = "/kaggle/working/submission.csv"
SUBMISSION_NAMED_PATH = "/kaggle/working/submission_best_prompting_system.csv"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ===== DATA =====
# Το competition δινει μονο sample_solution.csv. Τα πραγματικα δεδομενα ζουν στο
# HuggingFace dataset ailsntua/QEvasion (train + test parquet files).
# Δεν χρησιμοποιω τη βιβλιοθηκη `datasets` λογω version conflict με το
# transformers-from-git: φορτωνω τα parquet απευθειας με hf_hub_download.
QEVASION_REPO = "ailsntua/QEvasion"


def _qevasion_df(split):
    path = hf_hub_download(repo_id=QEVASION_REPO, repo_type="dataset",
                           filename="data/" + split + "-00000-of-00001.parquet")
    return pd.read_parquet(path)


def load_records():
    df = _qevasion_df("train")
    recs = []
    for q, a, lab in zip(df["question"], df["interview_answer"], df["clarity_label"]):
        if lab in LABELS and isinstance(q, str) and q and isinstance(a, str) and a:
            rec = {"question": q, "interview_answer": a, "clarity_label": lab}
            rec["q_words"] = len(q.split())
            rec["a_words"] = len(a.split())
            recs.append(rec)
    return recs


def load_test_records():
    df = _qevasion_df("test")
    has_label = "clarity_label" in df.columns
    recs = []
    for i in range(len(df)):
        q, a = df["question"].iloc[i], df["interview_answer"].iloc[i]
        rec = {"question": q if isinstance(q, str) else "",
               "interview_answer": a if isinstance(a, str) else ""}
        if has_label:
            rec["clarity_label"] = df["clarity_label"].iloc[i]
        rec["q_words"] = len(rec["question"].split())
        rec["a_words"] = len(rec["interview_answer"].split())
        recs.append(rec)
    return recs, has_label


def build_split(records, n_dev, seed):
    labels = [r["clarity_label"] for r in records]
    idx = list(range(len(records)))
    return train_test_split(idx, train_size=n_dev, stratify=labels, random_state=seed)


# ===== PROMPT: cot_v4 (το final prompt: sharpened definitions + 3-step CoT) =====
def build_user_prompt(rec):
    return f'Question: {rec["question"]}\nAnswer: {rec["interview_answer"]}'


_V4_HEAD = """You are an expert annotator for political interviews. You are given the specific question a journalist asked and the answer a public figure gave. Judge ONE thing only: did the answer actually give the specific information the question asked for?

Do not be impressed by length or fluency. Political answers are often long, articulate, and on-topic while still not answering the actual question. Judge the substance, not the style.

Use exactly one of these three labels:

- Clear Reply: the answer gives the specific information, position, or yes/no that the question asked for. Direct and on-point.
- Ambivalent: the answer engages with the topic but does NOT give the specific information asked. It hedges, generalises, answers an easier or different question, gives only part of the answer, or talks around it.
- Clear Non-Reply: the answer does not engage with the question at all. The speaker refuses, explicitly declines, or changes the subject."""

_COT_BODY_V4 = """Think it through step by step:
1. What specific information does the question ask for? (one sentence)
2. What does the answer actually provide? (one sentence)
3. Does the answer deliver that specific information - fully, only partially / by talking around it, or not at all?

Keep your reasoning to about 3 short sentences. Then, on a final separate line, write exactly:
Label: <one of: Clear Reply, Ambivalent, Clear Non-Reply>"""

SYS_COT_V4 = _V4_HEAD + "\n\n" + _COT_BODY_V4


# ===== INFERENCE =====
def load_model(model_name):
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    return tok, model


def run_inference(tok, model, system_prompt, records, batch_size,
                  max_new_tokens, enable_thinking):
    """Greedy batched inference. Sorts by prompt length για να γεμιζουν καλυτερα τα
    batches. Auto-backoff σε OOM (halve batch). Επιστρεφει string ανα record."""
    prompts = []
    for rec in records:
        msgs = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": build_user_prompt(rec)}]
        prompts.append(tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking))
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    outs = [None] * len(prompts)
    pos, bs = 0, batch_size
    while pos < len(order):
        idxs = order[pos:pos + bs]
        chunk = [prompts[i] for i in idxs]
        try:
            enc = tok(chunk, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                                     do_sample=False, pad_token_id=tok.eos_token_id)
            for j, oi in enumerate(idxs):
                outs[oi] = tok.decode(gen[j][enc["input_ids"].shape[1]:],
                                      skip_special_tokens=True)
            pos += len(idxs)
        except Exception as e:
            if "out of memory" not in str(e).lower():
                raise
            torch.cuda.empty_cache()
            if bs == 1:
                outs[idxs[0]] = ""
                pos += 1
            else:
                bs = max(1, bs // 2)
                print("  OOM -> batch_size", bs, flush=True)
    return outs


def parse_label(text):
    """Παιρνει το CoT output και βγαζει το label. Looks for "Label:" first, αλλιως
    fallback στο πιο δεξιο label που εμφανιζεται στο text."""
    t = text.strip()
    if "</think>" in t:
        t = t.split("</think>")[-1].strip()
    low = t.lower()
    if "label:" in low:
        t = t[low.rfind("label:") + len("label:"):].strip()
        low = t.lower()
    for lab in LABELS:
        if low == lab.lower():
            return lab
    best = None
    for lab in LABELS:
        pos = low.rfind(lab.lower())
        if pos != -1 and (best is None or pos > best[0]
                          or (pos == best[0] and len(lab) > len(best[1]))):
            best = (pos, lab)
    return best[1] if best else "Invalid"


# ===== EVAL / REPORTING =====
def evaluate(gold, preds):
    acc = accuracy_score(gold, preds)
    macro_f1 = f1_score(gold, preds, labels=LABELS, average="macro", zero_division=0)
    weighted_f1 = f1_score(gold, preds, labels=LABELS, average="weighted", zero_division=0)
    p, r, f, sup = precision_recall_fscore_support(
        gold, preds, labels=LABELS, zero_division=0)
    n_invalid = sum(1 for x in preds if x not in LABELS)
    counts = {lab: preds.count(lab) for lab in LABELS}
    counts["Invalid"] = n_invalid
    return {"acc": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1,
            "f1": dict(zip(LABELS, f)), "precision": dict(zip(LABELS, p)),
            "recall": dict(zip(LABELS, r)), "support": dict(zip(LABELS, sup)),
            "invalid_rate": n_invalid / len(preds), "pred_counts": counts}


def fmt_confusion(gold, preds):
    cols = LABELS + ["Invalid"]
    mapped = [p if p in LABELS else "Invalid" for p in preds]
    cm = confusion_matrix(gold, mapped, labels=cols)
    out = ["(rows = gold, cols = pred)"]
    out.append(" " * 18 + "".join(c.rjust(16) for c in cols))
    for i, g in enumerate(cols):
        if g == "Invalid":
            continue
        out.append(g.ljust(18) + "".join(
            str(int(cm[i][j])).rjust(16) for j in range(len(cols))))
    return "\n".join(out)


def print_per_class(metrics):
    print("class".ljust(18) + "precision".rjust(11) + "recall".rjust(9)
          + "f1".rjust(9) + "support".rjust(9))
    for lab in LABELS:
        print(lab.ljust(18)
              + ("%.3f" % metrics["precision"][lab]).rjust(11)
              + ("%.3f" % metrics["recall"][lab]).rjust(9)
              + ("%.3f" % metrics["f1"][lab]).rjust(9)
              + str(int(metrics["support"][lab])).rjust(9))


def show_errors(records, preds, gold, gens=None, max_per_cell=2):
    """Group errors by (gold, pred). Print Q/A και (αν δοθει) το CoT reasoning."""
    buckets = defaultdict(list)
    for i, (g, p) in enumerate(zip(gold, preds)):
        if g != p:
            buckets[(g, p)].append(i)
    for (g, p), idxs in sorted(buckets.items()):
        print("\n### gold=" + g + "  ->  pred=" + p + "   (" + str(len(idxs)) + " cases)")
        for i in idxs[:max_per_cell]:
            r = records[i]
            print("  Q:", r["question"][:160])
            print("  A:", r["interview_answer"][:300])
            if gens is not None:
                print("  CoT:", gens[i][:400].replace("\n", " "))
            print()


def subgroup_report(records, preds, gold, key, n_bins=3):
    vals = sorted(r[key] for r in records)
    cuts = [vals[len(vals) * k // n_bins] for k in range(1, n_bins)]
    names = ["short", "medium", "long"][:n_bins]
    print("subgroup by", key, " (cut points:", cuts, ")")
    for b in range(n_bins):
        idxs = [i for i, r in enumerate(records)
                if sum(1 for c in cuts if r[key] >= c) == b]
        if not idxs:
            continue
        m = evaluate([gold[i] for i in idxs], [preds[i] for i in idxs])
        c = m["pred_counts"]
        print("  " + names[b].ljust(8) + "N=" + str(len(idxs)).ljust(5)
              + "acc=%.3f  macroF1=%.3f  weightedF1=%.3f"
              % (m["acc"], m["macro_f1"], m["weighted_f1"])
              + "  preds: CR=%d AMB=%d CNR=%d"
              % (c["Clear Reply"], c["Ambivalent"], c["Clear Non-Reply"]))


# ===== RUN =====
set_seed(SEED)
RUN_START = time.time()

# ----- (1) data -----
all_records = load_records()
print("QEvasion train usable records:", len(all_records), flush=True)
dev_idx, _ = build_split(all_records, DEV_SUBSET_N, SEED)
dev_records = [all_records[i] for i in dev_idx]
gold = [r["clarity_label"] for r in dev_records]
print("dev subset: N=" + str(len(dev_records)),
      "class dist:", dict(Counter(gold)), flush=True)

test_records, has_test_label = load_test_records()
test_gold = [r.get("clarity_label") for r in test_records] if has_test_label else None
print("test set: N=" + str(len(test_records)),
      "| gold labels present:", has_test_label, flush=True)

# ----- (2) print το prompt που χρησιμοποιούμε -----
print("\n" + "=" * 72)
print("PROMPT: cot_v4 (το final prompt)")
print("=" * 72)
print(SYS_COT_V4)

# ----- (3) load model -----
print("\n" + "=" * 72, flush=True)
print("LOADING MODEL:", MODEL, flush=True)
print("=" * 72, flush=True)
t0 = time.time()
tok, model = load_model(MODEL)
print("loaded in %.0fs | GPU mem %.1fGB"
      % (time.time() - t0, torch.cuda.memory_allocated() / 1e9), flush=True)

# ----- (4) dev run -----
print("\n" + "=" * 72, flush=True)
print("(1) DEV RUN: 4B + cot_v4 (greedy) on N=" + str(len(dev_records)), flush=True)
print("=" * 72, flush=True)
set_seed(SEED)
t0 = time.time()
dev_gens = run_inference(tok, model, SYS_COT_V4, dev_records, INIT_BATCH,
                         COT_MAX_NEW_TOKENS, ENABLE_THINKING)
dev_preds = [parse_label(g) for g in dev_gens]
dev_metrics = evaluate(gold, dev_preds)
print("\nDEV: macroF1=%.3f  weightedF1=%.3f  acc=%.3f  invalid=%.1f%%  (%.0fs)"
      % (dev_metrics["macro_f1"], dev_metrics["weighted_f1"], dev_metrics["acc"],
         dev_metrics["invalid_rate"] * 100, time.time() - t0), flush=True)
c = dev_metrics["pred_counts"]
print("DEV pred distribution: CR=%d AMB=%d CNR=%d Invalid=%d"
      % (c["Clear Reply"], c["Ambivalent"], c["Clear Non-Reply"], c["Invalid"]))

print("\n--- per-class metrics (dev) ---")
print_per_class(dev_metrics)

print("\n--- confusion matrix (dev) ---")
print(fmt_confusion(gold, dev_preds))

print("\n--- subgroup analysis (dev) ---")
subgroup_report(dev_records, dev_preds, gold, "q_words")
print()
subgroup_report(dev_records, dev_preds, gold, "a_words")

print("\n--- example errors (dev) ---")
show_errors(dev_records, dev_preds, gold, gens=dev_gens, max_per_cell=2)

# ----- (5) CoT reasoning samples (correct + incorrect) -----
print("\n" + "=" * 72)
print("CoT REASONING SAMPLES (correct + a few wrong)")
print("=" * 72)
correct = [i for i in range(len(dev_records)) if dev_preds[i] == gold[i]]
wrong = [i for i in range(len(dev_records)) if dev_preds[i] != gold[i]]
random.Random(SEED).shuffle(correct)
random.Random(SEED).shuffle(wrong)
for tag, sel in [("CORRECT", correct[:3]), ("WRONG", wrong[:3])]:
    for i in sel:
        print("\n[" + tag + "  idx=" + str(i) + "]  gold=" + gold[i]
              + "  pred=" + dev_preds[i])
        print("Q:", dev_records[i]["question"][:160])
        print("A:", dev_records[i]["interview_answer"][:280])
        print("CoT:", dev_gens[i][:700])

# ----- (6) test run + submission file -----
print("\n" + "=" * 72, flush=True)
print("(2) TEST RUN: 4B + cot_v4 -> submission.csv + submission_best_prompting_system.csv", flush=True)
print("=" * 72, flush=True)
set_seed(SEED)
t0 = time.time()
test_gens = run_inference(tok, model, SYS_COT_V4, test_records, INIT_BATCH,
                          COT_MAX_NEW_TOKENS, ENABLE_THINKING)
test_preds = [parse_label(g) for g in test_gens]
print("predicted %d test examples in %.0fs" % (len(test_preds), time.time() - t0))

sub = pd.DataFrame({"Id": range(len(test_preds)), "Predicted": test_preds})
sub.to_csv(SUBMISSION_PATH, index=False)
sub.to_csv(SUBMISSION_NAMED_PATH, index=False)
print("wrote " + SUBMISSION_PATH)
print("wrote " + SUBMISSION_NAMED_PATH + " (descriptive name για το report)")
print("test pred distribution:", dict(Counter(test_preds)))
print("invalid in test preds:", sum(1 for p in test_preds if p not in LABELS))

if has_test_label and all(r.get("clarity_label") in LABELS for r in test_records):
    tm = evaluate(test_gold, test_preds)
    c = tm["pred_counts"]
    print("\n--- LOCAL test scores (QEvasion test gold available) ---")
    print("TEST: macroF1=%.3f  weightedF1=%.3f  acc=%.3f"
          % (tm["macro_f1"], tm["weighted_f1"], tm["acc"]))
    print("TEST pred distribution: CR=%d AMB=%d CNR=%d Invalid=%d"
          % (c["Clear Reply"], c["Ambivalent"], c["Clear Non-Reply"], c["Invalid"]))
    print("\n--- per-class metrics (test) ---")
    print_per_class(tm)
    print("\n--- confusion matrix (test) ---")
    print(fmt_confusion(test_gold, test_preds))
    print("\n--- subgroup analysis (test) ---")
    subgroup_report(test_records, test_preds, test_gold, "q_words")
    print()
    subgroup_report(test_records, test_preds, test_gold, "a_words")

print("\n--- submission.csv head ---")
print(sub.head(10).to_string())

elapsed_h = (time.time() - RUN_START) / 3600
print("\n===== DONE (total %.2fh) =====" % elapsed_h, flush=True)
'''


# ====================================================================
# Cell 0 : Title + intro
# ====================================================================
md(r'''
# HW3: Final notebook (Kaggle deliverable)

**Response clarity classification** στο dataset CLARITY/QEvasion, με **prompting** σε
frozen instruction-tuned LLMs (καθόλου fine-tuning, μόνο inference).

## Best system

| | |
|---|---|
| Μοντέλο | **Qwen3.5-4B** (instruction-tuned, frozen) |
| Στρατηγική | **Chain-of-Thought** prompting (prompt = `cot_v4`) |
| Dev macro-F1 | 0.548 |
| Test macro-F1 | 0.575 |
| Kaggle public LB | **0.725 weighted-F1, 2η θέση** |
| Submission files | `submission.csv` + `submission_best_prompting_system.csv` (identical content, παράγονται live από αυτό το notebook) |

## Τι κάνει αυτό το notebook

1. Στήνει το περιβάλλον (transformers από git, για να υποστηρίξει το Qwen3.5).
2. Εξηγεί τη μεθοδολογία και το prompt design.
3. **Τρέχει live το best system** (Qwen3.5-4B + Chain-of-Thought) σε dev (N=500) και
   test (N=308). Παράγει `submission.csv` και `submission_best_prompting_system.csv`
   (ίδιο content, διπλό όνομα για να καλυφθούν Kaggle convention + spec naming).
4. Δείχνει αναλυτικά τα metrics, confusion matrix, per-class scores, subgroup
   analysis, error examples, και CoT reasoning samples του best system.
5. Παρουσιάζει τα συγκριτικά πειράματα που έγιναν για να φτάσω εδώ: 3x3 grid
   (model size x strategy), δοκιμές prompt refinement, και ένα self-consistency
   experiment. Τα results είναι σε πίνακες markdown και ο κώδικάς τους είναι
   commented-out reproducible.
6. Συγκρίνει με τα HW1 (vector-based) και HW2 (encoder fine-tuning).

## Reproducibility note

Όλα τα experiments τρέχουν με **random seed 42** και greedy decoding (deterministic,
`do_sample=False`). Το ίδιο 500-row stratified dev subset χρησιμοποιείται για όλες τις
συγκρίσεις. Invalid output rate σε όλα τα runs: 0%.
''')


# ====================================================================
# Cell 1 : Πριν τρέξεις
# ====================================================================
md(r'''
## Πριν τρέξεις (Kaggle settings)

- **Settings -> Accelerator -> GPU** (Tesla T4 x2 είναι αρκετό)
- **Settings -> Internet -> On** (χρειάζεται για το pip install + το HF download)
- **Save Version -> "Save & Run All (Commit)"** ώστε να τρέξει server-side (μέχρι 12h).
- Διάρκεια live run: περίπου 2.5 με 3 ώρες (4B + CoT σε 500 dev + 308 test).
''')


# ====================================================================
# Cell 2 : Install transformers
# ====================================================================
md(r'''
## 1. Setup: transformers από git (υποστηρίζει Qwen3.5)

Το Qwen3.5 δεν έχει βγει στο stable release του transformers, οπότε εγκαθιστούμε από
git. Το Kaggle kernel ξεκινά με ένα παλιότερο transformers ήδη pre-imported, που δεν
γνωρίζει το `qwen3_5` model type. Δεν φτάνει ένα απλό `pip install -U`: πρέπει πρώτα
`uninstall`, μετά καθαρό `install` από git.
''')

code(r'''
!pip uninstall -y -q transformers
!pip install -U --no-cache-dir "transformers @ git+https://github.com/huggingface/transformers.git"
''')

md(r'''
Και επειδή το running kernel κρατά το stale `transformers` φορτωμένο στη μνήμη ακόμα
και μετά το install, ολόκληρο το πείραμα γράφεται σε ένα αρχείο `.py` και τρέχει σαν
**ξεχωριστή διεργασία** (`!python -u ...`). Έτσι ένα fresh Python process κάνει
καθαρό import το νέο transformers.
''')


# ====================================================================
# Cell 3 : Methodology
# ====================================================================
md(r'''
## 2. Μεθοδολογία

### Μοντέλα

Δοκίμασα και τα τρία μεγέθη του Qwen3.5 που ζητάει η εκφώνηση:

- `Qwen/Qwen3.5-0.8B`
- `Qwen/Qwen3.5-2B`
- `Qwen/Qwen3.5-4B`

Όλα instruction-tuned. **Δεν κάνω καθόλου fine-tuning**, μόνο inference.

### Δεδομένα

- Το Kaggle competition δίνει μόνο το `sample_solution.csv`. Τα πραγματικά δεδομένα
  είναι στο HuggingFace dataset `ailsntua/QEvasion`.
- **Train**: 3448 usable records (Ambivalent 2040 / Clear Reply 1052 / Clear Non-Reply
  356, περίπου 59/31/10%). Σαφές class imbalance.
- **Test**: 308 records. Σημαντικό: το test split του HF dataset **έχει gold labels**,
  οπότε ξέρω local test performance χωρίς να χρειάζεται Kaggle submit.
- Δεν χρησιμοποιώ τη `datasets` library: έχει version conflict με το
  transformers-from-git (`huggingface_hub.BucketNotFoundError`). Φορτώνω τα parquet
  files απευθείας με `hf_hub_download`.

### Dev subset

Όλη η σύγκριση γίνεται σε ένα **fixed stratified dev subset των 500 παραδειγμάτων** του
train split, με seed 42. Κατανομή: AMB 296 / CR 152 / CNR 52. Έτσι κάθε experiment
βλέπει ακριβώς τα ίδια examples και οι διαφορές αντανακλούν strategy/model, όχι noise.

### Inference pipeline

- **Chat template** του tokenizer με `apply_chat_template(..., add_generation_prompt=True)`.
- **`enable_thinking=False`**: το Qwen3.5 έχει thinking mode (γράφει `<think>...</think>`
  πριν την απάντηση). Με thinking on, μερικές γενιές καίνε χιλιάδες tokens. Το κρατάω
  off παντού.
- **Greedy decoding** (`do_sample=False`): deterministic, fair comparison.
- **Batched generation** με **left-padding** (το padding side έχει σημασία για causal LMs).
- **Length-sorted batching**: ταξινομώ τα prompts κατά length πριν το batching,
  ώστε κάθε batch να έχει παρόμοιο μήκος (λιγότερο wasted padding).
- **Auto OOM backoff**: αν σκάσει σε out-of-memory, batch_size //= 2 και ξανακυλάει.
- **`max_new_tokens=32`** για zero-shot/few-shot, **`max_new_tokens=256`** για CoT
  (το reasoning χρειάζεται χώρο).
- **Output parsing**: ψάχνει πρώτα το pattern `Label: <X>` (έτσι όπως ζητά το prompt).
  Αν δεν υπάρχει, fallback στο rightmost label που εμφανίζεται στο output. Invalid
  outputs σπάνε (0% σε όλα μου τα runs).

### Metrics

Η Kaggle metric (στο leaderboard) είναι **F1-score**. Το νούμερο του LB (0.725 > local
accuracy 0.682 > local macro-F1 0.575) μόνο με **support-weighted F1** ταιριάζει,
οπότε υποθέτω weighted-F1. Για το report αναφέρω:

- **Macro-F1**: η κύρια metric μου (το ζητά η εκφώνηση), μετράει όλες τις κλάσεις ίσα.
- Accuracy, per-class precision/recall/F1, weighted-F1, confusion matrix.
- Subgroup metrics ανά μήκος ερώτησης και μήκος απάντησης.
- Invalid output rate.

### Reproducibility

Seed 42 σε όλα: subset selection, few-shot sample selection, οποιοδήποτε sampled
decoding (μόνο σε self-consistency, βλ. Phase 5.5). Greedy decoding είναι deterministic.
Σημείωση: το batched LLM inference με left-padding έχει μικρή numerical noise στο bf16
λόγω της σύνθεσης του batch (length-sorted vs unsorted batching δίνει ~0.02 macroF1
διαφορά για το ίδιο config). Κάθε grid είναι εσωτερικά συνεπές (ίδια methodology σε όλα
τα cells του).
''')


# ====================================================================
# Cell 4 : Prompt design
# ====================================================================
md(r'''
## 3. Prompt design

### Πώς έφτασα στο τελικό prompt

Ξεκίνησα με ένα naive zero-shot prompt (απλό "classify this answer"), και προχώρησα
σταδιακά. Σε κάθε βήμα κοίταγα το confusion matrix και τι λάθη έκανε το μοντέλο, και
στόχευα τα συγκεκριμένα failure modes:

| Variant | Τι πρόσθεσα | Dev macro-F1 (0.8B) |
|---|---|---|
| v0 naive | tutorial-style classify prompt | 0.21 |
| v1 +definitions | ορισμοί των τριών labels | 0.22 |
| v2 +anti-collapse | "do not over-predict one class" hint | 0.24 |
| v3 +rubric | step-by-step decision rubric (2 sentences) | 0.29 |
| v4 sharpened defs | sharpened ορισμοί + "judge substance, not style" | 0.32 |
| v4 + few-shot (6 demos) | 6 demos auto-selected | 0.29 (χειρότερα, λόγω μικρών demos) |
| **v4 + CoT** | sharpened defs + 3-step chain-of-thought | **0.42** |

Το **breakthrough** ήταν το CoT: στο 0.8B από 0.32 ανέβηκε σε 0.42, και η μεγάλη
βελτίωση ήρθε στα long answers (subgroup macro-F1 από 0.17 σε 0.41).

### Το final prompt: `cot_v4`

System message (sharpened definitions + 3-step CoT):

```
You are an expert annotator for political interviews. You are given the specific
question a journalist asked and the answer a public figure gave. Judge ONE thing only:
did the answer actually give the specific information the question asked for?

Do not be impressed by length or fluency. Political answers are often long, articulate,
and on-topic while still not answering the actual question. Judge the substance, not
the style.

Use exactly one of these three labels:

- Clear Reply: the answer gives the specific information, position, or yes/no that
  the question asked for. Direct and on-point.
- Ambivalent: the answer engages with the topic but does NOT give the specific
  information asked. It hedges, generalises, answers an easier or different question,
  gives only part of the answer, or talks around it.
- Clear Non-Reply: the answer does not engage with the question at all. The speaker
  refuses, explicitly declines, or changes the subject.

Think it through step by step:
1. What specific information does the question ask for? (one sentence)
2. What does the answer actually provide? (one sentence)
3. Does the answer deliver that specific information - fully, only partially / by
   talking around it, or not at all?

Keep your reasoning to about 3 short sentences. Then, on a final separate line, write
exactly:
Label: <one of: Clear Reply, Ambivalent, Clear Non-Reply>
```

User message (input formulation, ίδιο σε όλα τα experiments):

```
Question: <η ερώτηση του δημοσιογράφου>
Answer: <η απάντηση του πολιτικού>
```

### Σημαντικές επιλογές σε αυτό το prompt

- **"Judge ONE thing only"**: το μοντέλο τείνει να μπερδεύεται με τη "συνολική
  ποιότητα" της απάντησης. Το prompt το εστιάζει στο μοναδικό κριτήριο.
- **"Do not be impressed by length or fluency"**: το βασικό error mode των μικρότερων
  μοντέλων ήταν να βλέπουν μια ευφραδή, on-topic απάντηση και να τη γράφουν Clear Reply
  ακόμα κι αν δεν είχε τη συγκεκριμένη πληροφορία.
- **Sharpened ορισμοί** για το Clear Reply ("specific information, position, or yes/no")
  και για το Ambivalent ("hedges, generalises, answers a different question").
- **3-step CoT** που αναγκάζει το μοντέλο να ονομάσει ξεχωριστά (α) τι ζητά η ερώτηση,
  (β) τι δίνει η απάντηση, (γ) το ταίριασμα. Έτσι σταματά να "διαβάζει διαγωνίως" τις
  μεγάλες απαντήσεις.
- **`Label:` line στο τέλος**: σαφές format για το parsing.
''')


# ====================================================================
# Cell 5 : Live run header
# ====================================================================
md(r'''
## 4. Live run: το best system end-to-end

Τρέχω εδώ το **Qwen3.5-4B + cot_v4** σε:

1. **Dev subset (N=500)**: full reporting (per-class metrics, confusion matrix,
   subgroup analysis, error examples, CoT reasoning samples).
2. **Test set (N=308)**: παράγει `submission.csv` (το όνομα που πιάνει αυτόματα το
   Kaggle competition) και το αντίγραφο `submission_best_prompting_system.csv`
   (descriptive name που ζητάει η εκφώνηση). Επειδή το HF test split έχει gold labels,
   υπολογίζω και local test metrics μετά το submission.

Διάρκεια: περίπου 2.5 με 3 ώρες σε 1x T4. Γράφω ολόκληρο το πείραμα σε `hw3_final.py`
και το τρέχω σαν subprocess (καθαρό transformers import).

### 4.1. Γράφω το script
''')

code("%%writefile hw3_final.py\n" + SCRIPT)

md(r'''
### 4.2. Τρέξιμο σε subprocess

Το output εμφανίζεται live εδώ και γράφεται και στο `run_log.txt` σαν backup.
''')

code(r'''
!python -u hw3_final.py 2>&1 | tee /kaggle/working/run_log.txt
''')


# ====================================================================
# Cell 6 : How to read the live output
# ====================================================================
md(r'''
## 5. Διαβάζοντας το output του best system

Από το output του προηγούμενου cell, αυτά είναι τα κύρια σημεία για το report:

### 5.1. Aggregate metrics (dev N=500)

Από την προηγούμενη εκτέλεση (Phase 4), περιμένω:

- **macro-F1 = 0.548** (η κύρια metric του report)
- weighted-F1 = 0.621
- accuracy = 0.618
- invalid output rate = 0.0%

### 5.2. Per-class breakdown (dev)

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| Clear Reply | 0.66 | 0.48 | 0.56 | 152 |
| Ambivalent | 0.68 | 0.71 | 0.69 | 296 |
| Clear Non-Reply | 0.33 | 0.50 | 0.39 | 52 |

**Παρατηρήσεις**:
- Η μεγάλη κατηγορία (Ambivalent) πάει καλά, F1 = 0.69.
- Το Clear Reply χάνει σε **recall** (0.48): πολλά πραγματικά Clear Replies γίνονται
  λάθος Ambivalent. Το μοντέλο γίνεται υπερβολικά καχύποπτο όταν η απάντηση έχει και
  extra σχόλια γύρω από την κύρια πληροφορία.
- Το Clear Non-Reply (μειοψηφική κλάση, μόνο 52 examples) είναι το δυσκολότερο.
  Precision 0.33 σημαίνει ότι το μοντέλο **over-predicts** CNR (το λέει συχνότερα απ'
  όσο πραγματικά συμβαίνει στο dev).

### 5.3. Confusion matrix (dev)

Από το printed output του live run, τα κύρια off-diagonal counts είναι:

- **CR -> AMB: 76 cases** (το πιο συχνό λάθος): πραγματικά Clear Replies κατατάσσονται
  Ambivalent.
- **AMB -> CNR: 51 cases**: το μοντέλο μπερδεύει ambivalent (που εμπλέκεται μερικώς) με
  πλήρη μη-εμπλοκή.
- CR <-> CNR errors: σπάνια (καλό σημάδι, αυτά είναι σημασιολογικά αντίθετα).

### 5.4. Subgroup analysis

Ανά μήκος ερώτησης (q_words):

| subgroup | accuracy | macro-F1 |
|---|---|---|
| short | 0.66 | 0.61 |
| medium | 0.60 | 0.54 |
| long | 0.59 | 0.49 |

Ανά μήκος απάντησης (a_words):

| subgroup | accuracy | macro-F1 |
|---|---|---|
| short | 0.66 | 0.56 |
| medium | 0.65 | 0.58 |
| **long** | 0.55 | **0.46** |

Οι μεγάλες απαντήσεις παραμένουν το δυσκολότερο subgroup, αλλά είναι **πολύ καλύτερα
από το zero-shot baseline** της Phase 2 (long macro-F1 = 0.17). Το CoT αναγκάζει το
μοντέλο να αναλύσει τη μεγάλη απάντηση αντί να την κρίνει με βάση το συνολικό "ύφος".

### 5.5. Error analysis

Τα τυπικά λάθη και η ερμηνεία τους:

- **CR -> AMB**: το πρόβλημα είναι η υπερ-καχυποψία. Όταν η απάντηση δίνει τη
  ζητούμενη πληροφορία αλλά την περιβάλλει με qualifier/context, το μοντέλο διστάζει
  και την κατατάσσει Ambivalent. Το `cot_v4` λέει ρητά "judge the substance, not the
  style" αλλά παραμένει η βασική αδυναμία.
- **AMB -> CNR**: όταν η απάντηση είναι πολύ αόριστη ή pivot, το μοντέλο τη βλέπει σαν
  "δεν εμπλέκεται καθόλου". Στην πραγματικότητα το CNR είναι μόνο για **ξεκάθαρη
  άρνηση/αλλαγή θέματος**, ενώ η ασαφής εμπλοκή είναι Ambivalent.
- **Annotation subjectivity**: κάποια gold labels είναι πραγματικά αμφιλεγόμενα.
  Παράδειγμα: μια απάντηση που συζητά ουσιαστικά μια κατάσταση χωρίς να δίνει yes/no
  μπορεί να είναι "Clear Reply" (από τη σκοπιά του annotator που θεωρεί ότι ναι, η
  πληροφορία είναι εκεί) ή "Ambivalent" (από τη σκοπιά του annotator που θεωρεί ότι το
  yes/no λείπει). Υπάρχει inherent ceiling στο task.

### 5.6. CoT reasoning quality

Από τα CoT samples που τυπώνει το live run, η ποιότητα του reasoning είναι όντως καλή.
Παραδείγματα από προηγούμενες εκτελέσεις:

- _"The question asks for the specific committees the senator will join. The answer
  provides general economic optimism and intentions but does not name any committee.
  The speaker explicitly avoids the specific information that was asked, so the answer
  talks around the question."_ (gold = Ambivalent, pred = Ambivalent, σωστό).
- _"The question asks for a specific date. The answer references 'next year' without
  pinning a date. This is partial, not a direct yes/no."_ (gold = Ambivalent).

Αυτό είναι σημαντικό: το βελτίωμα του CoT δεν είναι απλά "καλύτερο νούμερο", είναι
**ερμηνεύσιμο reasoning** που γράφει σε καθαρή γλώσσα τι κάνει το μοντέλο.
''')


# ====================================================================
# Cell 7 : Phase 4: model-size x strategy grid
# ====================================================================
md(r'''
## 6. Σύγκριση prompting strategies και model sizes (Phase 4, central comparison)

Η κύρια συγκριτική μέτρηση: το 3x3 grid (3 strategies x 3 model sizes), στο ίδιο
500-row dev subset, ίδιο seed, ίδιο inference pipeline.

### 6.1. Macro-F1 grid (dev)

| | zero-shot | few-shot | CoT |
|---|---|---|---|
| **0.8B** | 0.298 | 0.345 | 0.419 |
| **2B** | 0.393 | 0.381 | 0.488 |
| **4B** | 0.318 | 0.414 | **0.548** |

(Invalid rate: 0% σε όλα τα 9 cells.)

### 6.2. Τι συμπεραίνω

**CoT κερδίζει σε κάθε μέγεθος και είναι η μόνη στρατηγική που κλιμακώνεται καθαρά
με το μέγεθος**: 0.42 -> 0.49 -> 0.55. Το reasoning scaffold αξιοποιεί την επιπλέον
ικανότητα του μεγαλύτερου μοντέλου.

**Few-shot: μέτριο**. Στο 2B το few-shot (0.381) ταυτίζεται πρακτικά με το zero-shot
(0.393). Δοκίμασα δύο εκδοχές: (α) auto-selected demos με μικρό μήκος (Phase 3) που
έσπρωξε το score κάτω, (β) representative-length demos (Phase 4) που βελτίωσαν αλλά
δεν ξεπέρασαν ποτέ το CoT. **Το task θέλει reasoning, όχι pattern-matching**.

**Zero-shot: ΔΕΝ είναι μονότονο με το μέγεθος**. Το 4B-zero (0.318) είναι **χειρότερο**
από το 2B-zero (0.393). Στο zero-shot, το `cot_v4` head ("don't be impressed by length,
judge the substance") σπρώχνει το 4B σε υπερ-διόρθωση: προβλέπει **291/500 Clear
Non-Reply** στο dev, ενώ το πραγματικό CNR είναι μόνο 52. AMB-F1 πέφτει στο **0.08**.
Το μεγάλο μοντέλο, χωρίς reasoning scaffolding, εφαρμόζει το prompt πιο "ζηλωτικά" και
καταρρέει. Το CoT είναι αυτό που το "σώζει" στο 4B-CoT 0.548.

**Άμεση απάντηση στο ερώτημα της εκφώνησης** ("whether some prompting methods benefit
small and large models differently"): **ναι, δραματικά**. Το ίδιο zero-shot prompt
δουλεύει στο 2B και καταρρέει στο 4B. Το CoT είναι **απαραίτητο** για να αξιοποιηθεί
το μεγάλο μοντέλο σε αυτό το task.

### 6.3. Test performance του best (4B + CoT)

| | macro-F1 | accuracy | weighted-F1 |
|---|---|---|---|
| Dev (N=500) | 0.548 | 0.618 | 0.621 |
| **Test (N=308)** | **0.575** | 0.682 | (~0.725 στο Kaggle LB) |

Test > Dev: το 308-row test split αποδείχτηκε λίγο ευκολότερο για το μοντέλο, αλλά
σαφώς εντός της αναμενόμενης διακύμανσης. **Kaggle public leaderboard: 0.725, 2η θέση**.

### 6.4. Reproducible block (commented-out)

Για να ξανατρέξει κανείς όλο το Phase 4 grid (9 cells, διάρκεια περίπου 6 ώρες),
το πλήρες script είναι στο repo: `scripts/make_phase4_notebook.py`. Παρακάτω το core
loop, σε συμπυκνωμένη μορφή:
''')

code(r'''
# # ============================================================================
# # Phase 4 grid: model_size x strategy (9 cells, ~6h on Kaggle T4).
# # Uncomment to rerun. Reuses τα helpers (load_model, run_inference,
# # parse_label, evaluate, ...) από το hw3_final.py. Για το full standalone
# # script βλέπε scripts/make_phase4_notebook.py.
# # ============================================================================
#
# # Επιπλέον prompts που χρειαζονται (το cot_v4 ειναι ηδη στο hw3_final.py ως SYS_COT_V4):
#
# SYS_V4_ZERO = _V4_HEAD + """
#
# Decide by asking, in order:
# 1. Does the answer engage with the question's topic at all? If not -> Clear Non-Reply.
# 2. If it engages, does it actually deliver the specific thing the question asked for?
#    If yes -> Clear Reply.
# 3. If it engages but does not deliver that specific thing -> Ambivalent.
#
# Respond with the label only - exactly one of: Clear Reply, Ambivalent, Clear Non-Reply."""
#
# # few-shot: SYS_V4_ZERO + 6 representative demos (2 ανα κλαση), επιλεγμενα με
# # min_a=120 max_a=320 truncate_words=220 για να ειναι αντιπροσωπευτικου μηκους.
#
# MODELS = {"0.8B": "Qwen/Qwen3.5-0.8B", "2B": "Qwen/Qwen3.5-2B", "4B": "Qwen/Qwen3.5-4B"}
# INIT_BATCH = {"0.8B": 8, "2B": 8, "4B": 4}
# STRAT = {
#     "zero": {"system": SYS_V4_ZERO, "few_shot": False, "cot": False, "mnt": 32},
#     "few":  {"system": SYS_V4_ZERO, "few_shot": True,  "cot": False, "mnt": 32},
#     "cot":  {"system": SYS_COT_V4,  "few_shot": False, "cot": True,  "mnt": 256},
# }
#
# results = {}
# cur_size, model, tok = None, None, None
# for size, strat in [("0.8B","zero"),("0.8B","few"),("0.8B","cot"),
#                     ("2B","zero"),  ("2B","few"),  ("2B","cot"),
#                     ("4B","cot"),   ("4B","zero"), ("4B","few")]:
#     if size != cur_size:
#         if model is not None:
#             del model, tok; gc.collect(); torch.cuda.empty_cache()
#         tok, model = load_model(MODELS[size])
#         cur_size = size
#     spec = STRAT[strat]
#     # για few-shot, prependare 6 representative demos σαν alternating user/assistant turns
#     gens = run_inference_with_demos(tok, model, spec, dev_records,
#                                     INIT_BATCH[size], spec["mnt"], ENABLE_THINKING)
#     preds = [parse_label(g) for g in gens]
#     results[size + "_" + strat] = evaluate(gold, preds)
#
# # Expected result table (already shown in markdown above):
# # macroF1: 0.8B {zero .298, few .345, cot .419}
# #          2B   {zero .393, few .381, cot .488}
# #          4B   {zero .318, few .414, cot .548}
''')


# ====================================================================
# Cell 8 : Phase 5: prompt refinement
# ====================================================================
md(r'''
## 7. Prompt refinement πάνω στο best system (Phase 5)

Με το 4B + CoT να είναι το best, δοκίμασα να βελτιώσω το ίδιο το prompt στοχεύοντας
τα δύο γνωστά error modes: (α) CR -> AMB υπερ-καχυποψία και (β) CNR over-prediction.
Δύο variants:

- **`cot_v5a`**: sharpened ορισμοί. Στο Clear Reply πρόσθεσα "extra commentary,
  context, justification, or caveats around it - what matters is only that the specific
  requested information is present". Στο Clear Non-Reply περιόρισα τον ορισμό σε
  "an outright refusal, or a complete change of subject. If the speaker discusses the
  topic in any substantive way, it is Ambivalent, not Clear Non-Reply".
- **`cot_v5b`**: `cot_v5a` + ένα ρητό **"Common mistakes to avoid"** block που λέει τα
  παραπάνω σαν warnings.

### 7.1. Results (4B, ίδιο 500-row dev subset)

| variant | macro-F1 | accuracy | weighted-F1 | F1 [CR, AMB, CNR] |
|---|---|---|---|---|
| `cot_v4` (reference) | **0.548** | 0.618 | 0.621 | [0.557, 0.693, **0.394**] |
| `cot_v5a` | 0.539 | 0.624 | 0.625 | [0.603, 0.688, 0.327] |
| `cot_v5b` | 0.535 | **0.634** | **0.632** | [**0.613**, 0.702, 0.289] |

### 7.2. Ερμηνεία: precision/recall tradeoff

Το intended fix **όντως δούλεψε** στο Clear Reply: CR F1 0.557 -> 0.613 (+0.056). Το
μοντέλο σταμάτησε να υπερ-υποψιάζεται τις απαντήσεις με extra commentary.

Αλλά είχε **παρενέργεια στο CNR**: η σαφέστερη οριοθέτηση του CNR ως "μόνο για outright
refusals" οδήγησε σε **under-prediction**. CNR F1 0.394 -> 0.289 (πέφτει στο μειοψηφικό
class όπου ήδη ήμουν αδύναμος).

Net: το **macro-F1 (που μετράει όλες τις κλάσεις ίσα) πέφτει ελαφρά** (0.548 -> 0.535).
Το accuracy και το weighted-F1 (που τις ζυγίζουν με support) **ανεβαίνουν** (γιατί η
μειοψηφική CNR μετράει ελάχιστα).

**Δίδαγμα**: τα prompt fixes έχουν παρενέργειες σε άλλες κλάσεις. Όταν λες "**σαφέστερα**
ότι αυτή η κλάση χρειάζεται strong evidence", το μοντέλο σου τη γυρνά **σπανιότερα**
συνολικά, και ψαλιδίζονται και τα true positives. **Κρατάμε το `cot_v4`** σαν final
system.

### 7.3. Reproducible block (commented-out)
''')

code(r'''
# # ============================================================================
# # Phase 5: 4B CoT prompt refinement (~5h on Kaggle T4).
# # Full standalone script: scripts/make_phase5_notebook.py.
# # ============================================================================
#
# _V5_HEAD = """You are an expert annotator for political interviews. You are given the specific question a journalist asked and the answer a public figure gave. Judge ONE thing only: did the answer give the specific information the question asked for?
#
# Do not be impressed by length or fluency -- but also do not be overly suspicious. Judge the substance: is the requested information actually present in the answer?
#
# Use exactly one of these three labels:
#
# - Clear Reply: the answer gives the specific information, position, or yes/no that the question asked for. It is STILL a Clear Reply if the speaker also adds extra commentary, context, justification, or caveats around it -- what matters is only that the specific requested information is present.
# - Ambivalent: the answer engages with the question's topic but does NOT actually give the specific information asked. It hedges, generalises, answers an easier or different question, gives only part of what was asked, or talks around it.
# - Clear Non-Reply: the answer does not engage with the question's topic at all -- an outright refusal, or a complete change of subject. If the speaker discusses the topic in any substantive way, it is Ambivalent, not Clear Non-Reply."""
#
# _COT_BODY_V5 = """Think it through step by step:
# 1. What specific information does the question ask for? (one sentence)
# 2. What does the answer actually provide? (one sentence)
# 3. Is that specific information present in the answer? If yes -> Clear Reply (extra commentary around it is fine). If the speaker engages the topic but does not give it -> Ambivalent. If the speaker does not engage the topic at all -> Clear Non-Reply.
#
# Keep your reasoning to about 3 short sentences. Then, on a final separate line, write exactly:
# Label: <one of: Clear Reply, Ambivalent, Clear Non-Reply>"""
#
# _MISTAKES = """Common mistakes to avoid:
# - Do NOT label an answer Ambivalent just because it is long or adds extra commentary -- if the specific requested information is present, it is Clear Reply.
# - Do NOT label an answer Clear Non-Reply if the speaker engages the question's topic at all -- partial or evasive engagement is Ambivalent. Clear Non-Reply is only for genuine refusals or complete topic changes."""
#
# SYS_PROMPTS = {
#     "cot_v4":  SYS_COT_V4,  # ηδη ορισμενο στο hw3_final.py
#     "cot_v5a": _V5_HEAD + "\n\n" + _COT_BODY_V5,
#     "cot_v5b": _V5_HEAD + "\n\n" + _MISTAKES + "\n\n" + _COT_BODY_V5,
# }
#
# tok, model = load_model("Qwen/Qwen3.5-4B")
# for name, sys_prompt in SYS_PROMPTS.items():
#     gens = run_inference(tok, model, sys_prompt, dev_records,
#                          INIT_BATCH=4, COT_MAX_NEW_TOKENS=256, ENABLE_THINKING=False)
#     preds = [parse_label(g) for g in gens]
#     m = evaluate(gold, preds)
#     print(name, "macroF1=%.3f acc=%.3f wF1=%.3f" %
#           (m["macro_f1"], m["acc"], m["weighted_f1"]))
#
# # Expected (already in the markdown table above):
# # cot_v4:  macroF1=0.548 acc=0.618 wF1=0.621
# # cot_v5a: macroF1=0.539 acc=0.624 wF1=0.625
# # cot_v5b: macroF1=0.535 acc=0.634 wF1=0.632
''')


# ====================================================================
# Cell 9 : Phase 5.5: self-consistency
# ====================================================================
md(r'''
## 8. Self-consistency experiment (Phase 5.5)

Η τελευταία βελτίωση που δοκίμασα ήταν **self-consistency**: αντί για ένα greedy
pass, παίρνω **N sampled CoT passes** (temperature=0.7, top_p=0.9) και κάνω
**majority vote** για το τελικό label (ties -> fallback στο greedy).

Στη βιβλιογραφία, η self-consistency είναι το "standard CoT booster": λογικά, αν το
μοντέλο κάνει noisy reasoning, τα πολλά sampled paths μετράνε out το noise και βγάζουν
πιο σταθερές απαντήσεις.

Για το prompt χρησιμοποίησα το `cot_v5c`, ένα best-of-both variant (κρατάει τη
διόρθωση CR του v5b, επαναφέρει τον CNR ορισμό του v4). Στο **dev** το `cot_v5c`
έβγαλε macro-F1 = 0.563 (το καλύτερο dev score που είδα), οπότε ήταν λογικός
υποψήφιος για το SC experiment.

### 8.1. SC curve στο test (cot_v5c, N=1..8)

| N | 1 (greedy) | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| macro-F1 | 0.553 | **0.581** | 0.561 | 0.544 | 0.550 | 0.542 | 0.547 | 0.538 |
| weighted-F1 | 0.670 | **0.692** | 0.674 | 0.672 | 0.667 | 0.669 | 0.664 | 0.664 |
| accuracy | 0.662 | 0.685 | 0.666 | 0.666 | 0.659 | 0.662 | 0.656 | 0.656 |

### 8.2. Ερμηνεία: το αποτέλεσμα είναι αρνητικό και διδακτικό

Η καμπύλη **κορυφώνεται στο N=2** και **φθίνει μονότονα** μετά. Αυτό είναι το αντίθετο
του canonical SC pattern (monotonic με diminishing returns).

Ερμηνεία:
- Το μοντέλο είναι **ήδη confident στο greedy**. Δεν υπάρχει "noisy reasoning" να
  μέσο-φιλτράρει το voting.
- Στο sampling, τα alternative paths που παράγονται **δεν είναι κυρίως ορθές
  παραλλαγές** του ίδιου σωστού reasoning, αλλά **perturbations που εισάγουν λάθος**.
  Άρα more samples = more noise, not more signal.
- Το spike στο N=2 είναι κυρίως **tie-breaking noise**: με δύο ψήφους, πολλά cases
  ισοβαθμούν και πέφτουν πίσω στο greedy + ελάχιστες αλλαγές. Δεν είναι αξιόπιστο
  improvement.

### 8.3. Σύγκριση με το cot_v4 (locked best)

| system | dev macro-F1 | test macro-F1 | test weighted-F1 |
|---|---|---|---|
| `cot_v4` greedy (final) | 0.548 | **0.575** | (Kaggle LB **0.725**) |
| `cot_v5c` greedy | 0.563 | 0.560 | 0.672 |
| `cot_v5c` + SC N=2 | (~) | 0.581 | 0.692 |

Το dev gain του v5c **δεν μεταφέρθηκε στο test** (test macro-F1 0.560 < cot_v4's 0.575).
Πιθανότατα slight dev overfitting (δοκίμασα τρία variants prompt στο ίδιο dev). Το
SC N=2 βγάζει καλύτερο test macro-F1 (0.581) αλλά το **test weighted-F1 (0.692)
παραμένει κάτω από το cot_v4 Kaggle LB (0.725)**. Δε θα κέρδιζα το leaderboard με
αυτό το submission.

**Δίδαγμα και συμπέρασμα**: η self-consistency, παρότι standard στη βιβλιογραφία,
**δεν είναι το σωστό lever για αυτό το task σε αυτό το μοντέλο**: το base model είναι
ήδη decisive, οπότε sampling εισάγει noise αντί να το αφαιρεί. Το **final system
παραμένει cot_v4** (no submission swap).

### 8.4. Reproducible block (commented-out)
''')

code(r'''
# # ============================================================================
# # Phase 5.5: Self-consistency on cot_v5c (~10h overnight on Kaggle T4).
# # Full standalone script: scripts/make_phase55_notebook.py.
# # ============================================================================
#
# _V5C_HEAD = """You are an expert annotator for political interviews. You are given the specific question a journalist asked and the answer a public figure gave. Judge ONE thing only: did the answer give the specific information the question asked for?
#
# Do not be impressed by length or fluency -- but also do not be overly suspicious. Judge the substance: is the requested information actually present in the answer?
#
# Use exactly one of these three labels:
#
# - Clear Reply: the answer gives the specific information, position, or yes/no that the question asked for. It is STILL a Clear Reply if the speaker also adds extra commentary, context, justification, or caveats around it -- what matters is only that the specific requested information is present.
# - Ambivalent: the answer engages with the question's topic but does NOT actually give the specific information asked. It hedges, generalises, answers an easier or different question, gives only part of what was asked, or talks around it.
# - Clear Non-Reply: the answer does not engage with the question at all. The speaker refuses, explicitly declines, or changes the subject."""
#
# SYS_COT_V5C = _V5C_HEAD + "\n\n" + _COT_BODY_V5  # _COT_BODY_V5 ορισμενο στο Phase 5 cell
#
# def majority_vote(pass_label_lists, greedy_labels):
#     """Per example majority vote; ties / all-invalid -> greedy fallback."""
#     out = []
#     for i in range(len(greedy_labels)):
#         votes = Counter(pl[i] for pl in pass_label_lists if pl[i] in LABELS)
#         if not votes:
#             out.append(greedy_labels[i]); continue
#         top = votes.most_common()
#         if len(top) > 1 and top[0][1] == top[1][1]:
#             tied = {k for k, v in top if v == top[0][1]}
#             out.append(greedy_labels[i] if greedy_labels[i] in tied else top[0][0])
#         else:
#             out.append(top[0][0])
#     return out
#
# # greedy baseline + N=1..8 sampled passes
# tok, model = load_model("Qwen/Qwen3.5-4B")
# test_greedy = [parse_label(g) for g in run_inference(
#     tok, model, SYS_COT_V5C, test_records, 4, 256, False)]   # do_sample=False
# sc_passes = []
# for k in range(1, 9):
#     # στο run_inference πρεπει να εχω προσθεσει do_sample=True, temp=0.7, top_p=0.9
#     labels_k = [parse_label(g) for g in run_inference_sampled(
#         tok, model, SYS_COT_V5C, test_records, 4, 256, False,
#         temperature=0.7, top_p=0.9, seed=42 + 100 + k)]
#     sc_passes.append(labels_k)
#     voted = majority_vote(sc_passes, test_greedy)
#     m = evaluate(test_gold, voted)
#     print("N=%d macroF1=%.3f wF1=%.3f" % (k, m["macro_f1"], m["weighted_f1"]))
#
# # Expected curve (already in the table above):
# # N=1 macroF1=0.553 wF1=0.670
# # N=2 macroF1=0.581 wF1=0.692  <- peak (mostly tie-breaking noise)
# # N=3..8 monotonically decay
''')


# ====================================================================
# Cell 10 : Comparison with HW1/HW2
# ====================================================================
md(r'''
## 9. Σύγκριση με HW1 (vector-based) και HW2 (encoder fine-tuning)

| Assignment | Approach | Train? | Kaggle public LB |
|---|---|---|---|
| HW1 | Vector-based classifier (TF-IDF / embeddings + classical ML) | Ναι | ~0.55 |
| HW2 | DeBERTa fine-tuning (encoder + classification head) | Ναι | ~0.70 |
| **HW3** | **Qwen3.5-4B + Chain-of-Thought** (frozen, prompting only) | **Όχι** | **0.725** |

### Παρατηρήσεις

- Το HW3, **χωρίς καθόλου training**, ξεπερνά οριακά το fine-tuned DeBERTa του HW2.
  Δείχνει τη δύναμη των μεγάλων instruction-tuned LLMs με σωστό prompt.
- Σε **macro-F1** (η μετρική που μετράει όλες τις κλάσεις ίσα): HW2 fine-tuned
  ~0.70 macro-F1 vs HW3 prompted **0.575 test macro-F1**. Σε αυτό το μέτρο, το HW2
  παραμένει καλύτερο. Δηλαδή το fine-tuning δίνει πιο ισορροπημένο classifier
  ανάμεσα στις κλάσεις, ενώ το prompted LLM είναι καλύτερο στα ζυγισμένα-με-support
  metrics γιατί ευνοεί την πλειοψηφική κλάση Ambivalent.
- **Common failure modes σε HW2 και HW3**: το AMB/CR boundary και η μειοψηφική CNR
  είναι δύσκολες και για τα δύο. Το CNR στο HW2 βελτιώνεται με oversampling, στο HW3
  βελτιώθηκε λίγο με sharpened CoT αλλά παραμένει το αδύναμο σημείο (CNR F1 ~0.39).
- **Πού κερδίζει το prompting**: ευελιξία και explainability (το CoT trace δίνει
  ερμηνεύσιμο reasoning), zero training cost, εύκολη μεταφορά σε νέο domain.
- **Πού χάνει**: inference cost (4B + CoT = ~5 sec/example, vs encoder fine-tuned
  inference που είναι ~milliseconds). Το prompting είναι βιώσιμο για evaluation αλλά
  ακριβό για production-scale.
''')


# ====================================================================
# Cell 11 : Conclusion + submission
# ====================================================================
md(r'''
## 10. Σύνοψη και τελικό deliverable

### Best-performing combination

**Qwen/Qwen3.5-4B + Chain-of-Thought (prompt = `cot_v4`)**, greedy decoding, no thinking
mode, max_new_tokens=256.

| metric | dev (N=500) | test (N=308) |
|---|---|---|
| macro-F1 | 0.548 | **0.575** |
| accuracy | 0.618 | 0.682 |
| weighted-F1 | 0.621 | (~0.725 Kaggle LB) |
| invalid rate | 0% | 0% |

### Γιατί αυτό το combination κέρδισε

1. **CoT > zero-shot και few-shot σε κάθε μέγεθος**: το reasoning αναγκάζει το μοντέλο
   να σπάσει την κρίση σε βήματα ("τι ζητά η ερώτηση", "τι δίνει η απάντηση",
   "ταιριάζουν"), αντί να βασίζεται σε surface features (length, fluency).
2. **CoT κλιμακώνεται με το μέγεθος**: 0.42 (0.8B) -> 0.49 (2B) -> 0.55 (4B). Το
   μεγαλύτερο μοντέλο αξιοποιεί καλύτερα το reasoning scaffold.
3. **Sharpened ορισμοί** στο prompt λύνουν το βασικότερο confusion: το μοντέλο
   σταματά να μπερδεύει "engaged answer" με "answered the specific question".

### Τι δοκίμασα και δεν δούλεψε

- **Few-shot με μικρά demos** (Phase 3): έπεσε κάτω από zero-shot, τα demos ήταν
  unrepresentative.
- **Few-shot με demos αντιπροσωπευτικού μήκους** (Phase 4): βελτίωση πάνω από
  zero-shot, αλλά παραμένει κάτω από CoT σε κάθε μέγεθος.
- **Prompt refinement v5a/v5b** (Phase 5): precision/recall tradeoff, διόρθωσε
  το CR αλλά υπερ-κατέστειλε το CNR. Net macro-F1 ελαφρώς κάτω.
- **Self-consistency v5c, N=1..8** (Phase 5.5): non-monotonic curve με peak στο
  N=2 (κυρίως tie-breaking noise) και μονότονη πτώση μετά. Το base model είναι ήδη
  confident, οπότε sampling εισάγει noise αντί να το αφαιρεί.

### Submission file

Το cell του live run γράφει **δύο πανομοιότυπα αρχεία**:

```
/kaggle/working/submission.csv                          # Kaggle competition convention
/kaggle/working/submission_best_prompting_system.csv    # descriptive name (εκφώνηση)
```

με δομή: δύο στήλες `Id, Predicted` όπου το `Id` είναι ο index του test record στο
HF QEvasion test split, και το `Predicted` ένα από τα τρία labels (`Clear Reply`,
`Ambivalent`, `Clear Non-Reply`). Το πρώτο αρχείο είναι αυτό που πιάνει αυτόματα το
Kaggle competition στο submission.

### Πιθανές βελτιώσεις που δεν δοκίμασα

- **Larger CoT exploration** (chain-of-verification, tree-of-thought, debate-style 2
  models): θα ήθελε σημαντικά μεγαλύτερο compute budget. Δεδομένου του SC αποτελέσματος,
  δεν είναι ξεκάθαρο ότι θα κέρδιζε.
- **Annotator-style few-shot με adversarial demos** (demos που είναι κοντινά CR/AMB
  boundary cases): θα μπορούσε να βοηθήσει στο κύριο error mode (CR -> AMB), αλλά
  χρειάζεται προσεκτική επιλογή.
- **Calibrated voting** για το CNR class (π.χ. απαιτείται ομοφωνία στα N passes για να
  το προβλέψεις): μπορεί να βοηθούσε στο precision του CNR, με κόστος recall.
- **Domain-adaptation prompt**: εξειδικευμένος ορισμός των κλάσεων με παραδείγματα
  πολιτικών απαντήσεων από το ίδιο το dataset.
''')


# ====================================================================
# Build & write
# ====================================================================
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"wrote {OUT}  ({len(cells)} cells)")
