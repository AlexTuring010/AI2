"""Build notebooks/hw3_dev.ipynb — the Phase 2 dev harness.

One run on Qwen3.5-0.8B: fixed stratified 500-row dev subset, 4 prompt variants
(naive / +definitions / +anti-collapse hint / +decision rubric), full metrics,
confusion matrices, quick error + subgroup analysis. Text output only.

Cell sources are raw triple-single-quoted strings so notebook code may contain
double-quote and triple-double-quote literals. Regenerate with:

    python scripts/make_dev_notebook.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "hw3_dev.ipynb"

cells = []


def md(text):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip("\n").splitlines(keepends=True),
    })


def code(text):
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    })


md(r'''
# HW3 — Phase 2: Dev harness + baseline στο Qwen3.5-0.8B

Εδώ στήνουμε το **κανονικό experiment harness** και τρέχουμε το πρώτο σοβαρό πείραμα.

Από το smoke ξέρουμε ότι το naive prompt του tutorial **καταρρέει στο Clear Non-Reply**
(accuracy 0.19). Η υπόθεση: το μοντέλο δεν ξέρει τι σημαίνουν οι 3 κλάσεις. Οπότε εδώ
συγκρίνουμε 4 prompt variants πάνω σε ένα **σταθερό, stratified dev subset** (500
δείγματα από το QEvasion train):

- **v0_naive** — το naive prompt (baseline· αναπαράγουμε το collapse σωστά σε N=500)
- **v1_definitions** — + ορισμοί των 3 κλάσεων
- **v2_anticollapse** — v1 + ρητή υπόδειξη ότι το Ambivalent είναι ο συνηθισμένος ενδιάμεσος χώρος
- **v3_rubric** — v1 + ένα decision rubric 3 βημάτων

Όλα τα αποτελέσματα τυπώνονται ως κείμενο — δεν χρειάζεται κανένα download.
''')

md(r'''
## Πριν τρέξεις

- **Settings → Accelerator → GPU**
- **Settings → Internet → On** (για το pip install + το HF dataset)
- Το dataset του competition **δεν** χρειάζεται εδώ — δουλεύουμε με το HF `ailsntua/QEvasion`.
- Αν μετά το install ζητηθεί restart: Restart και ξανά Run All.

Run All (~25-30 λεπτά), και στείλε μου πίσω όλο το output (δες το τελευταίο cell).
''')

code(r'''
!pip install -q -U "transformers @ git+https://github.com/huggingface/transformers.git"
''')

md(r'''
## Control panel

Όλες οι ρυθμίσεις του experiment σε ένα σημείο — τίποτα κρυφό.
''')

code(r'''
# === CONTROL PANEL ===
MODEL_NAME      = "Qwen/Qwen3.5-0.8B"
DEV_SUBSET_N    = 500          # stratified subset του QEvasion train
SEED            = 42
BATCH_SIZE      = 8            # αν OOM -> ριξε σε 4 η 1
MAX_NEW_TOKENS  = 32           # zero-shot label-only output -> μικρο
ENABLE_THINKING = False        # zero-shot: deterministic + γρηγορο. Το thinking το μελεταμε στη Phase 3
VARIANTS_TO_RUN = ["v0_naive", "v1_definitions", "v2_anticollapse", "v3_rubric"]
''')

md(r'''
## Library — οι reusable συναρτήσεις του harness

(seeding, data loading, prompts, inference, parsing, evaluation, reporting)
''')

code(r'''
# --- lib A: imports + labels + seeding ---
import time, gc
import numpy as np
import torch
from collections import Counter, defaultdict
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_recall_fscore_support, confusion_matrix)

LABELS = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

print("lib A ok")
''')

code(r'''
# --- lib B: data ---
def load_records():
    """QEvasion train -> list of dicts με question/answer/label + word counts."""
    ds = load_dataset("ailsntua/QEvasion", split="train")
    qs, ans, labs = ds["question"], ds["interview_answer"], ds["clarity_label"]
    recs = []
    for q, a, lab in zip(qs, ans, labs):
        if lab in LABELS and q and a:
            rec = {"question": str(q), "interview_answer": str(a), "clarity_label": lab}
            rec["q_words"] = len(rec["question"].split())
            rec["a_words"] = len(rec["interview_answer"].split())
            recs.append(rec)
    return recs

def build_dev_subset(records, n, seed):
    """Stratified subset n δειγματων (ιδιες αναλογιες κλασεων, σταθερο seed)."""
    labels = [r["clarity_label"] for r in records]
    idx = list(range(len(records)))
    sub, _ = train_test_split(idx, train_size=n, stratify=labels, random_state=seed)
    return [records[i] for i in sub]

print("lib B ok")
''')

code(r'''
# --- lib C: prompts ---
def build_user_prompt(rec):
    return f'Question: {rec["question"]}\nAnswer: {rec["interview_answer"]}'

# V0: naive (tutorial baseline)
SYS_V0 = """You are a political response classifier.
Given a question and its corresponding answer, classify the answer into one of:
- Clear Reply
- Ambivalent
- Clear Non-Reply
Respond with the label only."""

# V1: + class definitions
SYS_V1 = """You are an expert annotator for political interviews. You are given a question a journalist asked and the answer a public figure gave. Decide how clearly the answer addresses the question, using exactly one of these three labels:

- Clear Reply: the answer directly and substantively addresses the question. The speaker gives the requested information or states a clear position.
- Ambivalent: the answer is partial or evasive. The speaker says something related to the question but hedges, stays vague, answers only part of it, or talks around it without really committing.
- Clear Non-Reply: the answer does not address the question at all. The speaker refuses to answer, deflects, or changes the subject.

Respond with the label only - exactly one of: Clear Reply, Ambivalent, Clear Non-Reply."""

_ANCHOR = "Respond with the label only - exactly one of:"

# V2: V1 + anti-collapse hint
SYS_V2 = SYS_V1.replace(_ANCHOR,
"""Note: most political answers are not a perfectly clear reply and not a total non-answer - they fall in between. If the answer engages with the question at all but does not fully and directly answer it, it is Ambivalent. Reserve Clear Non-Reply for answers that genuinely do not engage with the question.

""" + _ANCHOR)

# V3: V1 + decision rubric
SYS_V3 = SYS_V1.replace(_ANCHOR,
"""Decide by asking, in order:
1. Does the answer engage with the question at all? If it ignores it, refuses, or changes the subject -> Clear Non-Reply.
2. If it engages: does it fully and directly give the requested information or position? -> Clear Reply.
3. If it engages but only partially, vaguely, or with hedging and evasion -> Ambivalent.

""" + _ANCHOR)

PROMPT_VARIANTS = {
    "v0_naive": SYS_V0,
    "v1_definitions": SYS_V1,
    "v2_anticollapse": SYS_V2,
    "v3_rubric": SYS_V3,
}
print("lib C ok -", list(PROMPT_VARIANTS))
''')

code(r'''
# --- lib D: inference + parsing ---
def load_model(model_name):
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    return tok, model

def generate_batch(tok, model, system_prompt, records,
                   batch_size, max_new_tokens, enable_thinking):
    prompts = []
    for rec in records:
        msgs = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": build_user_prompt(rec)}]
        prompts.append(tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking))
    outs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        for j in range(len(batch)):
            new = gen[j][enc["input_ids"].shape[1]:]
            outs.append(tok.decode(new, skip_special_tokens=True))
    return outs

def parse_label(text):
    """Raw generation -> ενα απο τα 3 labels, αλλιως 'Invalid'."""
    t = text.strip()
    if "</think>" in t:
        t = t.split("</think>")[-1].strip()
    low = t.lower()
    for lab in LABELS:                       # exact match (καθαρη περιπτωση)
        if low == lab.lower():
            return lab
    hits = []                                # αλλιως: το label που εμφανιζεται πρωτο
    for lab in LABELS:
        pos = low.find(lab.lower())
        if pos != -1:
            hits.append((pos, -len(lab), lab))
    if hits:
        hits.sort()
        return hits[0][2]
    return "Invalid"

print("lib D ok")
''')

code(r'''
# --- lib E: evaluation + reporting ---
def evaluate(gold, preds):
    acc = accuracy_score(gold, preds)
    macro_f1 = f1_score(gold, preds, labels=LABELS, average="macro", zero_division=0)
    p, r, f, sup = precision_recall_fscore_support(
        gold, preds, labels=LABELS, zero_division=0)
    n_invalid = sum(1 for x in preds if x not in LABELS)
    counts = {lab: preds.count(lab) for lab in LABELS}
    counts["Invalid"] = n_invalid
    return {"acc": acc, "macro_f1": macro_f1,
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

def print_comparison(results):
    hdr = (f"{'variant':<18}{'macroF1':>9}{'acc':>8}{'F1_CR':>8}"
           f"{'F1_AMB':>8}{'F1_CNR':>8}{'inval%':>9}   pred_dist")
    print(hdr)
    print("-" * 86)
    for name, res in results.items():
        m = res["metrics"]
        c = m["pred_counts"]
        dist = (f"CR={c['Clear Reply']} AMB={c['Ambivalent']} "
                f"CNR={c['Clear Non-Reply']} INV={c['Invalid']}")
        print(f"{name:<18}{m['macro_f1']:>9.3f}{m['acc']:>8.3f}"
              f"{m['f1']['Clear Reply']:>8.3f}{m['f1']['Ambivalent']:>8.3f}"
              f"{m['f1']['Clear Non-Reply']:>8.3f}{m['invalid_rate']*100:>8.1f}%   {dist}")

def print_per_class(metrics):
    print(f"{'class':<18}{'precision':>11}{'recall':>9}{'f1':>9}{'support':>9}")
    for lab in LABELS:
        print(f"{lab:<18}{metrics['precision'][lab]:>11.3f}"
              f"{metrics['recall'][lab]:>9.3f}{metrics['f1'][lab]:>9.3f}"
              f"{int(metrics['support'][lab]):>9}")

def show_errors(records, preds, gold, max_per_cell=2):
    buckets = defaultdict(list)
    for i, (g, p) in enumerate(zip(gold, preds)):
        if g != p:
            buckets[(g, p)].append(i)
    for (g, p), idxs in sorted(buckets.items()):
        print(f"\n### gold={g}  ->  pred={p}   ({len(idxs)} cases)")
        for i in idxs[:max_per_cell]:
            r = records[i]
            print(f"  Q: {r['question'][:160]}")
            print(f"  A: {r['interview_answer'][:320]}")
            print()

def subgroup_report(records, preds, gold, key, n_bins=3):
    vals = sorted(r[key] for r in records)
    cuts = [vals[len(vals) * k // n_bins] for k in range(1, n_bins)]
    def binof(v):
        return sum(1 for c in cuts if v >= c)
    names = ["short", "medium", "long"][:n_bins]
    print(f"subgroup by {key}  (cut points: {cuts})")
    for b in range(n_bins):
        idxs = [i for i, r in enumerate(records) if binof(r[key]) == b]
        if not idxs:
            continue
        m = evaluate([gold[i] for i in idxs], [preds[i] for i in idxs])
        print(f"  {names[b]:<8} N={len(idxs):<4} acc={m['acc']:.3f}  "
              f"macroF1={m['macro_f1']:.3f}")

print("lib E ok")
''')

md(r'''
## 1. Δεδομένα — φτιάχνουμε το dev subset

Φορτώνουμε το QEvasion train, κρατάμε ένα **stratified** subset 500 δειγμάτων (ίδιες
αναλογίες κλάσεων με το σύνολο, σταθερό seed → αναπαραγώγιμο). Το ίδιο subset
χρησιμοποιείται σε **όλα** τα variants ώστε η σύγκριση να είναι δίκαιη.
''')

code(r'''
set_seed(SEED)
all_records = load_records()
print(f"QEvasion train usable records: {len(all_records)}")
print("full-train class distribution:", dict(Counter(r["clarity_label"] for r in all_records)))

dev_records = build_dev_subset(all_records, DEV_SUBSET_N, SEED)
gold = [r["clarity_label"] for r in dev_records]
print(f"\ndev subset: N={len(dev_records)}  (stratified by clarity_label, seed={SEED})")
print("dev class distribution:", dict(Counter(gold)))

qw = [r["q_words"] for r in dev_records]
aw = [r["a_words"] for r in dev_records]
print(f"question words: min={min(qw)} median={int(np.median(qw))} max={max(qw)}")
print(f"answer words:   min={min(aw)} median={int(np.median(aw))} max={max(aw)}")
''')

md(r'''## 2. Φόρτωση μοντέλου (Qwen3.5-0.8B)''')

code(r'''
set_seed(SEED)
assert torch.cuda.is_available(), "GPU OFF. Ενεργοποιησε Settings -> Accelerator -> GPU και κανε ξανα Run All."
print("GPU:", torch.cuda.get_device_name(0))

t0 = time.time()
tok, model = load_model(MODEL_NAME)
print(f"loaded {MODEL_NAME} in {time.time() - t0:.1f}s")
print("model device:", next(model.parameters()).device)   # πρεπει να λεει cuda
print(f"GPU mem allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
''')

md(r'''
## 3. Τρέξιμο των prompt variants

Κάθε variant τρέχει στο ίδιο dev subset, greedy decoding, ίδιο seed. Πρώτα τυπώνουμε
τα ακριβή system prompts (για το report), μετά τρέχουμε.
''')

code(r'''
for name in VARIANTS_TO_RUN:
    print("=" * 70)
    print(f"=== {name} ===")
    print(PROMPT_VARIANTS[name])
    print()
''')

code(r'''
results = {}
for name in VARIANTS_TO_RUN:
    t0 = time.time()
    gens = generate_batch(tok, model, PROMPT_VARIANTS[name], dev_records,
                          BATCH_SIZE, MAX_NEW_TOKENS, ENABLE_THINKING)
    dt = time.time() - t0
    preds = [parse_label(g) for g in gens]
    m = evaluate(gold, preds)
    results[name] = {"gens": gens, "preds": preds, "metrics": m, "time": dt}
    print(f"{name:<18} macroF1={m['macro_f1']:.3f}  acc={m['acc']:.3f}  "
          f"invalid={m['invalid_rate'] * 100:.1f}%  ({dt:.0f}s)")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
''')

md(r'''## 4. Σύγκριση των variants''')

code(r'''
print_comparison(results)
for name, res in results.items():
    print(f"\n[{name}] confusion matrix:")
    print(fmt_confusion(gold, res["preds"]))
''')

md(r'''
## 5. Error analysis του καλύτερου variant

Confusion matrix, per-class metrics, και συγκεκριμένα παραδείγματα λαθών ανά
(gold → pred) κελί.
''')

code(r'''
best = max(results, key=lambda k: results[k]["metrics"]["macro_f1"])
print(f"best variant by macro-F1: {best}  "
      f"(macroF1={results[best]['metrics']['macro_f1']:.3f})\n")

print("confusion matrix:")
print(fmt_confusion(gold, results[best]["preds"]))
print()
print_per_class(results[best]["metrics"])

print("\n--- example errors ---")
show_errors(dev_records, results[best]["preds"], gold, max_per_cell=2)
''')

md(r'''
## 6. Subgroup analysis του καλύτερου variant

Performance ανά μήκος ερώτησης και μήκος απάντησης (3 bins: short / medium / long).
''')

code(r'''
best_preds = results[best]["preds"]
print(f"subgroup analysis — best variant: {best}\n")
subgroup_report(dev_records, best_preds, gold, "q_words")
print()
subgroup_report(dev_records, best_preds, gold, "a_words")
''')

md(r'''
## Τι να μου στείλεις πίσω

Όλο το output, ειδικά:

- **Ενότητα 1** — μέγεθος & κατανομή κλάσεων του dev subset
- **Ενότητα 3** — η γραμμή ανά variant (macroF1 / acc / invalid / χρόνος)
- **Ενότητα 4** — ο πίνακας σύγκρισης + τα confusion matrices
- **Ενότητα 5** — confusion matrix, per-class metrics, παραδείγματα λαθών
- **Ενότητα 6** — οι πίνακες subgroup

Με αυτά: βλέπουμε ποιο prompt lever δουλεύει, ενημερώνω `EXPERIMENTS.md` /
`REPORT_EVIDENCE.md`, και αποφασίζουμε το επόμενο βήμα (Phase 3: few-shot, CoT,
περαιτέρω βελτίωση των prompts).
''')


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
