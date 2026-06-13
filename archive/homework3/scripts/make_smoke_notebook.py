"""Build notebooks/hw3_smoke.ipynb — the Phase 1 smoke / environment bring-up notebook.

Cell sources are kept here as raw triple-single-quoted strings so notebook code may
freely contain double-quote literals. Regenerate with:

    python scripts/make_smoke_notebook.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "hw3_smoke.ipynb"

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
# HW3 — Smoke test / environment bring-up

**Σκοπός:** να επιβεβαιώσουμε ότι το pipeline τρέχει end-to-end στο Kaggle πριν χτίσουμε
το κανονικό experiment harness. Εδώ **δεν** κάνουμε πειράματα — μόνο smoke test.

Τι θέλουμε να μάθουμε:
1. ότι το environment install δουλεύει (Qwen3.5 + latest transformers),
2. τι δομή έχουν τα δεδομένα του Kaggle competition (columns, labels, μέγεθος test set),
3. πώς συμπεριφέρεται το Qwen "thinking mode" (πόσα tokens / χρόνο καίει),
4. πόσο χρόνο θέλει ανά sample το μικρό μοντέλο (Qwen3.5-0.8B).
''')

md(r'''
## Πριν τρέξεις — checklist

- **Settings → Accelerator → GPU** (T4 x2 ή ό,τι δίνει το Kaggle).
- **Settings → Internet → On** (χρειάζεται για το pip install και το dataset).
- **Add Input →** πρόσθεσε το dataset του competition του Assignment 3.
- Αν μετά το install cell σου ζητήσει restart: κάνε **Restart** και ξανατρέξε από την αρχή.

Όταν τελειώσει, στείλε μου πίσω **όλο** το output (δες το τελευταίο cell).
''')

code(r'''
!pip install -q -U "transformers @ git+https://github.com/huggingface/transformers.git"
''')

md(r'''## 1. Environment check''')

code(r'''
import time, os, glob, gc
import numpy as np
import pandas as pd
import torch
import transformers

print("transformers:", transformers.__version__)
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU count:", torch.cuda.device_count())
''')

md(r'''
## 2. Δεδομένα του competition — τι αρχεία μας έδωσαν

Εδώ απλώς κοιτάμε τι υπάρχει στο `/kaggle/input`. Αυτό μας λέει το schema (columns,
ονόματα labels, μέγεθος test set) που θα χρειαστούμε στη Phase 2 για το πραγματικό submission.
''')

code(r'''
# Ολα τα αρχεια που ειναι attached στο notebook.
found = False
for root, _, files in os.walk("/kaggle/input"):
    for f in sorted(files):
        p = os.path.join(root, f)
        print(p, f"({os.path.getsize(p) / 1024:.0f} KB)")
        found = True
if not found:
    print("Δεν βρεθηκε τιποτα στο /kaggle/input — μηπως δεν εγινε Add Input το competition dataset;")
''')

code(r'''
# Φορτωσε καθε CSV που θα βρει και τυπωσε shape / columns / head.
csv_paths = sorted(glob.glob("/kaggle/input/**/*.csv", recursive=True))
print("CSV files:", csv_paths, "\n")

tables = {}
for p in csv_paths:
    name = os.path.basename(p)
    try:
        df = pd.read_csv(p)
    except Exception as e:
        print(f"[skip] {name}: {e}")
        continue
    tables[name] = df
    print("=" * 70)
    print(f"{name}  | shape: {df.shape}")
    print("columns:", list(df.columns))
    print(df.head(3).to_string())
    print()
''')

code(r'''
# Προσπαθησε να βρεις το label column και τυπωσε την κατανομη κλασεων.
CLARITY = {"Clear Reply", "Ambivalent", "Clear Non-Reply"}
for name, df in tables.items():
    for col in df.columns:
        uniq = set(str(v) for v in df[col].dropna().unique())
        if uniq and uniq <= CLARITY:
            print(f"{name} -> label column '{col}':")
            print(df[col].value_counts(), "\n")
''')

md(r'''## 3. Φόρτωση μοντέλου — Qwen3.5-0.8B''')

code(r'''
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3.5-0.8B"

t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map="auto")
print(f"model + tokenizer loaded in {time.time() - t0:.1f}s")
''')

md(r'''
## 4. Μικρό δείγμα για το smoke test

Για το smoke χρησιμοποιούμε το HF dataset του tutorial (`ailsntua/QEvasion`) γιατί έχει
γνωστό, σταθερό schema. Το schema του competition (ενότητα 2) θα το χρησιμοποιήσουμε από
τη Phase 2 και μετά για το κανονικό submission.
''')

code(r'''
from datasets import load_dataset

N_SMOKE = 32
ds = load_dataset("ailsntua/QEvasion", split=f"train[:{N_SMOKE}]")
print("fields:", ds.column_names)
print("\nπαραδειγμα [0]:")
for k, v in ds[0].items():
    print(f"  {k}: {str(v)[:200]}")
''')

md(r'''## 5. Zero-shot prompt + thinking-mode probe''')

code(r'''
LABELS = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]

system_prompt = """You are a political response classifier.
Given a question and its corresponding answer, classify the answer into one of:
- Clear Reply
- Ambivalent
- Clear Non-Reply
Respond with the label only."""

def build_user_prompt(ex):
    return f'Question: {ex["question"]}\nAnswer: {ex["interview_answer"]}'

print(system_prompt)
print("\n--- user prompt example ---\n")
print(build_user_prompt(ds[0])[:600])
''')

md(r'''
### 5α. Ένα παράδειγμα με thinking mode στο default

Βάζουμε σκόπιμα μεγάλο `max_new_tokens=512` για να **δούμε** αν το μοντέλο μπαίνει σε
μεγάλο thinking trace.
''')

code(r'''
ex = ds[0]
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": build_user_prompt(ex)},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

t0 = time.time()
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=512, do_sample=False,
                         pad_token_id=tokenizer.eos_token_id)
dt = time.time() - t0
gen = out[0][inputs["input_ids"].shape[1]:]
print(f"thinking=default | new tokens: {len(gen)} | time: {dt:.1f}s | gold: {ex['clarity_label']}")
print("\nRAW OUTPUT:")
print(repr(tokenizer.decode(gen, skip_special_tokens=True)))
''')

md(r'''
### 5β. Το ίδιο παράδειγμα με `enable_thinking=False`

Δοκιμάζουμε αν το chat template του Qwen3.5 δέχεται `enable_thinking=False` και τι αλλάζει
σε tokens / χρόνο.
''')

code(r'''
try:
    text_nt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text_nt, return_tensors="pt").to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    dt = time.time() - t0
    gen = out[0][inputs["input_ids"].shape[1]:]
    print(f"enable_thinking=False | new tokens: {len(gen)} | time: {dt:.1f}s")
    print("\nRAW OUTPUT:")
    print(repr(tokenizer.decode(gen, skip_special_tokens=True)))
except Exception as e:
    print("enable_thinking=False δεν εγινε δεκτο απο το chat template:")
    print(repr(e))
''')

md(r'''
## 6. Batched inference + timing στο μικρό δείγμα

Τρέχουμε και τα 32 samples σε batches για να μετρήσουμε ρεαλιστικό χρόνο ανά sample.
Αν βγει **OOM**, ρίξε το `BATCH_SIZE` σε 4 ή 1.
''')

code(r'''
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

BATCH_SIZE = 8
MAX_NEW_TOKENS = 512

def run_batch(dataset, max_new_tokens=MAX_NEW_TOKENS, batch_size=BATCH_SIZE):
    prompts = []
    for ex in dataset:
        msgs = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": build_user_prompt(ex)}]
        prompts.append(tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True))
    outs = []
    t0 = time.time()
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        for j in range(len(batch)):
            new = gen[j][enc["input_ids"].shape[1]:]
            outs.append(tokenizer.decode(new, skip_special_tokens=True))
    return outs, time.time() - t0

generations, elapsed = run_batch(ds)
print(f"{len(generations)} samples σε {elapsed:.1f}s  ->  {elapsed / len(generations):.2f}s/sample")
''')

md(r'''## 7. Parsing + γρήγορα metrics''')

code(r'''
def parse_label(text):
    t = text.strip()
    # Αν υπαρχει thinking trace, κρατα μονο ο,τι ειναι μετα το </think>.
    if "</think>" in t:
        t = t.split("</think>")[-1].strip()
    # Ταξινομηση κατα μηκος ωστε το "Clear Reply" να μην ταιριαξει μεσα στο "Clear Non-Reply".
    for lab in sorted(LABELS, key=len, reverse=True):
        if lab.lower() in t.lower():
            return lab
    return "Invalid"

preds = [parse_label(g) for g in generations]
gold = [ds[i]["clarity_label"] for i in range(len(ds))]

valid = sum(p != "Invalid" for p in preds)
acc = np.mean([p == g for p, g in zip(preds, gold)])
print(f"valid outputs : {valid}/{len(preds)}")
print(f"accuracy      : {acc:.3f}  (smoke, N={len(preds)})")
print(f"pred dist     : {pd.Series(preds).value_counts().to_dict()}")
print(f"gold dist     : {pd.Series(gold).value_counts().to_dict()}")

print("\n--- πρωτα 8 παραδειγματα ---")
for i in range(min(8, len(preds))):
    print(f"[{i}] gold={gold[i]:<16} pred={preds[i]:<16} raw={repr(generations[i][:160])}")
''')

md(r'''
## 8. Τι να μου στείλεις πίσω

Αντέγραψε και στείλε μου **όλο** το output που τύπωσαν τα cells, ειδικά:

- **Ενότητα 1** — versions + GPU
- **Ενότητα 2** — λίστα αρχείων, columns/head των CSV, κατανομή labels του competition
- **Ενότητα 5** — raw output με thinking on/off (tokens + χρόνος και στις δύο περιπτώσεις)
- **Ενότητα 6** — το `s/sample` του batched run
- **Ενότητα 7** — valid count, accuracy, distributions, raw παραδείγματα

Με αυτά: (1) κλείνω το dataset schema, (2) ρυθμίζω thinking + `max_new_tokens`,
(3) αποφασίζουμε Kaggle-only vs Colab Pro, (4) χτίζω το harness της Phase 2.
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
