"""Build notebooks/hw3_phase3.ipynb — Phase 3: sharpened definitions + few-shot + CoT.

One run on Qwen3.5-0.8B over the SAME fixed 500-row dev subset as Phase 2, comparing:
  v3_rubric  — Phase 2 best (reference)
  v4_sharp   — sharpened definitions (Clear Reply = answers the *specific* question)
  v5_fewshot — v4 + 6 few-shot examples (2/class, from train outside the dev subset)
  v6_cot     — v4 + chain-of-thought (reason step by step, then label)

Text output only. Regenerate with:  python scripts/make_phase3_notebook.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "hw3_phase3.ipynb"

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})


md(r'''
# HW3 — Phase 3: Sharpened definitions + few-shot + CoT

Από το Phase 2 ξέρουμε το πρόβλημα: το μοντέλο **δεν ξεχωρίζει Ambivalent από Clear
Reply** — βλέπει μια μεγάλη, ρέουσα, on-topic απάντηση και τη λέει "απάντησε"
(206/296 λάθη). Εδώ επιτιθέμεθα ακριβώς σε αυτό, με 3 levers πάνω στο best prompt
του Phase 2:

- **v3_rubric** — το best του Phase 2 (reference baseline, macroF1 0.291)
- **v4_sharp** — αυστηροποιημένοι ορισμοί: Clear Reply = απαντά στη *συγκεκριμένη*
  ερώτηση· ρητή οδηγία "μη σε εντυπωσιάζει το μήκος / η ευφράδεια"
- **v5_fewshot** — v4 + 6 few-shot παραδείγματα (2 ανά κλάση, από train *εκτός* του dev subset)
- **v6_cot** — v4 + chain-of-thought: το μοντέλο σκέφτεται βήμα-βήμα πριν δώσει label

Ίδιο dev subset (500, seed 42) με το Phase 2 → τα νούμερα είναι άμεσα συγκρίσιμα.
Όλα τα αποτελέσματα ως κείμενο — κανένα download.
''')

md(r'''
## Πριν τρέξεις

- **Settings → Accelerator → GPU**
- **Settings → Internet → On**
- Το dataset του competition δεν χρειάζεται — δουλεύουμε με το HF `ailsntua/QEvasion`.

Run All (~35-45 λεπτά — το CoT variant είναι πιο αργό), και στείλε πίσω όλο το output.
''')

code(r'''
!pip install -q -U "transformers @ git+https://github.com/huggingface/transformers.git"
''')

md(r'''
## Control panel
''')

code(r'''
# === CONTROL PANEL ===
MODEL_NAME          = "Qwen/Qwen3.5-0.8B"
DEV_SUBSET_N        = 500       # ιδιο stratified subset με το Phase 2
SEED                = 42
BATCH_SIZE          = 8         # αν OOM -> 4 η 1
MAX_NEW_TOKENS      = 32        # για label-only variants
COT_MAX_NEW_TOKENS  = 256       # για το CoT variant (χρειαζεται χωρο για reasoning)
FEWSHOT_K_PER_CLASS = 2         # few-shot παραδειγματα ανα κλαση -> 6 συνολικα
ENABLE_THINKING     = False     # κρατ. False: το μονο reasoning ειναι το ρητο CoT
VARIANTS_TO_RUN     = ["v3_rubric", "v4_sharp", "v5_fewshot", "v6_cot"]
''')

md(r'''
## Library — reusable συναρτήσεις του harness
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

def build_split(records, n_dev, seed):
    """Ιδιο call με το Phase 2 -> ιδιο dev subset. rest_idx = pool για few-shot."""
    labels = [r["clarity_label"] for r in records]
    idx = list(range(len(records)))
    dev_idx, rest_idx = train_test_split(
        idx, train_size=n_dev, stratify=labels, random_state=seed)
    return dev_idx, rest_idx

def select_fewshot(records, pool_idx, k_per_class, seed, min_a=20, max_a=180):
    """k_per_class παραδειγματα ανα κλαση απο το pool (μετριου μηκους απαντησεις)."""
    import random
    rng = random.Random(seed)
    by_class = {lab: [] for lab in LABELS}
    for i in pool_idx:
        r = records[i]
        if min_a <= r["a_words"] <= max_a:
            by_class[r["clarity_label"]].append(i)
    shots = []
    for lab in LABELS:
        pool = by_class[lab]
        picks = rng.sample(pool, min(k_per_class, len(pool)))
        shots += [records[i] for i in picks]
    rng.shuffle(shots)
    return shots

print("lib B ok")
''')

code(r'''
# --- lib C: prompts ---
def build_user_prompt(rec):
    return f'Question: {rec["question"]}\nAnswer: {rec["interview_answer"]}'

# ---- v3: Phase 2 best (reference) ----
SYS_V1 = """You are an expert annotator for political interviews. You are given a question a journalist asked and the answer a public figure gave. Decide how clearly the answer addresses the question, using exactly one of these three labels:

- Clear Reply: the answer directly and substantively addresses the question. The speaker gives the requested information or states a clear position.
- Ambivalent: the answer is partial or evasive. The speaker says something related to the question but hedges, stays vague, answers only part of it, or talks around it without really committing.
- Clear Non-Reply: the answer does not address the question at all. The speaker refuses to answer, deflects, or changes the subject.

Respond with the label only - exactly one of: Clear Reply, Ambivalent, Clear Non-Reply."""

_ANCHOR = "Respond with the label only - exactly one of:"

SYS_V3 = SYS_V1.replace(_ANCHOR,
"""Decide by asking, in order:
1. Does the answer engage with the question at all? If it ignores it, refuses, or changes the subject -> Clear Non-Reply.
2. If it engages: does it fully and directly give the requested information or position? -> Clear Reply.
3. If it engages but only partially, vaguely, or with hedging and evasion -> Ambivalent.

""" + _ANCHOR)

# ---- v4: sharpened definitions (shared head for v4 and v4-CoT) ----
_V4_HEAD = """You are an expert annotator for political interviews. You are given the specific question a journalist asked and the answer a public figure gave. Judge ONE thing only: did the answer actually give the specific information the question asked for?

Do not be impressed by length or fluency. Political answers are often long, articulate, and on-topic while still not answering the actual question. Judge the substance, not the style.

Use exactly one of these three labels:

- Clear Reply: the answer gives the specific information, position, or yes/no that the question asked for. Direct and on-point.
- Ambivalent: the answer engages with the topic but does NOT give the specific information asked. It hedges, generalises, answers an easier or different question, gives only part of the answer, or talks around it.
- Clear Non-Reply: the answer does not engage with the question at all. The speaker refuses, explicitly declines, or changes the subject."""

SYS_V4 = _V4_HEAD + """

Decide by asking, in order:
1. Does the answer engage with the question's topic at all? If not (refusal, deflection, topic change) -> Clear Non-Reply.
2. If it engages, does it actually deliver the specific thing the question asked for? If yes -> Clear Reply.
3. If it engages but does not deliver that specific thing (hedging, generalities, partial, a different question) -> Ambivalent.

Respond with the label only - exactly one of: Clear Reply, Ambivalent, Clear Non-Reply."""

SYS_V4_COT = _V4_HEAD + """

Think it through step by step:
1. What specific information does the question ask for? (one sentence)
2. What does the answer actually provide? (one sentence)
3. Does the answer deliver that specific information - fully, only partially / by talking around it, or not at all?

Keep your reasoning to about 3 short sentences. Then, on a final separate line, write exactly:
Label: <one of: Clear Reply, Ambivalent, Clear Non-Reply>"""

VARIANT_SPECS = {
    "v3_rubric":  {"system": SYS_V3,     "few_shot": False, "cot": False},
    "v4_sharp":   {"system": SYS_V4,     "few_shot": False, "cot": False},
    "v5_fewshot": {"system": SYS_V4,     "few_shot": True,  "cot": False},
    "v6_cot":     {"system": SYS_V4_COT, "few_shot": False, "cot": True},
}
print("lib C ok -", list(VARIANT_SPECS))
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

def run_variant(tok, model, spec, dev_records, fewshot_examples,
                batch_size, max_new_tokens, enable_thinking):
    shots = fewshot_examples if spec["few_shot"] else []
    prompts = []
    for rec in dev_records:
        msgs = [{"role": "system", "content": spec["system"]}]
        for ex in shots:                       # few-shot: user/assistant demo turns
            msgs.append({"role": "user", "content": build_user_prompt(ex)})
            msgs.append({"role": "assistant", "content": ex["clarity_label"]})
        msgs.append({"role": "user", "content": build_user_prompt(rec)})
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
    """Raw generation -> ενα απο τα 3 labels, αλλιως 'Invalid'.
    Δουλευει και για zero-shot (καθαρο label) και για CoT (reasoning + 'Label: X')."""
    t = text.strip()
    if "</think>" in t:
        t = t.split("</think>")[-1].strip()
    low = t.lower()
    if "label:" in low:                        # CoT: κρατα ο,τι ειναι μετα το τελευταιο 'label:'
        t = t[low.rfind("label:") + len("label:"):].strip()
        low = t.lower()
    for lab in LABELS:                         # exact match
        if low == lab.lower():
            return lab
    best = None                                # αλλιως: το label που εμφανιζεται ΤΕΛΕΥΤΑΙΟ
    for lab in LABELS:
        pos = low.rfind(lab.lower())
        if pos != -1 and (best is None or pos > best[0]
                          or (pos == best[0] and len(lab) > len(best[1]))):
            best = (pos, lab)
    return best[1] if best else "Invalid"

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
## 1. Δεδομένα — dev subset (ίδιο με Phase 2) + few-shot pool

Το dev subset βγαίνει με το **ίδιο call** όπως στο Phase 2 (seed 42) → είναι ακριβώς το
ίδιο 500άρι. Τα few-shot παραδείγματα επιλέγονται από το **υπόλοιπο** του train (εκτός
dev subset) — οπότε δεν υπάρχει leakage.
''')

code(r'''
set_seed(SEED)
all_records = load_records()
print(f"QEvasion train usable records: {len(all_records)}")

dev_idx, rest_idx = build_split(all_records, DEV_SUBSET_N, SEED)
dev_records = [all_records[i] for i in dev_idx]
gold = [r["clarity_label"] for r in dev_records]
print(f"dev subset: N={len(dev_records)}  class dist: {dict(Counter(gold))}")

fewshot_examples = select_fewshot(all_records, rest_idx, FEWSHOT_K_PER_CLASS, SEED)
print(f"\nfew-shot examples ({len(fewshot_examples)}, {FEWSHOT_K_PER_CLASS}/class, "
      f"from train outside dev subset):")
for ex in fewshot_examples:
    print(f"\n  [{ex['clarity_label']}]  ({ex['a_words']} words)")
    print(f"  Q: {ex['question'][:140]}")
    print(f"  A: {ex['interview_answer'][:260]}")
''')

md(r'''## 2. Φόρτωση μοντέλου (Qwen3.5-0.8B)''')

code(r'''
set_seed(SEED)
assert torch.cuda.is_available(), "GPU OFF. Ενεργοποιησε Settings -> Accelerator -> GPU και κανε ξανα Run All."
print("GPU:", torch.cuda.get_device_name(0))

t0 = time.time()
tok, model = load_model(MODEL_NAME)
print(f"loaded {MODEL_NAME} in {time.time() - t0:.1f}s")
print("model device:", next(model.parameters()).device)
print(f"GPU mem allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
''')

md(r'''## 3. Τα system prompts που συγκρίνουμε''')

code(r'''
for name in VARIANTS_TO_RUN:
    spec = VARIANT_SPECS[name]
    tag = ("few-shot" if spec["few_shot"] else "") + (" + CoT" if spec["cot"] else "")
    print("=" * 70)
    print(f"=== {name}  [{tag.strip(' +') or 'zero-shot'}] ===")
    print(spec["system"])
    print()
''')

md(r'''
## 4. Τρέξιμο των variants

Κάθε variant στο ίδιο dev subset, greedy decoding. Το `v6_cot` είναι πιο αργό
(παράγει reasoning).
''')

code(r'''
results = {}
for name in VARIANTS_TO_RUN:
    spec = VARIANT_SPECS[name]
    mnt = COT_MAX_NEW_TOKENS if spec["cot"] else MAX_NEW_TOKENS
    t0 = time.time()
    gens = run_variant(tok, model, spec, dev_records, fewshot_examples,
                       BATCH_SIZE, mnt, ENABLE_THINKING)
    dt = time.time() - t0
    preds = [parse_label(g) for g in gens]
    m = evaluate(gold, preds)
    results[name] = {"gens": gens, "preds": preds, "metrics": m, "time": dt}
    print(f"{name:<14} macroF1={m['macro_f1']:.3f}  acc={m['acc']:.3f}  "
          f"invalid={m['invalid_rate'] * 100:.1f}%  ({dt:.0f}s)")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
''')

md(r'''## 5. Σύγκριση των variants''')

code(r'''
print_comparison(results)
for name, res in results.items():
    print(f"\n[{name}] confusion matrix:")
    print(fmt_confusion(gold, res["preds"]))
''')

md(r'''
## 6. Δείγμα από τα CoT outputs

Βλέπουμε πώς "σκέφτεται" το μοντέλο στο `v6_cot` — αν το reasoning έχει νόημα ή είναι
απλώς φλυαρία.
''')

code(r'''
if "v6_cot" in results:
    res = results["v6_cot"]
    print("--- 6 δειγματα raw CoT outputs ---")
    for i in range(0, len(dev_records), max(1, len(dev_records) // 6))[:6]:
        print(f"\n[{i}] gold={gold[i]}  pred={res['preds'][i]}")
        print(f"Q: {dev_records[i]['question'][:140]}")
        print("OUTPUT:", repr(res["gens"][i][:600]))
else:
    print("v6_cot δεν ετρεξε")
''')

md(r'''
## 7. Error analysis του καλύτερου variant
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
## 8. Subgroup analysis του καλύτερου variant
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

- **Ενότητα 1** — το dev subset + τα 6 few-shot παραδείγματα που επιλέχθηκαν
- **Ενότητα 4** — η γραμμή ανά variant (macroF1 / acc / invalid / χρόνος)
- **Ενότητα 5** — ο πίνακας σύγκρισης + τα confusion matrices
- **Ενότητα 6** — τα δείγματα CoT outputs
- **Ενότητα 7** — error analysis του καλύτερου variant
- **Ενότητα 8** — subgroup analysis

Με αυτά βλέπουμε ποιο lever (sharpening / few-shot / CoT) δουλεύει, και αποφασίζουμε
το Phase 3.5 (συνδυασμός των winners) ή το Phase 4 (scale σε 2B / 4B).
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
