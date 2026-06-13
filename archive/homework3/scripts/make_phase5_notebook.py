"""Build notebooks/hw3_phase5.ipynb — Phase 5: refine the 4B CoT prompt.

Phase 4 best = Qwen3.5-4B + CoT (dev macroF1 0.548). Its two known error modes:
  - 76 real Clear Replies wrongly downgraded to Ambivalent (model over-suspicious)
  - Clear Non-Reply over-predicted (precision 0.33)
Phase 5 tries two refined CoT prompts that target exactly those, on 4B.

Same subprocess pattern as Phase 4 (clean transformers import).
Regenerate with:  python scripts/make_phase5_notebook.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "hw3_phase5.ipynb"

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})


SCRIPT = r'''# hw3_phase5.py — refine the 4B CoT prompt. Subprocess -> clean transformers import.
import time, gc, sys
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
    print("OK -- Qwen3.5 supported (model_type =", _cfg.model_type, ")", flush=True)
except Exception as e:
    print("FATAL: cannot load Qwen3.5 config:", repr(e)[:300], flush=True)
    sys.exit(1)
assert torch.cuda.is_available(), "GPU OFF"
print("=" * 72, flush=True)

LABELS = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]

# ===== CONFIG =====
MODEL = "Qwen/Qwen3.5-4B"
DEV_SUBSET_N = 500
SEED = 42
INIT_BATCH = 4
COT_MAX_NEW_TOKENS = 256
ENABLE_THINKING = False
PREDICT_TEST = True
VARIANTS = ["cot_v4", "cot_v5a", "cot_v5b"]   # cot_v4 = Phase 4 best (reference)


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ===== DATA =====
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
        recs.append(rec)
    return recs, has_label


def build_split(records, n_dev, seed):
    labels = [r["clarity_label"] for r in records]
    idx = list(range(len(records)))
    return train_test_split(idx, train_size=n_dev, stratify=labels, random_state=seed)


# ===== PROMPTS =====
def build_user_prompt(rec):
    return f'Question: {rec["question"]}\nAnswer: {rec["interview_answer"]}'


# --- v4 head (Phase 4 best — reference) ---
_V4_HEAD = """You are an expert annotator for political interviews. You are given the specific question a journalist asked and the answer a public figure gave. Judge ONE thing only: did the answer actually give the specific information the question asked for?

Do not be impressed by length or fluency. Political answers are often long, articulate, and on-topic while still not answering the actual question. Judge the substance, not the style.

Use exactly one of these three labels:

- Clear Reply: the answer gives the specific information, position, or yes/no that the question asked for. Direct and on-point.
- Ambivalent: the answer engages with the topic but does NOT give the specific information asked. It hedges, generalises, answers an easier or different question, gives only part of the answer, or talks around it.
- Clear Non-Reply: the answer does not engage with the question at all. The speaker refuses, explicitly declines, or changes the subject."""

# --- v5 head (refined: fixes CR->AMB over-suspicion and CNR over-prediction) ---
_V5_HEAD = """You are an expert annotator for political interviews. You are given the specific question a journalist asked and the answer a public figure gave. Judge ONE thing only: did the answer give the specific information the question asked for?

Do not be impressed by length or fluency -- but also do not be overly suspicious. Judge the substance: is the requested information actually present in the answer?

Use exactly one of these three labels:

- Clear Reply: the answer gives the specific information, position, or yes/no that the question asked for. It is STILL a Clear Reply if the speaker also adds extra commentary, context, justification, or caveats around it -- what matters is only that the specific requested information is present.
- Ambivalent: the answer engages with the question's topic but does NOT actually give the specific information asked. It hedges, generalises, answers an easier or different question, gives only part of what was asked, or talks around it.
- Clear Non-Reply: the answer does not engage with the question's topic at all -- an outright refusal, or a complete change of subject. If the speaker discusses the topic in any substantive way, it is Ambivalent, not Clear Non-Reply."""

_COT_BODY_V4 = """Think it through step by step:
1. What specific information does the question ask for? (one sentence)
2. What does the answer actually provide? (one sentence)
3. Does the answer deliver that specific information - fully, only partially / by talking around it, or not at all?

Keep your reasoning to about 3 short sentences. Then, on a final separate line, write exactly:
Label: <one of: Clear Reply, Ambivalent, Clear Non-Reply>"""

_COT_BODY_V5 = """Think it through step by step:
1. What specific information does the question ask for? (one sentence)
2. What does the answer actually provide? (one sentence)
3. Is that specific information present in the answer? If yes -> Clear Reply (extra commentary around it is fine). If the speaker engages the topic but does not give it -> Ambivalent. If the speaker does not engage the topic at all -> Clear Non-Reply.

Keep your reasoning to about 3 short sentences. Then, on a final separate line, write exactly:
Label: <one of: Clear Reply, Ambivalent, Clear Non-Reply>"""

_MISTAKES = """Common mistakes to avoid:
- Do NOT label an answer Ambivalent just because it is long or adds extra commentary -- if the specific requested information is present, it is Clear Reply.
- Do NOT label an answer Clear Non-Reply if the speaker engages the question's topic at all -- partial or evasive engagement is Ambivalent. Clear Non-Reply is only for genuine refusals or complete topic changes."""

SYS = {
    "cot_v4":  _V4_HEAD + "\n\n" + _COT_BODY_V4,
    "cot_v5a": _V5_HEAD + "\n\n" + _COT_BODY_V5,
    "cot_v5b": _V5_HEAD + "\n\n" + _MISTAKES + "\n\n" + _COT_BODY_V5,
}


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


def run_variant(tok, model, system_prompt, records, batch_size,
                max_new_tokens, enable_thinking):
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


# ===== EVAL =====
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


def print_per_class(metrics):
    print("class".ljust(18) + "precision".rjust(11) + "recall".rjust(9)
          + "f1".rjust(9) + "support".rjust(9))
    for lab in LABELS:
        print(lab.ljust(18)
              + ("%.3f" % metrics["precision"][lab]).rjust(11)
              + ("%.3f" % metrics["recall"][lab]).rjust(9)
              + ("%.3f" % metrics["f1"][lab]).rjust(9)
              + str(int(metrics["support"][lab])).rjust(9))


def show_errors(records, preds, gold, max_per_cell=2):
    buckets = defaultdict(list)
    for i, (g, p) in enumerate(zip(gold, preds)):
        if g != p:
            buckets[(g, p)].append(i)
    for (g, p), idxs in sorted(buckets.items()):
        print("\n### gold=" + g + "  ->  pred=" + p + "   (" + str(len(idxs)) + " cases)")
        for i in idxs[:max_per_cell]:
            r = records[i]
            print("  Q:", r["question"][:160])
            print("  A:", r["interview_answer"][:320])
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
        print("  " + names[b].ljust(8) + "N=" + str(len(idxs)).ljust(5)
              + "acc=%.3f  macroF1=%.3f" % (m["acc"], m["macro_f1"]))


# ===== DATA PREP =====
set_seed(SEED)
all_records = load_records()
print("QEvasion train usable records:", len(all_records), flush=True)
dev_idx, _ = build_split(all_records, DEV_SUBSET_N, SEED)
dev_records = [all_records[i] for i in dev_idx]
gold = [r["clarity_label"] for r in dev_records]
print("dev subset: N=" + str(len(dev_records)),
      "class dist:", dict(Counter(gold)), flush=True)

print("\n" + "=" * 72)
print("PROMPTS UNDER TEST")
print("=" * 72)
for name in VARIANTS:
    print("\n===== " + name + " =====")
    print(SYS[name])

# ===== RUN (4B, 3 CoT variants) =====
print("\n" + "=" * 72, flush=True)
print("RUN — 4B, 3 CoT variants", flush=True)
print("=" * 72, flush=True)
t0 = time.time()
tok, model = load_model(MODEL)
print("loaded %s in %.0fs | GPU mem %.1fGB"
      % (MODEL, time.time() - t0, torch.cuda.memory_allocated() / 1e9), flush=True)

results = {}
for name in VARIANTS:
    print("\n--- " + name + " ---", flush=True)
    try:
        t0 = time.time()
        gens = run_variant(tok, model, SYS[name], dev_records,
                           INIT_BATCH, COT_MAX_NEW_TOKENS, ENABLE_THINKING)
        dt = time.time() - t0
        preds = [parse_label(g) for g in gens]
        m = evaluate(gold, preds)
        results[name] = {"gens": gens, "preds": preds, "metrics": m, "time": dt}
        print("%s: macroF1=%.3f  acc=%.3f  invalid=%.1f%%  (%.0fs)"
              % (name, m["macro_f1"], m["acc"], m["invalid_rate"] * 100, dt), flush=True)
    except Exception as e:
        print("!!!", name, "FAILED:", repr(e)[:200], flush=True)
        results[name] = {"error": repr(e)[:200]}
    torch.cuda.empty_cache()

# ===== COMPARISON =====
print("\n" + "=" * 72)
print("COMPARISON (4B, CoT variants)")
print("=" * 72)
print("variant".ljust(12) + "macroF1".rjust(9) + "acc".rjust(8)
      + "F1_CR".rjust(8) + "F1_AMB".rjust(8) + "F1_CNR".rjust(8) + "   pred_dist")
for name in VARIANTS:
    r = results.get(name)
    if r and "metrics" in r:
        m = r["metrics"]
        c = m["pred_counts"]
        print(name.ljust(12)
              + ("%.3f" % m["macro_f1"]).rjust(9) + ("%.3f" % m["acc"]).rjust(8)
              + ("%.3f" % m["f1"]["Clear Reply"]).rjust(8)
              + ("%.3f" % m["f1"]["Ambivalent"]).rjust(8)
              + ("%.3f" % m["f1"]["Clear Non-Reply"]).rjust(8)
              + "   CR=%d AMB=%d CNR=%d INV=%d"
              % (c["Clear Reply"], c["Ambivalent"], c["Clear Non-Reply"], c["Invalid"]))
    else:
        print(name.ljust(12) + "FAILED")
for name in VARIANTS:
    r = results.get(name)
    if r and "metrics" in r:
        print("\n[" + name + "] confusion matrix:")
        print(fmt_confusion(gold, r["preds"]))

# ===== ERROR ANALYSIS + SUBGROUP (best) =====
ok = {k: v for k, v in results.items() if "metrics" in v}
if ok:
    best = max(ok, key=lambda k: ok[k]["metrics"]["macro_f1"])
    print("\n" + "=" * 72)
    print("ERROR ANALYSIS — best:", best,
          "(macroF1=%.3f)" % ok[best]["metrics"]["macro_f1"])
    print("=" * 72)
    print(fmt_confusion(gold, ok[best]["preds"]))
    print()
    print_per_class(ok[best]["metrics"])
    print("\n--- example errors ---")
    show_errors(dev_records, ok[best]["preds"], gold, max_per_cell=2)
    print("\n" + "=" * 72)
    print("SUBGROUP — best:", best)
    print("=" * 72)
    subgroup_report(dev_records, ok[best]["preds"], gold, "q_words")
    print()
    subgroup_report(dev_records, ok[best]["preds"], gold, "a_words")

# ===== TEST PREDICTION =====
print("\n" + "=" * 72)
print("TEST PREDICTION -> submission.csv")
print("=" * 72)
if PREDICT_TEST and ok:
    best = max(ok, key=lambda k: ok[k]["metrics"]["macro_f1"])
    print("best variant:", best, "(dev macroF1=%.3f)" % ok[best]["metrics"]["macro_f1"])
    test_records, has_label = load_test_records()
    print("test set:", len(test_records), "examples | gold present:", has_label)
    t0 = time.time()
    test_gens = run_variant(tok, model, SYS[best], test_records,
                            INIT_BATCH, COT_MAX_NEW_TOKENS, ENABLE_THINKING)
    test_preds = [parse_label(g) for g in test_gens]
    print("predicted %d in %.0fs" % (len(test_preds), time.time() - t0))
    sub = pd.DataFrame({"Id": range(len(test_preds)), "Predicted": test_preds})
    sub.to_csv("/kaggle/working/submission.csv", index=False)
    print("wrote /kaggle/working/submission.csv")
    print("test pred distribution:", dict(Counter(test_preds)))
    print("invalid in test preds:", sum(1 for p in test_preds if p not in LABELS))
    if has_label and all(r.get("clarity_label") in LABELS for r in test_records):
        tg = [r["clarity_label"] for r in test_records]
        tm = evaluate(tg, test_preds)
        print("LOCAL test score (QEvasion test gold): macroF1=%.3f  acc=%.3f"
              % (tm["macro_f1"], tm["acc"]))
    print(sub.head(10).to_string())
else:
    print("skipped")

print("\n===== DONE =====", flush=True)
'''


# ===== the thin notebook =====
md(r'''
# HW3 — Phase 5: refine the 4B CoT prompt

Phase 4 best = **Qwen3.5-4B + CoT** (dev macroF1 0.548). Δύο γνωστά λάθη:
- 76 πραγματικά Clear Replies υποβαθμίζονται λάθος σε Ambivalent (υπερβολικός σκεπτικισμός)
- το Clear Non-Reply υπερ-προβλέπεται (precision 0.33)

Εδώ δοκιμάζουμε 2 βελτιωμένα CoT prompts που στοχεύουν ακριβώς αυτά, στο 4B:
- **cot_v4** — το Phase 4 prompt (reference)
- **cot_v5a** — αυστηροποιημένοι ορισμοί (extra commentary γύρω από μια απάντηση ΔΕΝ
  την κάνει Ambivalent· το CNR μόνο για πραγματική μη-εμπλοκή)
- **cot_v5b** — v5a + ρητό "common mistakes to avoid"

Subprocess pattern (καθαρό transformers). Στο τέλος: test prediction με το best → `submission.csv`.
''')

md(r'''
## Πριν τρέξεις

- **Settings → Accelerator → GPU**, **Internet → On**
- **Save Version → "Save & Run All (Commit)"**
- Διάρκεια ~5–6 ώρες (3 CoT runs στο 4B + test). Στείλε μου το εκτελεσμένο notebook.

## 1. Install transformers (git — Qwen3.5)
''')

code(r'''
!pip uninstall -y -q transformers
!pip install -U --no-cache-dir "transformers @ git+https://github.com/huggingface/transformers.git"
''')

md(r'''## 2. Γράφουμε το script''')

code("%%writefile hw3_phase5.py\n" + SCRIPT)

md(r'''
## 3. Τρέξιμο σε subprocess
''')

code(r'''
!python -u hw3_phase5.py 2>&1 | tee /kaggle/working/run_log.txt
''')

md(r'''
## Τι να μου στείλεις πίσω

Το εκτελεσμένο notebook — το output του cell 3 έχει: τη σύγκριση των 3 CoT variants,
confusion matrices, error/subgroup analysis του καλύτερου, και το test prediction.
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
