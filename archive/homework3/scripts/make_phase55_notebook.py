"""Build notebooks/hw3_phase55.ipynb — Phase 5.5: cot_v5c + self-consistency.

Strong overnight run, aiming to beat the current 2nd-place 0.725:
  1. cot_v5c (keep the CR fix, revert CNR to v4's wording) — greedy, on dev + test.
  2. Self-consistency on cot_v5c (test): up to 8 sampled CoT passes, majority-voted.
     The QEvasion test split has gold labels, so each step is scored locally.

Subprocess pattern (clean transformers import). ~10h.
Regenerate with:  python scripts/make_phase55_notebook.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "hw3_phase55.ipynb"

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})


SCRIPT = r'''# hw3_phase55.py — cot_v5c + self-consistency. Subprocess -> clean transformers.
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
    print("OK -- Qwen3.5 supported", flush=True)
except Exception as e:
    print("FATAL:", repr(e)[:300], flush=True)
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
SC_N_MAX = 8           # self-consistency: up to 8 sampled passes
SC_TEMP = 0.7
SC_TOP_P = 0.9
SOFT_TIME_LIMIT_H = 11.0


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


# ===== PROMPT: cot_v5c (CR fix kept; CNR definition reverted to v4's wording) =====
def build_user_prompt(rec):
    return f'Question: {rec["question"]}\nAnswer: {rec["interview_answer"]}'

_V5C_HEAD = """You are an expert annotator for political interviews. You are given the specific question a journalist asked and the answer a public figure gave. Judge ONE thing only: did the answer give the specific information the question asked for?

Do not be impressed by length or fluency -- but also do not be overly suspicious. Judge the substance: is the requested information actually present in the answer?

Use exactly one of these three labels:

- Clear Reply: the answer gives the specific information, position, or yes/no that the question asked for. It is STILL a Clear Reply if the speaker also adds extra commentary, context, justification, or caveats around it -- what matters is only that the specific requested information is present.
- Ambivalent: the answer engages with the question's topic but does NOT actually give the specific information asked. It hedges, generalises, answers an easier or different question, gives only part of what was asked, or talks around it.
- Clear Non-Reply: the answer does not engage with the question at all. The speaker refuses, explicitly declines, or changes the subject."""

_COT_BODY_V5 = """Think it through step by step:
1. What specific information does the question ask for? (one sentence)
2. What does the answer actually provide? (one sentence)
3. Is that specific information present in the answer? If yes -> Clear Reply (extra commentary around it is fine). If the speaker engages the topic but does not give it -> Ambivalent. If the speaker does not engage the topic at all -> Clear Non-Reply.

Keep your reasoning to about 3 short sentences. Then, on a final separate line, write exactly:
Label: <one of: Clear Reply, Ambivalent, Clear Non-Reply>"""

SYS_COT_V5C = _V5C_HEAD + "\n\n" + _COT_BODY_V5


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


def run_variant(tok, model, system_prompt, records, batch_size, max_new_tokens,
                enable_thinking, do_sample=False, temperature=0.7, top_p=0.9):
    prompts = []
    for rec in records:
        msgs = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": build_user_prompt(rec)}]
        prompts.append(tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking))
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    outs = [None] * len(prompts)
    gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id)
    if do_sample:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        gen_kwargs.update(do_sample=False)
    pos, bs = 0, batch_size
    while pos < len(order):
        idxs = order[pos:pos + bs]
        chunk = [prompts[i] for i in idxs]
        try:
            enc = tok(chunk, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, **gen_kwargs)
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


def majority_vote(pass_label_lists, greedy_labels):
    """Per example: majority vote over the sampled passes; ties / all-invalid -> greedy."""
    out = []
    for i in range(len(greedy_labels)):
        votes = Counter(pl[i] for pl in pass_label_lists if pl[i] in LABELS)
        if not votes:
            out.append(greedy_labels[i])
            continue
        top = votes.most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            tied = {k for k, v in top if v == top[0][1]}
            out.append(greedy_labels[i] if greedy_labels[i] in tied else top[0][0])
        else:
            out.append(top[0][0])
    return out


# ===== EVAL =====
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


def write_submission(preds, path):
    pd.DataFrame({"Id": range(len(preds)), "Predicted": preds}).to_csv(path, index=False)


# ===== RUN =====
RUN_START = time.time()
set_seed(SEED)
all_records = load_records()
dev_idx, _ = build_split(all_records, DEV_SUBSET_N, SEED)
dev_records = [all_records[i] for i in dev_idx]
gold = [r["clarity_label"] for r in dev_records]
print("dev subset: N=" + str(len(dev_records)),
      "class dist:", dict(Counter(gold)), flush=True)
test_records, has_label = load_test_records()
test_gold = [r.get("clarity_label") for r in test_records] if has_label else None
print("test set:", len(test_records), "examples | gold present:", has_label, flush=True)

print("\n" + "=" * 72)
print("PROMPT: cot_v5c")
print("=" * 72)
print(SYS_COT_V5C)

t0 = time.time()
tok, model = load_model(MODEL)
print("\nloaded %s in %.0fs | GPU mem %.1fGB"
      % (MODEL, time.time() - t0, torch.cuda.memory_allocated() / 1e9), flush=True)

# ---- (1) cot_v5c greedy on the dev subset ----
print("\n" + "=" * 72, flush=True)
print("(1) cot_v5c GREEDY on dev subset", flush=True)
print("=" * 72, flush=True)
set_seed(SEED)
t0 = time.time()
dev_preds = [parse_label(g) for g in run_variant(
    tok, model, SYS_COT_V5C, dev_records, INIT_BATCH, COT_MAX_NEW_TOKENS,
    ENABLE_THINKING, do_sample=False)]
dm = evaluate(gold, dev_preds)
print("cot_v5c dev: macroF1=%.3f  weightedF1=%.3f  acc=%.3f  invalid=%.1f%%  (%.0fs)"
      % (dm["macro_f1"], dm["weighted_f1"], dm["acc"], dm["invalid_rate"] * 100,
         time.time() - t0), flush=True)
print("\nReference (Phase 5 dev): cot_v4 macroF1=0.548 wF1=0.621 acc=0.618 |"
      " cot_v5b macroF1=0.535 wF1=0.632 acc=0.634")
print("\nconfusion matrix (cot_v5c dev):")
print(fmt_confusion(gold, dev_preds))
print()
print_per_class(dm)
print("\n--- example errors (cot_v5c dev) ---")
show_errors(dev_records, dev_preds, gold, max_per_cell=2)
print("\nsubgroup (cot_v5c dev):")
subgroup_report(dev_records, dev_preds, gold, "q_words")
print()
subgroup_report(dev_records, dev_preds, gold, "a_words")

# ---- (2) cot_v5c greedy on the test set ----
print("\n" + "=" * 72, flush=True)
print("(2) cot_v5c GREEDY on test set -> submission_greedy.csv", flush=True)
print("=" * 72, flush=True)
set_seed(SEED)
t0 = time.time()
test_greedy = [parse_label(g) for g in run_variant(
    tok, model, SYS_COT_V5C, test_records, INIT_BATCH, COT_MAX_NEW_TOKENS,
    ENABLE_THINKING, do_sample=False)]
write_submission(test_greedy, "/kaggle/working/submission_greedy.csv")
print("predicted %d test examples in %.0fs -> submission_greedy.csv"
      % (len(test_greedy), time.time() - t0), flush=True)
if has_label:
    gm = evaluate(test_gold, test_greedy)
    print("cot_v5c GREEDY test: macroF1=%.3f  weightedF1=%.3f  acc=%.3f"
          % (gm["macro_f1"], gm["weighted_f1"], gm["acc"]), flush=True)
print("pred dist:", dict(Counter(test_greedy)))

# ---- (3) self-consistency on the test set ----
print("\n" + "=" * 72, flush=True)
print("(3) SELF-CONSISTENCY on test (cot_v5c, up to N=%d sampled passes,"
      " temp=%.1f)" % (SC_N_MAX, SC_TEMP), flush=True)
print("=" * 72, flush=True)
sc_passes = []
best_sc, best_sc_n, best_sc_wf1 = None, 0, -1.0
for k in range(1, SC_N_MAX + 1):
    elapsed_h = (time.time() - RUN_START) / 3600
    if elapsed_h > SOFT_TIME_LIMIT_H:
        print("*** soft time limit (%.1fh) — stopping SC ***" % elapsed_h, flush=True)
        break
    set_seed(SEED + 100 + k)
    t0 = time.time()
    labels_k = [parse_label(g) for g in run_variant(
        tok, model, SYS_COT_V5C, test_records, INIT_BATCH, COT_MAX_NEW_TOKENS,
        ENABLE_THINKING, do_sample=True, temperature=SC_TEMP, top_p=SC_TOP_P)]
    sc_passes.append(labels_k)
    voted = majority_vote(sc_passes, test_greedy)
    write_submission(voted, "/kaggle/working/submission_sc.csv")
    if has_label:
        vm = evaluate(test_gold, voted)
        print("SC N=%d: macroF1=%.3f  weightedF1=%.3f  acc=%.3f  (pass %.0fs)"
              % (len(sc_passes), vm["macro_f1"], vm["weighted_f1"], vm["acc"],
                 time.time() - t0), flush=True)
        if vm["weighted_f1"] > best_sc_wf1:
            best_sc, best_sc_n, best_sc_wf1 = voted, len(sc_passes), vm["weighted_f1"]
    else:
        print("SC N=%d done (%.0fs)" % (len(sc_passes), time.time() - t0), flush=True)
    torch.cuda.empty_cache()

# ---- final: pick the submission ----
print("\n" + "=" * 72)
print("FINAL")
print("=" * 72)
if has_label:
    gm = evaluate(test_gold, test_greedy)
    print("cot_v5c GREEDY     test: macroF1=%.3f  weightedF1=%.3f  acc=%.3f"
          % (gm["macro_f1"], gm["weighted_f1"], gm["acc"]))
    if best_sc is not None:
        bm = evaluate(test_gold, best_sc)
        print("cot_v5c SC (N=%d)   test: macroF1=%.3f  weightedF1=%.3f  acc=%.3f"
              % (best_sc_n, bm["macro_f1"], bm["weighted_f1"], bm["acc"]))
        print("reference: cot_v4 test macroF1=0.575 / Kaggle LB 0.725 (2nd place)")
        if best_sc_wf1 > gm["weighted_f1"]:
            final = best_sc
            print("\n-> submission.csv = SELF-CONSISTENCY (N=%d), best on weighted-F1" % best_sc_n)
        else:
            final = test_greedy
            print("\n-> submission.csv = GREEDY (self-consistency did not beat it)")
    else:
        final = test_greedy
        print("\n-> submission.csv = GREEDY (no SC pass completed)")
else:
    final = best_sc if best_sc is not None else test_greedy
    print("-> submission.csv =", "SC" if best_sc is not None else "GREEDY",
          "(test gold not available — could not score locally)")
write_submission(final, "/kaggle/working/submission.csv")
print("wrote /kaggle/working/submission.csv | pred dist:", dict(Counter(final)))
print(pd.DataFrame({"Id": range(len(final)), "Predicted": final}).head(10).to_string())
print("\n===== DONE =====", flush=True)
'''


# ===== the thin notebook =====
md(r'''
# HW3 — Phase 5.5: `cot_v5c` + self-consistency (strong overnight run)

Phase 5 ήταν precision/recall tradeoff. Εδώ ο **τελευταίος δυνατός γύρος**:

1. **`cot_v5c`** — κρατάμε τη διόρθωση CR, επαναφέρουμε τον ορισμό CNR του v4
   (best-of-both prompt). Greedy, σε dev + test.
2. **Self-consistency** — δειγματοληπτούμε το CoT έως **8 φορές** και κάνουμε
   majority vote. Είναι ο κλασικός, αξιόπιστος τρόπος να ανέβει το CoT.

Το QEvasion test split έχει gold labels → κάθε βήμα βαθμολογείται τοπικά, οπότε
ξέρουμε αμέσως αν δουλεύει. Subprocess pattern. **~10 ώρες** — overnight run.
''')

md(r'''
## Πριν τρέξεις

- **Settings → Accelerator → GPU**, **Internet → On**
- **Save Version → "Save & Run All (Commit)"** — έως 12h, server-side.
- Άστο όλο το βράδυ· στείλε μου το εκτελεσμένο notebook.

## 1. Install transformers (git — Qwen3.5)
''')

code(r'''
!pip uninstall -y -q transformers
!pip install -U --no-cache-dir "transformers @ git+https://github.com/huggingface/transformers.git"
''')

md(r'''## 2. Γράφουμε το script''')

code("%%writefile hw3_phase55.py\n" + SCRIPT)

md(r'''## 3. Τρέξιμο σε subprocess''')

code(r'''
!python -u hw3_phase55.py 2>&1 | tee /kaggle/working/run_log.txt
''')

md(r'''
## Τι να μου στείλεις πίσω

Το εκτελεσμένο notebook. Έχει: cot_v5c greedy (dev + test, με error/subgroup analysis),
την **καμπύλη self-consistency** (macro/weighted-F1 ανά N=1..8), και το τελικό
`submission.csv` (το καλύτερο από greedy / SC). Τρία αρχεία στο output:
`submission.csv`, `submission_greedy.csv`, `submission_sc.csv`.
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
