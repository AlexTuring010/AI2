"""Build notebooks/hw3_phase4.ipynb — Phase 4 grid.

Kaggle's kernel pre-imports an old `transformers` at boot that does not support
`qwen3_5`, and it cannot be hot-swapped in-process (purging sys.modules corrupts it).
So the notebook is thin: it (1) installs transformers-from-git, (2) writes the whole
experiment to `hw3_run.py`, (3) runs it as a **subprocess** — a fresh Python process
that imports the freshly-installed transformers cleanly.

Regenerate with:  python scripts/make_phase4_notebook.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "hw3_phase4.ipynb"

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})


# ====================================================================
# hw3_run.py — the full experiment, run as a subprocess.
# ====================================================================
SCRIPT = r'''# hw3_run.py — HW3 Phase 4 grid. Τρεχει σαν subprocess ωστε το transformers
# να φορτωθει ΚΑΘΑΡΟ (το Kaggle kernel εχει pre-imported το παλιο transformers).
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
    print("FATAL: transformers", transformers.__version__,
          "cannot load Qwen3.5 config:", repr(e)[:300], flush=True)
    sys.exit(1)
assert torch.cuda.is_available(), "GPU OFF -- enable Settings -> Accelerator -> GPU"
print("=" * 72, flush=True)

LABELS = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]

# ===== CONFIG =====
MODELS = {"0.8B": "Qwen/Qwen3.5-0.8B", "2B": "Qwen/Qwen3.5-2B", "4B": "Qwen/Qwen3.5-4B"}
INIT_BATCH = {"0.8B": 8, "2B": 8, "4B": 4}   # auto-backs-off on OOM
DEV_SUBSET_N = 500
SEED = 42
MAX_NEW_TOKENS = 32
COT_MAX_NEW_TOKENS = 256
FEWSHOT_K_PER_CLASS = 2
ENABLE_THINKING = False
SOFT_TIME_LIMIT_H = 10.5
PREDICT_TEST = True
RUN_PLAN = [("0.8B", "zero"), ("0.8B", "few"), ("0.8B", "cot"),
            ("2B", "zero"),   ("2B", "few"),   ("2B", "cot"),
            ("4B", "cot"),    ("4B", "zero"),  ("4B", "few")]


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


def select_fewshot(records, pool_idx, k_per_class, seed,
                   min_a=120, max_a=320, truncate_words=220):
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
        for i in rng.sample(pool, min(k_per_class, len(pool))):
            ex = dict(records[i])
            w = ex["interview_answer"].split()
            if len(w) > truncate_words:
                ex["interview_answer"] = " ".join(w[:truncate_words]) + " [...]"
            shots.append(ex)
    rng.shuffle(shots)
    return shots


# ===== PROMPTS =====
def build_user_prompt(rec):
    return f'Question: {rec["question"]}\nAnswer: {rec["interview_answer"]}'


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

STRAT = {
    "zero": {"system": SYS_V4,     "few_shot": False, "cot": False},
    "few":  {"system": SYS_V4,     "few_shot": True,  "cot": False},
    "cot":  {"system": SYS_V4_COT, "few_shot": False, "cot": True},
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


def run_variant(tok, model, spec, records, fewshot_examples,
                batch_size, max_new_tokens, enable_thinking):
    shots = fewshot_examples if spec["few_shot"] else []
    prompts = []
    for rec in records:
        msgs = [{"role": "system", "content": spec["system"]}]
        for ex in shots:
            msgs.append({"role": "user", "content": build_user_prompt(ex)})
            msgs.append({"role": "assistant", "content": ex["clarity_label"]})
        msgs.append({"role": "user", "content": build_user_prompt(rec)})
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


# ===== EVAL / REPORTING =====
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


def print_grid(results):
    sizes, strats = ["0.8B", "2B", "4B"], ["zero", "few", "cot"]
    for metric in ["macro_f1", "acc"]:
        print("\n" + metric + ":")
        print(" " * 8 + "".join(s.rjust(10) for s in strats))
        for sz in sizes:
            row = sz.ljust(8)
            for st in strats:
                r = results.get(sz + "_" + st)
                row += (("%.3f" % r["metrics"][metric]) if r and "metrics" in r
                        else "--").rjust(10)
            print(row)


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
dev_idx, rest_idx = build_split(all_records, DEV_SUBSET_N, SEED)
dev_records = [all_records[i] for i in dev_idx]
gold = [r["clarity_label"] for r in dev_records]
print("dev subset: N=" + str(len(dev_records)),
      "class dist:", dict(Counter(gold)), flush=True)
fewshot_examples = select_fewshot(all_records, rest_idx, FEWSHOT_K_PER_CLASS, SEED)
print("\nfew-shot demos (" + str(len(fewshot_examples)) + "):", flush=True)
for ex in fewshot_examples:
    print("  [" + ex["clarity_label"] + "]  Q:", ex["question"][:120])
    print("       A:", ex["interview_answer"][:200])

# ===== GRID =====
print("\n" + "=" * 72, flush=True)
print("GRID — 9 runs", flush=True)
print("=" * 72, flush=True)
GRID_START = time.time()
results = {}
cur_size, model, tok = None, None, None

for size, strat in RUN_PLAN:
    elapsed_h = (time.time() - GRID_START) / 3600
    if elapsed_h > SOFT_TIME_LIMIT_H:
        print("\n*** soft time limit reached (%.1fh) — skipping rest ***" % elapsed_h,
              flush=True)
        break
    if size != cur_size:
        if model is not None:
            del model, tok
            gc.collect()
            torch.cuda.empty_cache()
            model = tok = None
        print("\n" + "#" * 60, flush=True)
        print("# loading", size, "(%.1fh elapsed)" % elapsed_h, flush=True)
        print("#" * 60, flush=True)
        try:
            t0 = time.time()
            tok, model = load_model(MODELS[size])
            cur_size = size
            print("loaded %s in %.0fs | GPU mem %.1fGB"
                  % (size, time.time() - t0, torch.cuda.memory_allocated() / 1e9),
                  flush=True)
        except Exception as e:
            print("!!! load", size, "FAILED:", repr(e)[:200], flush=True)
            cur_size = None
            continue
    spec = STRAT[strat]
    mnt = COT_MAX_NEW_TOKENS if spec["cot"] else MAX_NEW_TOKENS
    key = size + "_" + strat
    print("\n--- " + key + " ---", flush=True)
    try:
        t0 = time.time()
        gens = run_variant(tok, model, spec, dev_records, fewshot_examples,
                           INIT_BATCH[size], mnt, ENABLE_THINKING)
        dt = time.time() - t0
        preds = [parse_label(g) for g in gens]
        m = evaluate(gold, preds)
        results[key] = {"size": size, "strat": strat, "gens": gens,
                        "preds": preds, "metrics": m, "time": dt}
        print("%s: macroF1=%.3f  acc=%.3f  invalid=%.1f%%  (%.0fs)"
              % (key, m["macro_f1"], m["acc"], m["invalid_rate"] * 100, dt), flush=True)
    except Exception as e:
        print("!!!", key, "FAILED:", repr(e)[:200], flush=True)
        results[key] = {"size": size, "strat": strat, "error": repr(e)[:200]}
    torch.cuda.empty_cache()

print("\ngrid done — total elapsed %.2fh" % ((time.time() - GRID_START) / 3600), flush=True)

# ===== COMPARISON =====
print("\n" + "=" * 72)
print("GRID COMPARISON")
print("=" * 72)
print_grid(results)
print("\nper-cell detail:")
for key, r in results.items():
    if "metrics" in r:
        m = r["metrics"]
        c = m["pred_counts"]
        print("\n[" + key + "]  macroF1=%.3f acc=%.3f  F1[CR,AMB,CNR]=[%.3f,%.3f,%.3f]"
              "  invalid=%.1f%%  (%.0fs)"
              % (m["macro_f1"], m["acc"], m["f1"]["Clear Reply"],
                 m["f1"]["Ambivalent"], m["f1"]["Clear Non-Reply"],
                 m["invalid_rate"] * 100, r["time"]))
        print("  pred_dist: CR=%d AMB=%d CNR=%d INV=%d"
              % (c["Clear Reply"], c["Ambivalent"], c["Clear Non-Reply"], c["Invalid"]))
        print(fmt_confusion(gold, r["preds"]))
    else:
        print("\n[" + key + "]  FAILED:", r.get("error"))

# ===== ERROR ANALYSIS =====
ok = {k: v for k, v in results.items() if "metrics" in v}
if ok:
    best = max(ok, key=lambda k: ok[k]["metrics"]["macro_f1"])
    print("\n" + "=" * 72)
    print("ERROR ANALYSIS — best cell:", best,
          "(macroF1=%.3f)" % ok[best]["metrics"]["macro_f1"])
    print("=" * 72)
    print(fmt_confusion(gold, ok[best]["preds"]))
    print()
    print_per_class(ok[best]["metrics"])
    print("\n--- example errors ---")
    show_errors(dev_records, ok[best]["preds"], gold, max_per_cell=2)

    print("\n" + "=" * 72)
    print("SUBGROUP — best cell:", best)
    print("=" * 72)
    subgroup_report(dev_records, ok[best]["preds"], gold, "q_words")
    print()
    subgroup_report(dev_records, ok[best]["preds"], gold, "a_words")
else:
    print("\nno cell completed — skipping analysis")

# ===== CoT SAMPLES =====
cot_ok = {k: v for k, v in results.items()
          if v.get("strat") == "cot" and "metrics" in v}
if cot_ok:
    bk = max(cot_ok, key=lambda k: cot_ok[k]["metrics"]["macro_f1"])
    r = cot_ok[bk]
    print("\n" + "=" * 72)
    print("CoT REASONING SAMPLES —", bk)
    print("=" * 72)
    step = max(1, len(dev_records) // 6)
    for i in range(0, len(dev_records), step):
        print("\n[" + str(i) + "] gold=" + gold[i] + "  pred=" + r["preds"][i])
        print("Q:", dev_records[i]["question"][:140])
        print("OUTPUT:", r["gens"][i][:900])

# ===== TEST PREDICTION =====
print("\n" + "=" * 72)
print("TEST PREDICTION -> submission.csv")
print("=" * 72)
elapsed_h = (time.time() - GRID_START) / 3600
if not PREDICT_TEST or not ok:
    print("skipped (no completed cell or PREDICT_TEST=False)")
elif elapsed_h > 11.0:
    print("elapsed %.1fh — too late, skipped" % elapsed_h)
else:
    best = max(ok, key=lambda k: ok[k]["metrics"]["macro_f1"])
    bsize, bstrat = ok[best]["size"], ok[best]["strat"]
    print("best cell:", best, "(dev macroF1=%.3f)" % ok[best]["metrics"]["macro_f1"])
    test_records, has_label = load_test_records()
    print("test set:", len(test_records), "examples | gold labels present:", has_label)
    if cur_size != bsize:
        if model is not None:
            del model, tok
            gc.collect()
            torch.cuda.empty_cache()
        tok, model = load_model(MODELS[bsize])
        cur_size = bsize
    spec = STRAT[bstrat]
    mnt = COT_MAX_NEW_TOKENS if spec["cot"] else MAX_NEW_TOKENS
    t0 = time.time()
    test_gens = run_variant(tok, model, spec, test_records, fewshot_examples,
                            INIT_BATCH[bsize], mnt, ENABLE_THINKING)
    test_preds = [parse_label(g) for g in test_gens]
    print("predicted %d test examples in %.0fs" % (len(test_preds), time.time() - t0))
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

print("\n===== DONE =====", flush=True)
'''


# ====================================================================
# The notebook — thin: install, write the script, run it as a subprocess.
# ====================================================================
md(r'''
# HW3 — Phase 4: model-size × strategy grid (overnight run)

Όλο το πείραμα: **zero-shot / few-shot / CoT × 0.8B / 2B / 4B** (9 runs) στο ίδιο
500άρι dev subset, μετά test prediction με το best cell → `submission.csv`.

**Γιατί subprocess:** το Kaggle kernel ξεκινάει με pre-imported ένα παλιό `transformers`
που δεν υποστηρίζει το Qwen3.5 και δεν αλλάζει μέσα στο ίδιο kernel. Οπότε γράφουμε όλο
το πείραμα σε ένα script (`hw3_run.py`) και το τρέχουμε σαν **ξεχωριστή διεργασία** —
ένα φρέσκο Python που φορτώνει καθαρά το νέο transformers.
''')

md(r'''
## Πριν τρέξεις

- **Settings → Accelerator → GPU**, **Internet → On**
- **Save Version → "Save & Run All (Commit)"** (έως 12h, τρέχει server-side).
- Διάρκεια ~7–10 ώρες. Άστο να τρέξει το βράδυ· στείλε μου το εκτελεσμένο notebook.

## 1. Install transformers (από git — υποστηρίζει Qwen3.5)
''')

code(r'''
!pip uninstall -y -q transformers
!pip install -U --no-cache-dir "transformers @ git+https://github.com/huggingface/transformers.git"
''')

md(r'''## 2. Γράφουμε το script του πειράματος σε αρχείο''')

code("%%writefile hw3_run.py\n" + SCRIPT)

md(r'''
## 3. Τρέξιμο σε subprocess

Φρέσκο Python process → καθαρό `import transformers`. Το output εμφανίζεται live εδώ
και γράφεται και στο `run_log.txt` (backup). Διαρκεί ~7–10 ώρες.
''')

code(r'''
!python -u hw3_run.py 2>&1 | tee /kaggle/working/run_log.txt
''')

md(r'''
## Τι να μου στείλεις πίσω

Στείλε μου το **εκτελεσμένο notebook** (το output του cell 3 τα έχει όλα): το grid
comparison, τα confusion matrices, error/subgroup analysis, CoT samples, και το test
prediction. Αν το output κοπεί, υπάρχει και το `run_log.txt` στα outputs.
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
