# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project goal

Build a reusable experiment framework for **Homework 3** on the CLARITY dataset, using
**prompting techniques** with instruction-tuned LLMs (no fine-tuning).
Optimize for fast iteration, fair comparison, reproducibility, and report-ready analysis.
Find the best (model size × prompting strategy) combination — but do not optimize only
for the final Kaggle score; the analysis matters more than the raw number.

## Hard assignment constraints

- **Prompting only — no training.** Classify by prompting frozen, instruction-tuned
  LLMs. No weight updates, no fine-tuning, inference only.
- Use **Hugging Face Transformers** (latest, installed from GitHub source — Qwen3.5
  needs it).
- Required models (use exactly these — instructor confirmed):
  - `Qwen/Qwen3.5-0.8B`
  - `Qwen/Qwen3.5-2B`
  - `Qwen/Qwen3.5-4B`
- Must investigate at least: **zero-shot**, **few-shot**, **chain-of-thought (CoT)**,
  plus other prompting techniques we consider relevant.
- Produce **one** Kaggle notebook with the full analysis and the final submission
  (not three — instructor confirmed).
- Use a **random seed** for reproducibility.
- Kaggle metric: F1-score. Report also: accuracy, precision, recall, Macro-F1.
- Keep the implementation compatible with report requirements:
  - prompt-design analysis
  - prompting-method comparison (zero-shot / few-shot / CoT / …)
  - model-size comparison (0.8B / 2B / 4B)
  - input formulation analysis
  - subgroup analysis (question length, answer length)
  - error analysis
  - discussion of modifications tried (including what failed and why)
  - comparison with Assignment 1 (vector-based) and Assignment 2 (encoder fine-tuning)
- **Grading note:** code without clear prompting-method and model-comparison analysis
  in the report receives no implementation credit. Experiments and report are inseparable.

## Compute and the notebook-handoff workflow

This laptop has **no GPU**. All inference runs on a GPU host.

- Claude authors notebooks and `src/` modules locally.
- The user runs them on Kaggle (and possibly Colab) and returns the outputs.
- Claude logs results, interprets, and decides the next experiment.
- The **final graded notebook must run on Kaggle.**

**Round-trip economy:** every GPU run costs the user's time and scarce GPU hours
(Kaggle gives ~30h/week). Each run must do as much useful work as possible — batch many
configs per run — and results must be cached so a new idea re-runs only what changed.

**Results handoff — text first.** The user returns results as **copy-pasted text**, not
downloaded files. So:
- Notebooks must be *self-analyzing*: they compute metrics, confusion matrices, subgroup
  tables, invalid-rate, and select + print representative error examples on the GPU host
  — the **printed output is the evidence**.
- No result zips for the normal iteration loop. The durable cross-session record of past
  runs is `EXPERIMENTS.md` + `REPORT_EVIDENCE.md` (text).
- Every experiment notebook ends with an explicit **"what to send back"** cell listing
  exactly what to copy; the chat handoff restates it.
- Request a file only when genuinely unavoidable — essentially only report figures (a
  few small PNGs, Phase 5–7). When a file is needed, say so explicitly and explain why.
- HW3 does no training, so there are no checkpoints or large artifacts — text-first is
  realistic here in a way it was not for HW2.

## Leaderboard submissions

The Kaggle leaderboard is the only score on the *unlabeled test set* — an independent,
held-out check against over-tuning prompts to the dev subset. But each submission needs
an extra inference pass over the full test set, and Kaggle caps submissions per day, so:
- Screen experiments on the dev subset only — do not submit every one.
- Submit at milestones: an early baseline (to measure the dev↔leaderboard gap), the best
  config per model size, and final candidates.
- Experiment notebooks expose an optional "also predict the test set" switch — off by
  default, flipped on for milestone runs.

## Run modes

Every experiment notebook must support these modes:

| Mode | Purpose |
|------|---------|
| `smoke` | Tiny subset (~16–32), one model, 1 pass — verifies the pipeline runs |
| `dev` | Fixed representative subset (~300–500), 0.8B first — screen prompt ideas |
| `confirm` | Chosen strategies across all 3 models on the full subset, fixed seed(s) |
| `final` | Assemble the single Kaggle deliverable notebook |

Expensive runs (2B/4B) must not be the default. Screen every idea on `Qwen3.5-0.8B`
before promoting it to the larger models.

## Source priority

When sources conflict, follow this order:
1. Homework 3 specification (`refs/AI2_Homework_3_2026v2.pdf`)
2. Instructor / forum clarifications (`refs/forum_insights.md`, `refs/discord_insights.md`)
3. Course tutorial (`refs/HW3-tutorial/` — `hw3-demo.ipynb` + slides)
4. Report template (`AI2Template`)
5. Assignment 1 & 2 reports and notebooks (`archive/homework2/`)

## Dev vs deliverable split

- Local development: modular `.py` files in `src/`, thin orchestrator notebook in `notebooks/`.
- Final Kaggle notebook: `src/` inlined into a library cell via `scripts/build_notebook.py`.
- The final notebook: the **best system runs end-to-end** (no cached `.pkl` files —
  instructor rule); other experiments may be **commented out but reproducible**, with
  their results visible in markdown cells (instructor-approved).
- Notebooks are thin orchestrators, not dumping grounds.

## Anti-rerun and caching rules

A new idea should rerun only the most local expensive stage, not the whole pipeline.

- Within a notebook run, cache generations/predictions in memory (or the GPU host's
  working dir) so re-running a cell does not redo all inference. Caches live on the GPU
  host — they are not downloaded.
- Across sessions, the durable record is text: a one-row summary per run in
  `EXPERIMENTS.md` plus analysis notes in `REPORT_EVIDENCE.md`. We do not re-run a
  config whose result is already logged.
- Scope each notebook run to the new configs being tested; keep the printed output
  comprehensive enough that nothing needs recomputing later.

Print for every run: config, seed, metrics, confusion matrix, invalid-rate, prediction
distribution, and representative example errors — then log a one-row summary in
`EXPERIMENTS.md`.

## Reproducibility

- Fix the seed everywhere it matters: subset selection, few-shot example sampling, any
  sampled decoding.
- Use the **same subset and seed** across model/strategy comparisons so differences
  reflect the model or the prompt, not randomness.
- Prefer greedy decoding (`do_sample=False`) — deterministic, makes comparisons fair.
  If sampling, fix and log the seed.
- Log the seed with every experiment result.

## Assignment 1 & 2 reuse policy

Assignment 1 and 2 materials are reference material for the required cross-assignment
comparison — not architectural constraints.

Use them for: prior findings about CLARITY, error patterns, lessons about metrics and
class imbalance, and the report comparison. HW2 results live in `archive/homework2/`.

Do NOT inherit the old notebook structure.

## Tutorial policy

`refs/HW3-tutorial/hw3-demo.ipynb` is the canonical reference for: the environment
install, dataset loading, the chat template, batched inference, and output validation.
The slides cover prompt structure, system vs user prompt, output parsing, and decoding
parameters. Adapt the tutorial to reusable experiment management — don't copy it
mechanically.

## Known HW3 pitfalls

From `refs/forum_insights.md` and `refs/discord_insights.md`:
- Qwen3.5 needs `transformers` from GitHub, and the Kaggle environment fights this:
  (1) Kaggle preinstalls a `transformers` that lacks `qwen3_5`, and a plain
  `pip install -U "transformers @ git+..."` may not override it — `pip uninstall -y
  transformers` first, then install from git. (2) The Kaggle kernel **pre-imports**
  `transformers` at boot, so after the install the running kernel keeps the stale
  version; purging `sys.modules` gets the version string right but leaves `transformers`
  internally corrupted (model classes fail to load). **Robust fix: write the experiment
  to a `.py` file and run it as a subprocess** (`!python -u run.py`) — a fresh process
  imports the new `transformers` cleanly. See `scripts/make_phase4_notebook.py`.
- Qwen "thinking mode" can burn thousands of output tokens — watch `enable_thinking`
  and `max_new_tokens`.
- Kaggle OOM is common; expect small batch sizes; clear GPU memory between models.
- A near-random F1 (~0.25) means something is wrong (e.g. the model collapses to one
  class) — analyze and fix the prompt, don't just report it.
- The Kaggle-preinstalled `datasets` library version-conflicts with the `huggingface_hub`
  that transformers-from-git pulls in (`ImportError: cannot import name
  'BucketNotFoundError'`). Don't use the `datasets` library — load the QEvasion data via
  `huggingface_hub.hf_hub_download` of the parquet files + `pandas.read_parquet`.

## Project state tracking

Maintain these living files throughout the project:

- `PROJECT_STATE.md` — current phase, open questions, next actions, report-evidence status
- `EXPERIMENTS.md` — one row per run: id, config, model, strategy, key metrics, takeaway
- `PROJECT_LOG.md` — compressed historical record once `PROJECT_STATE.md` gets crowded
- `REPORT_EVIDENCE.md` — story material for the report: observations, surprises,
  decisions, concrete error examples, plot↔claim mapping

Update these after every meaningful step or result. Keep `PROJECT_STATE.md` lean — move
stale content to `PROJECT_LOG.md`.

## Collaboration mode

After each meaningful experiment or implementation step:
1. Summarize what changed and what we learned.
2. Say whether the result is promising.
3. Recommend the next best step.
4. State what evidence is still missing.
5. Note what should be reflected in the final report.

Do not wait for the user to ask for interpretation, next-step planning, report updates,
or reminders about missing evidence.

## Language and writing style

Chat with the user in English. Use Greek for:
- notebook markdown cells and explanations
- intermediate summaries written into project files
- the final report

Use natural student-style Greek, not polished or corporate prose. Mix in standard
English technical terms where that sounds normal — don't force awkward Greek
translations for ML/NLP/programming vocabulary.

Acceptable mixed style: `"κάνουμε zero-shot prompting στο model"`, `"κρατάμε fixed
representative subset"`, `"εδώ φαίνεται improvement στο macro-F1"`.

For the final report: more organized and academically acceptable than notebook
markdowns, but still a believable Greek student voice.

## Coding style

- Small single-responsibility functions.
- All experiment settings in config objects or top-level control cells — no hidden constants.
- All prompt text lives in one place (`src/prompts.py`) — no prompt strings scattered around.
- Maintainability beats continuity with old notebook structure.

## Version control

- Commit after each meaningful milestone (foundations ready, smoke works, dev harness
  ready, each phase confirmed, report drafted).
- Never commit `data/`, `models/`, caches, or large artifacts.
- Suggest commit messages but do not auto-commit.
