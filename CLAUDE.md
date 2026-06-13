# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **STATUS: provisional — Homework 4 (spec not yet added).**
> Homework 3 is finished and archived under `archive/homework3/` (its instructions are
> preserved verbatim in `archive/homework3/CLAUDE_homework3.md`). This file keeps the
> **project-wide working agreements** that have held across HW2 and HW3, and marks every
> assignment-specific detail as **TBD**. Once the HW4 spec lands in `refs/`, fill in the
> `## Project goal`, `## Hard assignment constraints`, `## Source priority`, `## Tutorial
> policy`, and `## Known pitfalls` sections and drop this banner.

## Project goal

**TBD — pending the HW4 specification.**

Build a reusable experiment framework for **Homework 4**. Optimize for fast iteration,
fair comparison, reproducibility, and report-ready analysis — the analysis matters more
than the raw leaderboard number.

Open until the spec is read: the dataset (HW1–HW3 used the CLARITY / QEvasion data — HW4
may continue with it or switch), the task, the required models/methods, the allowed
techniques, and the evaluation metric.

## Hard assignment constraints

**TBD — read `refs/` (the HW4 spec + any forum/instructor clarifications) first and fill
this in before writing any code.** Capture here at minimum: required models/libraries,
what is and isn't allowed (e.g. training vs inference-only), the required methods to
investigate, the deliverable format (one notebook? script?), the metric(s) to report,
and any analysis the report must contain. Grading historically ties code credit to having
the required comparisons analyzed in the report — confirm this for HW4.

## Compute and the notebook-handoff workflow

This laptop has **no GPU**. Heavy compute runs on a GPU host (Kaggle so far; possibly
Colab).

- Claude authors notebooks and `src/` modules locally.
- The user runs them on the GPU host and returns the outputs.
- Claude logs results, interprets, and decides the next experiment.
- The **final graded deliverable must run on the target platform** (Kaggle for HW1–HW3 —
  confirm for HW4).

**Round-trip economy:** every GPU run costs the user's time and scarce GPU hours (Kaggle
gives ~30h/week). Each run must do as much useful work as possible — batch many configs
per run — and results must be cached so a new idea re-runs only what changed.

**Results handoff — text first.** The user returns results as **copy-pasted text**, not
downloaded files. So:
- Notebooks must be *self-analyzing*: they compute metrics, tables, and select + print
  representative examples on the GPU host — the **printed output is the evidence**.
- No result zips for the normal iteration loop. The durable cross-session record of past
  runs is `EXPERIMENTS.md` + `REPORT_EVIDENCE.md` (text).
- Every experiment notebook ends with an explicit **"what to send back"** cell listing
  exactly what to copy; the chat handoff restates it.
- Request a file only when genuinely unavoidable — essentially only report figures (a few
  small PNGs). When a file is needed, say so explicitly and explain why.

## Leaderboard submissions (if HW4 uses a Kaggle competition)

The leaderboard is the only score on the *unlabeled test set* — an independent check
against over-tuning to the dev subset. Each submission costs an extra full-test inference
pass and Kaggle caps submissions per day, so:
- Screen experiments on the dev subset only — do not submit every one.
- Submit at milestones: an early baseline (to measure the dev↔leaderboard gap), the best
  config per major axis, and final candidates.
- Experiment notebooks expose an optional "also predict the test set" switch — off by
  default, flipped on for milestone runs.

## Run modes

Every experiment notebook should support these modes (adapt the sizes to HW4):

| Mode | Purpose |
|------|---------|
| `smoke` | Tiny subset, minimal config, 1 pass — verifies the pipeline runs |
| `dev` | Fixed representative subset, cheapest model first — screen ideas |
| `confirm` | Chosen strategies across all required configs on the full subset, fixed seed(s) |
| `final` | Assemble the single graded deliverable |

Expensive runs must not be the default. Screen every idea on the cheapest config before
promoting it.

## Source priority

When sources conflict, follow this order (update the paths once HW4 materials are in):
1. **Homework 4 specification** (`refs/…` — TBD)
2. Instructor / forum / discord clarifications (`refs/…` — TBD)
3. Course tutorial (`refs/…` — TBD)
4. Report template
5. Previous assignments — HW1, HW2, HW3 reports and notebooks (`archive/`)

## Dev vs deliverable split

- Local development: modular `.py` files in `src/`, thin orchestrator notebooks in
  `notebooks/`.
- Final deliverable notebook: `src/` inlined into a library cell via a build script in
  `scripts/`.
- The final notebook: the **best system runs end-to-end**; other experiments may be
  commented out but reproducible, with their results visible in markdown cells.
- Notebooks are thin orchestrators, not dumping grounds.

## Anti-rerun and caching rules

A new idea should rerun only the most local expensive stage, not the whole pipeline.

- Within a run, cache generations/predictions in memory (or the GPU host's working dir)
  so re-running a cell does not redo all compute. Caches live on the GPU host — they are
  not downloaded.
- Across sessions, the durable record is text: a one-row summary per run in
  `EXPERIMENTS.md` plus analysis notes in `REPORT_EVIDENCE.md`. We do not re-run a config
  whose result is already logged.
- Print for every run: config, seed, metrics, and representative examples — then log a
  one-row summary in `EXPERIMENTS.md`.

## Reproducibility

- Fix the seed everywhere it matters: subset selection, example sampling, any sampled
  decoding/training.
- Use the **same subset and seed** across comparisons so differences reflect the model or
  method, not randomness.
- Prefer deterministic settings; if sampling, fix and log the seed.
- Log the seed with every experiment result.

## Previous-assignment reuse policy

HW1 (vector-based), HW2 (encoder fine-tuning), and HW3 (prompting LLMs) are **reference
material** for the required cross-assignment comparison — not architectural constraints.
They live under `archive/` (`archive/homework2/`, `archive/homework3/`; HW1 materials are
referenced inside those).

Use them for: prior findings about the dataset, error patterns, lessons about metrics and
class imbalance, and the report comparison. **Do NOT inherit the old notebook structure.**

## Tutorial policy

**TBD — pending the HW4 tutorial.** When it arrives, treat it as the canonical reference
for the environment install, data loading, and the core API, but adapt it to reusable
experiment management — don't copy it mechanically.

## Known HW4 pitfalls

**TBD — accumulate here as we hit them.** (HW3's hard-won pitfalls — transformers-from-git
on Kaggle, thinking-mode token blowups, OOM, the `datasets`/`huggingface_hub` conflict —
are recorded in `archive/homework3/CLAUDE_homework3.md` if HW4 reuses that stack.)

## Project state tracking

Maintain these living files throughout the project (create fresh ones when HW4 work
begins — the HW3 versions are archived):

- `PROJECT_STATE.md` — current phase, open questions, next actions, report-evidence status
- `EXPERIMENTS.md` — one row per run: id, config, key metrics, takeaway
- `PROJECT_LOG.md` — compressed historical record once `PROJECT_STATE.md` gets crowded
- `REPORT_EVIDENCE.md` — story material for the report: observations, surprises,
  decisions, concrete examples, plot↔claim mapping

Update these after every meaningful step or result. Keep `PROJECT_STATE.md` lean — move
stale content to `PROJECT_LOG.md`.

## Collaboration mode

After each meaningful experiment or implementation step:
1. Summarize what changed and what we learned.
2. Say whether the result is promising.
3. Recommend the next best step.
4. State what evidence is still missing.
5. Note what should be reflected in the final report.

Do not wait for the user to ask for interpretation, next-step planning, report updates, or
reminders about missing evidence.

## Language and writing style

Chat with the user in English. Use Greek for:
- notebook markdown cells and explanations
- intermediate summaries written into project files
- the final report

Use natural student-style Greek, not polished or corporate prose. Mix in standard English
technical terms where that sounds normal — don't force awkward Greek translations for
ML/NLP/programming vocabulary. Acceptable mixed style: `"κάνουμε zero-shot prompting στο
model"`, `"κρατάμε fixed representative subset"`, `"εδώ φαίνεται improvement στο macro-F1"`.

For the final report: more organized and academically acceptable than notebook markdowns,
but still a believable Greek student voice.

## Coding style

- Small single-responsibility functions.
- All experiment settings in config objects or top-level control cells — no hidden constants.
- All prompt/template text lives in one place (e.g. `src/prompts.py`) — not scattered around.
- Maintainability beats continuity with old notebook structure.

## Version control

- Commit after each meaningful milestone (foundations ready, smoke works, dev harness
  ready, each phase confirmed, report drafted).
- Never commit `data/`, `models/`, caches, or large artifacts.
- Suggest commit messages but do not auto-commit.
