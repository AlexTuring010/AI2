# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project goal

Build a reusable experiment framework for Homework 2 on the CLARITY dataset.
Optimize for fast iteration, fair comparison, reproducibility, and report-ready analysis.
Do not optimize only for final Kaggle score.

## Hard assignment constraints

- Use PyTorch and Hugging Face Transformers.
- Use an explicit training loop.
- Do NOT use Hugging Face Trainer API.
- Required models:
  - `bert-base-uncased`
  - `distilbert-base-uncased`
  - `microsoft/deberta-v3-base`
- Produce three Kaggle notebooks, one per required model.
- Keep the implementation compatible with report requirements:
  - model comparison
  - input formulation analysis
  - hyperparameter analysis
  - subgroup analysis
  - error analysis
  - discussion of modifications tried
  - comparison with Assignment 1

## Run modes

Every training script/notebook must support these modes:

| Mode | Purpose |
|------|---------|
| `smoke` | Tiny data subset, 1 epoch — verifies pipeline runs |
| `dev` | Fixed validation split, DistilBERT first, normal epochs |
| `confirm` | Full data, 2–3 seeds — report mean ± std |
| `final` | Clean packaging into Kaggle notebooks |

Expensive training must not be the default behavior. Screen ideas on DistilBERT before promoting to BERT/DeBERTa.

## Source priority

When sources conflict, follow this order:
1. Homework 2 specification
2. Instructor/forum clarifications
3. Course tutorial notebook
4. Report template
5. Assignment 1 report and notebook

## Dev vs deliverable split

- Local development: modular `.py` files in `src/`, thin orchestrator notebooks in `notebooks/`.
- Final Kaggle notebooks: import from `src/` where Kaggle allows, otherwise inline the minimum needed.
- Notebooks are thin orchestrators, not dumping grounds.

## Anti-rerun and caching rules

A new idea should rerun only the most local expensive stage, not the whole pipeline.

Cache upstream artifacts whenever possible:
- cleaned data
- split indices
- tokenized datasets keyed by `(model/tokenizer, max_length, input_format, split_id)`
- subgroup metadata

Save for every run: config, seed, epoch metrics, best checkpoint, validation predictions, one-row summary in `EXPERIMENTS.md`.

## Reproducibility

- Use the same seed across model comparisons so differences reflect the model, not randomness.
- For confirm runs, run 2–3 seeds and report mean ± std for key metrics.
- Log the seed with every experiment result.

## Assignment 1 reuse policy

Assignment 1 materials are reference material, not architectural constraints.

Use them for: prior findings about CLARITY, reusable preprocessing ideas, report comparison, lessons about metrics/class imbalance/error patterns.

Do NOT inherit the Assignment 1 notebook structure — it became messy and should not define the architecture here.

## Tutorial policy

Use the tutorial as the preferred reference for tokenizer/encoding flow, DataLoaders, explicit PyTorch training loop, optimizer/scheduler structure, and evaluation rhythm. Adapt it to multiclass CLARITY classification and reusable experiment management — don't copy it mechanically.

## Project state tracking

Maintain three living files throughout the project:

- `PROJECT_STATE.md` — current phase, open questions, next actions, report evidence status
- `EXPERIMENTS.md` — one row per experiment: id, config hash, seed, model, key metrics, one-line takeaway
- `PROJECT_LOG.md` — compressed historical record once `PROJECT_STATE.md` gets crowded

Update these after every meaningful step or result. Keep `PROJECT_STATE.md` lean — move stale content to `PROJECT_LOG.md`.

## Collaboration mode

After each meaningful experiment or implementation step:
1. Summarize what changed and what we learned.
2. Say whether the result is promising.
3. Recommend the next best step.
4. State what evidence is still missing.
5. Note what should be reflected in the final report.

Do not wait for the user to ask for interpretation, next-step planning, report updates, or reminders about missing evidence.

## Language and writing style

Chat with the user in English. Use Greek for:
- notebook markdown cells and explanations
- intermediate summaries written into project files
- the final report

Use natural student-style Greek, not polished or corporate prose. Mix in standard English technical terms where that sounds normal — don't force awkward Greek translations for ML/NLP/programming vocabulary.

Acceptable mixed style: `"κάνουμε fine-tuning του model"`, `"κρατάμε fixed validation split"`, `"εδώ φαίνεται improvement στο macro-F1"`.

For the final report: more organized and academically acceptable than notebook markdowns, but still a believable Greek student voice.

## Coding style

- Small single-responsibility functions.
- All experiment settings in config objects or top-level control cells — no hidden constants.
- Maintainability beats continuity with old notebook structure.

## Version control

- Commit after each meaningful milestone (smoke works, dev harness ready, each model confirmed, report drafted).
- Never commit `data/`, `models/`, or checkpoints.
- Suggest commit messages but do not auto-commit.