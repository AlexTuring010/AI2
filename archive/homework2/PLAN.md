> This is the human-facing project roadmap. The authoritative rules 
> for Claude are in CLAUDE.md. If anything here conflicts with 
> CLAUDE.md, CLAUDE.md wins.

# Homework 2 — Claude Code Project Plan

This is the operating plan for working with Claude Code on the CLARITY Homework 2 assignment. Follow the phases in order. Don't skip ahead.

---

## Phase 0 — Gather source materials

Before you open Claude Code, collect these into one folder (e.g. `refs/`):

1. Homework 2 specification
2. Instructor/forum Q&A notes
3. Course tutorial notebook
4. Report template
5. Assignment 1 report
6. Assignment 1 notebook

**Authority order** (tell Claude to respect this when sources disagree):

1. Homework 2 spec → hard requirements
2. Instructor/forum answers → clarifications
3. Tutorial → implementation style reference
4. Report template → report scaffold
5. Assignment 1 materials → task insight and baseline comparison only, **not** architecture template

---

## Phase 1 — Set up the project directory

Inside WSL, create a clean working folder on the Linux filesystem (not `/mnt/c/...` — it's much slower):

```bash
mkdir -p ~/projects/hw2-clarity
cd ~/projects/hw2-clarity
mkdir data notebooks src models refs
touch README.md .gitignore
```

Put a sensible `.gitignore` in place:

```
data/
models/
*.h5
*.pt
*.pkl
.ipynb_checkpoints/
__pycache__/
.env
venv/
```

Initialize git and connect to GitHub:

```bash
git init
git add .
git commit -m "initial project structure"
git remote add origin https://github.com/<your-username>/<your-repo>.git
git branch -M main
git push -u origin main
```

Drop your Phase 0 reference files into `refs/`.

---

## Phase 2 — Launch Claude Code and run `/init`

From the project root:

```bash
claude
```

Once inside, run:

```
/init
```

This lets Claude scan the project and generate a starter `CLAUDE.md`. **Don't accept it as-is** — you're going to overwrite it with the version below, but running `/init` first gives Claude a mental model of your repo.

---

## Phase 3 — Write `CLAUDE.md`

Create `CLAUDE.md` in the project root with the following content:

```md
# Project goal
Build a reusable experiment framework for Homework 2 on the CLARITY dataset.
Optimize for fast iteration, fair comparison, reproducibility, and report-ready analysis.
Do not optimize only for final Kaggle score.

# Hard assignment constraints
- Use PyTorch and Hugging Face Transformers.
- Use an explicit training loop.
- Do NOT use Hugging Face Trainer API.
- Required models:
  - bert-base-uncased
  - distilbert-base-uncased
  - microsoft/deberta-v3-base
- Produce three Kaggle notebooks, one per required model.
- Keep the implementation compatible with report requirements:
  - model comparison
  - input formulation analysis
  - hyperparameter analysis
  - subgroup analysis
  - error analysis
  - discussion of modifications tried
  - comparison with Assignment 1

# Source priority
1. Homework 2 specification
2. Instructor/forum clarifications
3. Course tutorial notebook
4. Report template
5. Assignment 1 report and notebook

# Assignment 1 reuse policy
Assignment 1 materials are reference material, not architectural constraints.

Use them for:
- prior findings about the CLARITY task
- reusable preprocessing ideas when appropriate
- report comparison with the previous assignment
- lessons about metrics, class imbalance, and error patterns

Do not inherit the Assignment 1 notebook structure by default.
That notebook became messy and should not define the architecture for Assignment 2.

# Tutorial policy
Use the tutorial as the preferred reference for:
- tokenizer / encoding flow
- dataloaders
- explicit PyTorch training loop
- optimizer/scheduler structure
- evaluation rhythm

But do not copy it mechanically. Adapt it to multiclass CLARITY
classification, the three required models, reusable experiment
management, report-ready analysis, and Kaggle submission constraints.

# Dev vs deliverable split
- Local development: modular .py files in src/, thin notebooks in notebooks/.
- Final Kaggle notebooks: import from src/ where Kaggle allows, otherwise inline the minimum needed.
- Notebooks are thin orchestrators, not dumping grounds.

# Workflow rules
- Always support smoke, dev, and full modes.
- Expensive training must not be the default behavior.
- Use a fixed validation split for fair comparison.
- Keep seeds explicit.

# Reproducibility and seeds
- Use the same seed across model comparisons so differences reflect the model, not randomness.
- For final confirm runs, run 2–3 seeds and report mean ± std for key metrics.
- Log the seed with every experiment result.

# Anti-rerun workflow rules
A new idea should rerun only the most local expensive stage, not the whole pipeline.

Cache upstream artifacts whenever possible:
- cleaned data
- split indices
- tokenized datasets keyed by (model/tokenizer, max_length, input format, split id)
- subgroup metadata

Save for every run:
- config
- seed
- epoch metrics
- best checkpoint
- validation predictions
- one-row summary in the experiments table

Screen ideas on a cheap model (DistilBERT) before promoting them.

# Version control
- Commit after each meaningful milestone (smoke works, dev harness ready, each model confirmed, report drafted).
- Never commit data/, models/, or checkpoints.
- Claude should suggest commit messages but not auto-commit.

# Collaboration mode — active guide, not passive assistant
Continuously track:
- current phase
- completed work
- open uncertainties
- next recommended action
- report implications of latest results

After each meaningful experiment or implementation step:
1. summarize what changed
2. summarize what we learned
3. say whether the result is promising or not
4. recommend the next best step
5. state what evidence is still missing
6. note what should be reflected in the final report

Do not wait for the user to explicitly ask for interpretation,
next-step planning, report updates, or reminders about missing evidence.
That is part of your default job.

# Project state tracking
Maintain three living files throughout the project:

- PROJECT_STATE.md — current phase, open questions, next actions, report evidence status
- EXPERIMENTS.md — one row per experiment: id, config hash, seed, model, key metrics, one-line takeaway
- PROJECT_LOG.md — compressed historical record once PROJECT_STATE.md gets crowded

Update these files whenever a meaningful step or result changes the project state.
Keep PROJECT_STATE.md lean — move stale content to PROJECT_LOG.md.

# Language and writing style
Chat with the user in English. But use Greek for:
- notebook markdown cells
- notebook explanations
- intermediate summaries written into project files
- the final report

Use natural student-style Greek, not polished or corporate prose. Mix
in standard English technical terms where that sounds normal — don't
force awkward Greek translations for ML/NLP/programming vocabulary.

Preferences:
- human, clear, slightly informal where appropriate, but suitable for a university assignment
- avoid robotic, theatrical, or excessively formal tone
- no generic filler or repetitive phrasing
- concise, grounded explanations that sound like a real student wrote them
- keep model names, library names, parameter names, and metrics in English

Acceptable mixed style examples:
- "κάνουμε fine-tuning του model"
- "κρατάμε fixed validation split"
- "το training loop είναι explicit"
- "εδώ φαίνεται improvement στο macro-F1"

For the final report: more organized and academically acceptable than
notebook markdowns, but still a believable Greek student voice. Not
translated AI prose.

# Report format
The final report must be written in LaTeX.

- The university template is located at refs/AI2_Template.zip — extract and use it as the base structure.
- Do not invent a custom LaTeX structure; adapt the provided template.
- Tables should use booktabs style (\toprule, \midrule, \bottomrule).
- Figures should be properly captioned and labeled for \ref{} cross-referencing.
- All metrics and numbers should be inside math mode where appropriate.
- Keep the .tex file clean and readable — one sentence per line in the source.
- Claude should draft each section as LaTeX-ready text, not plain prose that needs converting later.

# Coding style
- Readable, testable code with small single-responsibility functions.
- All experiment settings in config objects or top-level control cells.
- No hidden constants.
- Maintainability beats continuity with old notebook structure.
```

---

## Phase 4 — Create `PROJECT_STATE.md`, `EXPERIMENTS.md`, `PROJECT_LOG.md`

Create three empty files:

```bash
touch PROJECT_STATE.md EXPERIMENTS.md PROJECT_LOG.md
```

Let Claude populate them in the next phase. Commit everything:

```bash
git add .
git commit -m "add CLAUDE.md and project tracking files"
git push
```

---

## Phase 5 — The kickoff prompt

In Claude Code, paste:

```
Read CLAUDE.md fully, then scan refs/ for the assignment spec, tutorial,
report template, and Assignment 1 materials.

Do NOT write any code yet. Stay in planning mode.

Produce:
1. A requirements summary from the Homework 2 spec.
2. A clarifications summary from the instructor/forum notes.
3. A reuse-vs-redesign analysis of Assignment 1 materials.
4. A proposed folder structure for src/ and notebooks/.
5. A caching/checkpoint plan and an anti-rerun plan.
6. A description of our collaboration process going forward.
7. An initial PROJECT_STATE.md with current phase, open questions, and the next recommended action.

Ask me any clarifying questions before finalizing.
```

If Claude starts writing notebook code anyway, interrupt:

```
Stop. Stay in planning mode until we've agreed on architecture.
```

---

## Phase 6 — The smallest working slice

Once the plan is approved:

```
Now implement only the smallest working vertical slice.

I want:
- data loading
- preprocessing
- one tokenization path
- one model path (DistilBERT — cheapest)
- one tiny training run (smoke mode, 1 epoch, tiny subset)
- one validation pass
- saved outputs
- one row appended to EXPERIMENTS.md

Do not build the full notebook yet.
After implementing, tell me:
- exactly what command I should run
- what outputs I should expect
- what failures are most likely and how to recognize them
```

Run it. Paste back whatever happens — errors, logs, metrics, screenshots. Don't edit — let Claude interpret.

Commit when it works:

```bash
git add . && git commit -m "smoke slice working on distilbert" && git push
```

---

## Phase 7 — Your job during the project

You do **not** orchestrate manually. Your role is small:

- Run what Claude tells you to run
- Paste errors, logs, metrics, confusion matrices, curves, observations
- Answer Claude's questions when it needs real-world evidence
- Reject directions that feel too expensive or messy
- Commit to git at milestones

Claude's job is to:

- Keep phase/state straight
- Interpret results
- Recommend next steps
- Update `PROJECT_STATE.md` and `EXPERIMENTS.md`
- Keep report implications visible
- Prevent chaos

---

## Phase 8 — Expand the harness gradually

```
Expand the system incrementally. Add:
- smoke/dev/full modes
- model swapping for the three required models
- experiment config handling
- checkpoint saving/loading
- results logging into EXPERIMENTS.md
- validation predictions saving
- subgroup analysis helpers
- error analysis helpers

Keep it modular. After each meaningful step:
- update PROJECT_STATE.md
- append to EXPERIMENTS.md if a run happened
- summarize what changed and what I can now test
- recommend the next best step
- note what report evidence this unlocks
```

---

## Phase 9 — The experiment ladder

**Smoke** — tiny subset, 1 epoch. Only tests that the pipeline works.

**Dev** — fixed validation split, DistilBERT first. Try: input format variations, max_length, learning rate, batch size, scheduler, class weighting, regularization.

**Confirm** — full data, 2–3 seeds, best handful of settings only. Promote to stronger models once ideas have shown signal.

**Final** — clean packaging into the three required Kaggle notebooks.

Don't skip the ladder. The spec rewards systematic comparison, not final score dumping.

---

## Phase 10 — Context hygiene

When sessions get long, run `/compact` to summarize. Use it **proactively**, at phase transitions, not just when context fills up.

If `PROJECT_STATE.md` grows past ~300 lines, tell Claude:

```
Compress PROJECT_STATE.md — move stale completed work into PROJECT_LOG.md and keep only currently-relevant content in PROJECT_STATE.md.
```

Between sessions, resume with:

```
/continue
```

Or pick a specific past session with `/resume`.

---

## Phase 11 — If Claude drifts

If it stops interpreting results or skips the project files, remind it:

```
Stay in active guide mode. Update PROJECT_STATE.md, append to
EXPERIMENTS.md, interpret the result, recommend the next step, and
tell me the report implications by default.
```

---

## Phase 12 — Assignment 1 as insight only

Lean on Assignment 1 for:

- task behavior insight (class imbalance, macro-F1 importance)
- the question/answer length mismatch
- the hard boundary between Ambivalent and Clear Reply
- the question–answer interaction mattering more than generic text quality
- report comparison

Do **not** copy its notebook structure.

---

## Phase 13 — Package the Kaggle notebooks

Only after the pipeline and settings are stable. For each of BERT, DistilBERT, DeBERTa:

```
Package the final Kaggle notebook for [MODEL] using the validated
project structure and confirmed settings.

Keep it clean and readable. No architectural changes now.
Produce the required submission file and make sure it's suitable for TA inspection.
```

Commit each one separately.

---

## Phase 14 — Assemble the final report

```
Assemble the final report from validated results and accumulated project state.

Extract and use refs/AI2_Template.zip as the base LaTeX structure.
Do not replace the template — fill it in.
Output each section as LaTeX source ready to compile.
Use booktabs for tables and proper \label/\ref throughout.

Use the template as a scaffold, not a prison.
Optimize for Homework 2 requirements first.

I want:
- final section outline
- polished Greek text for each section, in natural student voice
- model-by-model comparison
- subgroup analysis
- error analysis
- modifications tried and their effects
- comparison with Assignment 1
- table/figure placement suggestions
- final checklist of missing items

Draw evidence from EXPERIMENTS.md, PROJECT_STATE.md, and PROJECT_LOG.md.
Do not invent results that aren't documented.
```

---

## The one principle

**Do not ask Claude to build the final notebooks first. Build the experiment system, let Claude guide you through the project, package the final deliverables at the end.**

---

## Quick reference — the full order

1. Gather reference materials into `refs/`
2. Create project directory, git init, push to GitHub
3. Launch `claude`, run `/init`
4. Write `CLAUDE.md` (overwriting the one from `/init`)
5. Create `PROJECT_STATE.md`, `EXPERIMENTS.md`, `PROJECT_LOG.md`
6. Send the kickoff prompt — force planning first
7. Approve architecture and anti-rerun design
8. Ask for the smallest smoke slice
9. Run it, paste results, let Claude interpret
10. Expand the harness incrementally
11. Run dev experiments on DistilBERT first
12. Promote to confirm runs with 2–3 seeds
13. Package the three Kaggle notebooks
14. Assemble the final report
