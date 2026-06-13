# PROJECT_LOG.md

Compressed historical record. Detailed current state lives in `PROJECT_STATE.md`;
entries move here once they go stale.

## 2026-05-21 — Project start

- Homework 2 finished and archived under `archive/homework2/`.
- Homework 3 started: response clarity classification via **prompting** instruction-tuned
  Qwen3.5 models (0.8B / 2B / 4B). No fine-tuning, inference only.
- Read all HW3 references; distilled `refs/discord_insights.md` and `refs/forum_insights.md`.
- 8-phase plan approved (`.claude/plans/plan-next-steps-workflow-memoized-nebula.md`).
- Phase 0 done: `CLAUDE.md` rewritten for HW3; `PROJECT_STATE.md` / `EXPERIMENTS.md` /
  `PROJECT_LOG.md` / `REPORT_EVIDENCE.md` created; `requirements.txt` updated; Phase 1
  smoke notebook (`notebooks/hw3_smoke.ipynb`) authored and handed to the user.

## 2026-05-21 — Phase 1 smoke done

- `hw3_smoke.ipynb` ran clean on Kaggle (2× Tesla T4). Environment install OK
  (`transformers 5.8.0.dev0` from git).
- Resolved: dataset = HF `ailsntua/QEvasion` (competition ships only the 308-row
  `sample_solution.csv`); thinking mode does not blow up for 0.8B zero-shot; timing
  ~0.5s/sample (batch 8, no OOM).
- Finding: the naive zero-shot prompt collapses to "Clear Non-Reply" (acc 0.19, N=32).
- Decided: stay Kaggle-only for now (0.8B iteration is fast and OOM-free).

## 2026-05-21 — Phase 2 done (dev harness + 0.8B baseline)

- Built `hw3_dev.ipynb`; ran 4 zero-shot prompt variants on a fixed 500-row stratified
  dev subset of QEvasion train (AMB 296 / CR 152 / CNR 52).
- Results (macroF1 / acc): v0 naive 0.209 / 0.202, v1 +definitions 0.222 / 0.330,
  v2 +anti-collapse 0.239 / 0.354, v3 +rubric 0.291 / 0.384. invalid 0% throughout.
- Diagnosis: the model conflates fluent/engaged answers with answering — labels 206/296
  Ambivalent answers as Clear Reply. CNR nearly dead (recall 0.04). Long answers fail
  worst (subgroup macroF1 0.17 vs 0.38 for short).
- Next: Phase 3 — sharpened definitions + few-shot + CoT on 0.8B.

## 2026-05-21 — Phase 3 done (sharpened defs + few-shot + CoT)

- Ran v3/v4/v5/v6 on the same 500-row dev subset, 0.8B.
- macroF1: v3 0.291, v4_sharp 0.322, v5_fewshot 0.285, v6_cot **0.417**. invalid 0%.
- CoT is the breakthrough (+0.13 over v3): AMB recall 0.28→0.55, long-answer subgroup
  macroF1 0.17→0.41. Few-shot hurt (auto-selected demos too short / unrepresentative).
  Sharpened definitions helped modestly and are the base for the CoT prompt.
- Best system now = v6_cot (CoT on sharpened definitions).
- Next: Phase 4 — scale zero-shot (v4) + CoT (v6) to 2B and 4B.

## 2026-05-22 — Phase 4 done (model-size × strategy grid)

- Subprocess approach fixed the Kaggle env gauntlet — full 9-cell grid ran clean
  (~6.2h, invalid 0% everywhere). (Env fixes: drop `datasets` lib; uninstall+reinstall
  transformers from git; run the experiment as a subprocess for a clean import.)
- macroF1 grid: 0.8B {zero .298, few .345, cot .419}, 2B {.393, .381, .488},
  4B {zero .318, few .414, cot **.548**}.
- Best = **4B + CoT** — dev macroF1 .548, **test .575 / acc .682** (QEvasion test split
  has gold labels). `submission.csv` produced.
- Findings: CoT scales with model size; 4B zero-shot collapses to CNR (worse than 2B);
  few-shot modest, never wins; 4B+CoT reasoning quality genuinely good.
- Next: Phase 5 — refine the CoT prompt (CR→AMB over-suspicion, CNR precision).

## 2026-05-22 — Kaggle leaderboard: 2nd place

- Submitted the Phase 4 4B+CoT `submission.csv` → Kaggle public leaderboard **0.725,
  2nd place**.
- 0.725 > local accuracy 0.682 > local macro-F1 0.575 → Kaggle's "F1-score" is the
  **support-weighted F1** (majority Ambivalent dominates; minority CNR barely counts).
  The report still uses macro-F1 (per the spec) as the primary metric.
- "Good place" confirmed — after Phase 5, pivot to the report.

## 2026-05-22 — Phase 5 done (4B CoT prompt refinement)

- Ran cot_v4 (ref) / cot_v5a / cot_v5b on the 500-dev subset, 4B. invalid 0%.
- Result is a **precision/recall tradeoff**: the refinement fixed the CR over-suspicion
  (CR F1 0.557→0.613) but over-suppressed the minority CNR (CNR F1 0.394→0.289).
  macroF1: v4 0.548 (best) / v5a 0.539 / v5b 0.535. acc + weighted-F1: v5b best
  (0.634 / ~0.632). cot_v4 reproduced Phase 4 exactly.
- Next: Phase 5.5 — `cot_v5c` = keep the CR clarification, revert CNR to v4's wording
  (best-of-both attempt). Last experiment, then the report.
