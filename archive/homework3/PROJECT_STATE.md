# PROJECT_STATE.md

## Current phase

**Phase 6 authored — final notebook ready for user to run on Kaggle.**
`notebooks/hw3_final.ipynb` built from `scripts/make_final_notebook.py`. Live run = 4B
+ cot_v4 end-to-end on dev (N=500) + test (N=308). Other experiments (Phase 4 grid,
Phase 5, Phase 5.5) presented as markdown tables + commented-out reproducible blocks.
→ **Phase 7 (report)** after user returns the executed notebook.

## Phase progress

- [x] Phase 0–3 — foundations, smoke, dev harness, prompt iteration
- [x] Phase 4 — model-size × strategy grid → best 4B+CoT (test 0.575 macroF1; LB 0.725, 2nd)
- [x] Phase 5 — 4B CoT prompt refinement (cot_v4 / v5a / v5b)
- [x] Phase 5.5 — `cot_v5c` + self-consistency (failed: did not beat cot_v4)
- [x] Phase 6 — assembled `notebooks/hw3_final.ipynb` (Kaggle deliverable). Awaiting run.
- [ ] Phase 7 — Report (Greek) ← **next**

Full roadmap: `C:\Users\alexg\.claude\plans\plan-next-steps-workflow-memoized-nebula.md`.

## Best system so far

**Qwen3.5-4B + CoT.** Two operating points from Phase 5:
- `cot_v4`: macroF1 0.548, acc 0.618, weighted-F1 ~0.621. Test: macroF1 0.575 / acc 0.682
  → Kaggle public LB **0.725, 2nd place**.
- `cot_v5b`: macroF1 0.535, acc 0.634, weighted-F1 ~0.632 (best on Kaggle's metric).

## What Phase 5 established (2026-05-22)

Refined the 4B CoT prompt, 3 variants on the 500-dev subset (invalid 0%):

| variant | macroF1 | acc | weighted-F1 | F1 [CR, AMB, CNR] |
|---|---|---|---|---|
| cot_v4 (ref) | **0.548** | 0.618 | 0.621 | [0.557, 0.693, 0.394] |
| cot_v5a | 0.539 | 0.624 | 0.625 | [0.603, 0.688, 0.327] |
| cot_v5b | 0.535 | **0.634** | **0.632** | [0.613, 0.702, 0.289] |

- The refinement is a **precision/recall tradeoff**: it **fixed the CR over-suspicion**
  (CR F1 0.557 → 0.613 — the intended fix worked) but **over-suppressed the minority CNR**
  (CNR F1 0.394 → 0.289). Net: macroF1 slightly down, accuracy + weighted-F1 up.
- `cot_v4` reproduced Phase 4 exactly (0.548) — confirms determinism.
- The Phase 5 `submission.csv` = cot_v4 (script picked by macroF1) = identical to the
  already-submitted file → no new leaderboard entry from Phase 5.

## Phase 5.5 results (2026-05-26, 4B, ~10h overnight on Kaggle T4)

- **cot_v5c greedy dev**: macroF1 **0.563** (best dev so far, +0.015 over cot_v4).
- **cot_v5c greedy test**: macroF1 0.560 / weightedF1 0.672 — **regression** vs cot_v4
  test 0.575 (dev gain didn't transfer).
- **cot_v5c + SC (N=1..8)**: peaks at N=2 (macroF1 0.581, weightedF1 0.692), then
  monotonically decays to N=8 (0.538). Non-monotonic curve — not the canonical SC
  pattern. Interpretation: model is already confident; sampling perturbs rather than
  averages. **SC is not the lever for this task.**
- v5c+SC's test weighted-F1 (0.692) is **below cot_v4 Kaggle LB (0.725)** → would lose
  ground on Kaggle. **Don't submit.**
- CNR remains the weak class (F1 0.387 dev). Best-of-both worked partially (CNR F1
  0.29 → 0.39, vs v4's 0.394) but didn't surpass v4.

## Next actions

1. **User:** open `notebooks/hw3_final.ipynb` στο Kaggle, Settings -> GPU + Internet On,
   Save & Run All (Commit). Διάρκεια ~2.5h. Επιστρέφει το εκτελεσμένο notebook με την
   `submission_best_prompting_system.csv` στα outputs. Reproducibility expectation:
   dev macroF1 ~0.548, test macroF1 ~0.575 (Phase 4 baseline του 4B+cot_v4). Το
   submission είναι το ίδιο file με το ήδη submitted στο Kaggle (LB 0.725).
2. **Claude:** Phase 7 — write the Greek report based on REPORT_EVIDENCE.md +
   εκτελεσμένο final notebook. Phase 5.5 γίνεται το "modifications tried, including
   what failed and why" section.
3. **User:** No new Kaggle submit needed (cot_v4 is already 2nd at 0.725). Το νέο
   `submission_best_prompting_system.csv` πρέπει να είναι byte-identical με το
   submitted (deterministic) — αν διαφέρει, μικρό bf16 numerical noise από διαφορετική
   batch composition, αλλά οι predictions πρέπει να συμφωνούν στο >99%.

## Open questions

- Kaggle's official metric name (assumed weighted-F1 based on the 0.725 LB ↔ test
  weighted-F1 alignment, but never directly confirmed from the Evaluation page).

## Report evidence status

`REPORT_EVIDENCE.md` has the full grid, CoT scaling, the 4B-zero collapse, the Phase 5
precision/recall tradeoff, error analysis, subgroup, the leaderboard result.
