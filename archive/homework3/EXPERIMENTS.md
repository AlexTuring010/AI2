# EXPERIMENTS.md

One row per experiment run. `run_id` encodes the config (see `src/config.py`, built in
Phase 2). Metrics are on the fixed dev subset unless noted. `invalid%` = share of
generations that did not parse to a valid label.

| run_id | model | strategy | prompt_variant | shots | decoding | seed | macro_F1 | accuracy | per-class F1 [CR, AMB, CNR] | invalid% | takeaway |
|--------|-------|----------|----------------|-------|----------|------|----------|----------|------------------------------|----------|----------|
| smoke-0.8b-zs-naive | Qwen3.5-0.8B | zero-shot | naive (tutorial) | 0 | greedy, no-think | — | — | 0.188 | — | 0% | smoke N=32 (HF `train` head, not the dev subset); collapses to Clear Non-Reply (24/32 preds) |
| dev-0.8b-v0-naive | Qwen3.5-0.8B | zero-shot | v0 naive | 0 | greedy, no-think | 42 | 0.209 | 0.202 | [0.244, 0.227, 0.157] | 0% | collapses to all-Clear Non-Reply (356/500) — reproduces smoke at N=500 |
| dev-0.8b-v1-defs | Qwen3.5-0.8B | zero-shot | v1 +definitions | 0 | greedy, no-think | 42 | 0.222 | 0.330 | [0.457, 0.180, 0.030] | 0% | definitions flip the collapse to all-Clear Reply (426/500) |
| dev-0.8b-v2-anticollapse | Qwen3.5-0.8B | zero-shot | v2 +anti-collapse hint | 0 | greedy, no-think | 42 | 0.239 | 0.354 | [0.457, 0.259, 0.000] | 0% | small AMB gain; CNR predicted only 3× |
| dev-0.8b-v3-rubric | Qwen3.5-0.8B | zero-shot | v3 +decision rubric | 0 | greedy, no-think | 42 | 0.291 | 0.384 | [0.443, 0.369, 0.060] | 0% | best; 206/296 Ambivalent mislabelled Clear Reply; CNR recall 0.04 |
| dev-0.8b-v4-sharp | Qwen3.5-0.8B | zero-shot | v4 sharpened defs | 0 | greedy, no-think | 42 | 0.322 | 0.368 | [0.436, 0.347, 0.183] | 0% | sharpening helps, esp. CNR (F1 0.06→0.18) |
| dev-0.8b-v5-fewshot | Qwen3.5-0.8B | few-shot | v4 + 6-shot (2/class) | 6 | greedy, no-think | 42 | 0.285 | 0.332 | [0.450, 0.197, 0.207] | 0% | few-shot HURTS — short auto-selected demos unrepresentative; AMB F1 collapses to 0.20 |
| dev-0.8b-v6-cot | Qwen3.5-0.8B | CoT | v4 + chain-of-thought | 0 | greedy, no-think | 42 | 0.417 | 0.490 | [0.437, 0.596, 0.217] | 0% | **breakthrough**; AMB recall 0.28→0.55; long-answer subgroup macroF1 0.17→0.41 |

### Phase 4 — model-size × strategy grid (2026-05-22, 500-row dev subset, length-sorted batching)

| run_id | model | strategy | prompt | shots | decoding | seed | macro_F1 | accuracy | per-class F1 [CR, AMB, CNR] | invalid% | takeaway |
|--------|-------|----------|--------|-------|----------|------|----------|----------|------------------------------|----------|----------|
| p4-0.8b-zero | Qwen3.5-0.8B | zero-shot | v4-sharp | 0 | greedy | 42 | 0.298 | 0.358 | [0.439, 0.337, 0.118] | 0% | CR-biased |
| p4-0.8b-few | Qwen3.5-0.8B | few-shot | v4 + 6-shot (repr.) | 6 | greedy | 42 | 0.345 | 0.402 | [0.427, 0.425, 0.184] | 0% | representative demos; modest gain over zero |
| p4-0.8b-cot | Qwen3.5-0.8B | CoT | v4-cot | 0 | greedy | 42 | 0.419 | 0.492 | [0.455, 0.589, 0.213] | 0% | CoT best at 0.8B |
| p4-2b-zero | Qwen3.5-2B | zero-shot | v4-sharp | 0 | greedy | 42 | 0.393 | 0.410 | [0.496, 0.316, 0.367] | 0% | |
| p4-2b-few | Qwen3.5-2B | few-shot | v4 + 6-shot | 6 | greedy | 42 | 0.381 | 0.402 | [0.482, 0.344, 0.317] | 0% | few-shot ≈ zero at 2B |
| p4-2b-cot | Qwen3.5-2B | CoT | v4-cot | 0 | greedy | 42 | 0.488 | 0.566 | [0.477, 0.664, 0.324] | 0% | CoT scales with model size |
| p4-4b-zero | Qwen3.5-4B | zero-shot | v4-sharp | 0 | greedy | 42 | 0.318 | 0.326 | [0.603, 0.083, 0.268] | 0% | **collapses to Clear Non-Reply** (AMB F1 0.08); worse than 2B-zero |
| p4-4b-cot | Qwen3.5-4B | CoT | v4-cot | 0 | greedy | 42 | **0.548** | 0.618 | [0.557, 0.693, 0.394] | 0% | **BEST system** |
| p4-4b-few | Qwen3.5-4B | few-shot | v4 + 6-shot | 6 | greedy | 42 | 0.414 | 0.428 | [0.555, 0.394, 0.293] | 0% | OOM → auto-backoff to batch 2 |
| p4-4b-cot-TEST | Qwen3.5-4B | CoT | v4-cot | 0 | greedy | 42 | 0.575 | 0.682 | — | 0% | best system on the 308 QEvasion **test** set → `submission.csv` → **Kaggle public LB 0.725, 2nd place** (weighted-F1) |

Note: Phase 4 uses length-sorted batching; "same" configs differ ~0.02 macroF1 vs Phase 3
(bf16 numerical noise from batch composition). The Phase 4 grid is internally consistent.

### Phase 5 — 4B CoT prompt refinement (2026-05-22, same 500-row dev subset)

| run_id | model | strategy | prompt | shots | decoding | seed | macro_F1 | accuracy | per-class F1 [CR, AMB, CNR] | invalid% | takeaway |
|--------|-------|----------|--------|-------|----------|------|----------|----------|------------------------------|----------|----------|
| p5-4b-cot-v4 | Qwen3.5-4B | CoT | cot_v4 (reference) | 0 | greedy | 42 | 0.548 | 0.618 | [0.557, 0.693, 0.394] | 0% | reproduced Phase 4 exactly (determinism confirmed) |
| p5-4b-cot-v5a | Qwen3.5-4B | CoT | cot_v5a (sharpened defs) | 0 | greedy | 42 | 0.539 | 0.624 | [0.603, 0.688, 0.327] | 0% | CR fixed (F1 0.56→0.60); CNR hurt; macro slightly down, acc up |
| p5-4b-cot-v5b | Qwen3.5-4B | CoT | cot_v5b (+ "mistakes" block) | 0 | greedy | 42 | 0.535 | **0.634** | [0.613, 0.702, 0.289] | 0% | CR fixed best (F1 0.61); CNR over-suppressed; best acc + weighted-F1 |

Weighted-F1 (Kaggle's metric, = per-class F1 × support / N): cot_v4 0.621, cot_v5a 0.625,
cot_v5b **0.632**. Phase 5 = a precision/recall tradeoff — the refinement fixes the CR
over-suspicion but over-suppresses the minority CNR class.

### Phase 5.5 — `cot_v5c` (best-of-both) + self-consistency (2026-05-26, Qwen3.5-4B, ~10h overnight)

`cot_v5c` keeps the v5b CR clarification ("extra commentary around the requested
information is fine") but reverts the CNR definition to v4 (broader). Self-consistency =
up to 8 sampled passes (temp=0.7, top_p=0.9), majority vote, ties → greedy.

**Dev (N=500):**

| run_id | model | strategy | prompt | shots | decoding | seed | macro_F1 | accuracy | per-class F1 [CR, AMB, CNR] | invalid% | takeaway |
|--------|-------|----------|--------|-------|----------|------|----------|----------|------------------------------|----------|----------|
| p55-4b-cot-v5c | Qwen3.5-4B | CoT | cot_v5c (best-of-both) | 0 | greedy | 42 | **0.563** | 0.626 | [0.625, 0.679, 0.387] | 0% | best dev macroF1 so far; weighted-F1 0.632 (= v5b); CNR partially recovered (F1 0.29→0.39) but not back to v4 (0.394) |

**Test (N=308, gold available locally):**

| run_id | model | prompt | decoding | macro_F1 | weighted_F1 | accuracy | pred dist | takeaway |
|--------|-------|--------|----------|----------|-------------|----------|-----------|----------|
| p55-4b-cot-v5c-TEST-greedy | Qwen3.5-4B | cot_v5c | greedy | 0.560 | 0.672 | 0.662 | AMB 174 / CR 103 / CNR 31 | **worse than cot_v4** (test 0.575); dev gain didn't transfer |
| p55-4b-cot-v5c-TEST-SC-N2 | Qwen3.5-4B | cot_v5c | SC N=2 sampled | **0.581** | **0.692** | 0.685 | AMB 181 / CR 101 / CNR 26 | best of phase 5.5; still **below cot_v4 Kaggle LB 0.725 weighted-F1** |

**Self-consistency curve (test, cot_v5c, temp=0.7, top_p=0.9):**

| N | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| macroF1 | 0.553 | **0.581** | 0.561 | 0.544 | 0.550 | 0.542 | 0.547 | 0.538 |
| weightedF1 | 0.670 | **0.692** | 0.674 | 0.672 | 0.667 | 0.669 | 0.664 | 0.664 |
| acc | 0.662 | 0.685 | 0.666 | 0.666 | 0.659 | 0.662 | 0.656 | 0.656 |

Curve peaks at N=2 and **decays monotonically after** — non-monotonic, opposite to the
canonical SC pattern. Interpretation: the model is already confident at greedy; sampling
perturbs the answer rather than averaging out a noisy reasoning step. N=2 spike is
largely tie-breaking noise (with 2 votes most cases tie → fall back to greedy + a few
perturbations). **SC is not the lever for this task on this model.**

**Phase 5.5 verdict:** cot_v5c + SC does NOT beat cot_v4 on Kaggle's weighted-F1
(estimated 0.692 vs cot_v4's LB 0.725). **Final system stays = cot_v4.** Submission on
Kaggle is unchanged. Phase 5.5 stays as a documented failed experiment for the report
("modifications tried, what failed and why").
