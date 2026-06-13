# REPORT_EVIDENCE.md

Running store of material for the final report, so it tells a **story** rather than
listing numbers. Add to it continuously — observations, surprises, decisions, concrete
error examples, and which plot/table supports which claim.

The report (Greek, student voice) is Phase 7. Sections below mirror the planned report
structure.

## Narrative beats (the story so far)

- HW3 = prompting frozen Qwen3.5 LLMs on CLARITY — no training. Διαφορετικό paradigm
  από τη HW2 (fine-tuning).
- **Phase 1 (smoke).** Το naive prompt του tutorial κατέρρευσε στο Clear Non-Reply (acc 0.19).
- **Phase 2 (dev N=500).** 4 zero-shot variants, best v3_rubric 0.291. Diagnosis: το
  μοντέλο **δεν ξεχωρίζει Ambivalent από Clear Reply** — μήκος + ευφράδεια = "απάντησε".
- **Phase 3.** Sharpened definitions (v4) + few-shot + CoT. **CoT = breakthrough**
  (0.42 στο 0.8B). Few-shot με πολύ κοντά demos απέτυχε.
- **Phase 4 — το μεγάλο grid: 3 στρατηγικές × 3 μεγέθη.** Best = **4B + CoT, dev macroF1
  0.548, test 0.575**. Το CoT κλιμακώνεται με το μέγεθος· το zero-shot στο 4B καταρρέει.

## Setup facts (for the methodology section)

- Μοντέλα: Qwen3.5-0.8B / 2B / 4B, instruction-tuned, frozen — μόνο inference.
- Δεδομένα: HF `ailsntua/QEvasion`. Train 3448 (AMB 2040 / CR 1052 / CNR 356 ≈ 59/31/10%).
  Test 308 — **το test split έχει gold labels**, οπότε μετράμε test performance τοπικά.
- Dev subset: 500, stratified, seed 42 (AMB 296 / CR 152 / CNR 52), αναπαραγώγιμο.
- Περιβάλλον: Kaggle 2× Tesla T4. invalid outputs: 0% σε ΟΛΑ τα 9+1 runs του Phase 4.
- Dummy baseline (πάντα "Ambivalent"): macroF1 0.25 / acc 0.59.

## Prompting strategies — η σύγκριση (Phase 4 grid)

macro-F1, ίδιο 500άρι dev subset:

|        | zero-shot | few-shot | CoT   |
|--------|-----------|----------|-------|
| 0.8B   | 0.298     | 0.345    | 0.419 |
| 2B     | 0.393     | 0.381    | 0.488 |
| 4B     | 0.318     | 0.414    | **0.548** |

- **CoT κερδίζει σε κάθε μέγεθος**, και είναι η μόνη στρατηγική που κλιμακώνεται καθαρά
  με το μέγεθος (0.42 → 0.49 → 0.55).
- Few-shot: μέτριο· ποτέ δεν νικάει το CoT. Στο 2B το few-shot (0.381) ≈ zero-shot (0.393).
- Zero-shot: ΔΕΝ είναι μονότονο — 4B-zero (0.318) < 2B-zero (0.393). Δες model-size.

## Model-size comparison (0.8B / 2B / 4B)

- Για το CoT, το μέγεθος βοηθάει σταθερά: το reasoning scaffold αξιοποιεί την επιπλέον
  ικανότητα του μοντέλου.
- **4B zero-shot collapse:** το 4B χωρίς CoT καταρρέει στο Clear Non-Reply (προβλέπει
  CNR 291/500, AMB μόνο 16, AMB-F1 0.08). Το μεγαλύτερο μοντέλο + το "σκεπτικιστικό"
  V4 prompt ("don't be impressed by length/fluency... did it ACTUALLY answer") + χωρίς
  βήματα reasoning → υπερ-διόρθωση. Το CoT το "σώζει" (4B-CoT 0.548).
- → Άμεση απάντηση στο ερώτημα της εκφώνησης «whether some prompting methods benefit
  small and large models differently»: **ναι, δραματικά** — το ίδιο zero-shot prompt
  είναι ΟΚ στο 2B αλλά ρίχνει το 4B σε collapse· το CoT είναι απαραίτητο για να
  αξιοποιηθεί το μεγάλο μοντέλο.

## Prompt-design choices

- v0 naive → v1 +ορισμοί → v2 +hint → v3 +rubric (Phase 2/3).
- **v4 (sharpened):** Clear Reply = δίνει τη *συγκεκριμένη* πληροφορία που ζητήθηκε·
  ρητό "judge the substance, not the style".
- **v4-cot:** ίδιοι ορισμοί + chain-of-thought 3 βημάτων (τι ζητά η ερώτηση / τι δίνει η
  απάντηση / ταιριάζουν;) πριν το `Label:`.

## Error analysis (best = 4B + CoT)

Per-class (dev): CR P=0.66 R=0.48 F1=0.56 · AMB P=0.68 R=0.71 F1=0.69 · CNR P=0.33 R=0.50 F1=0.39.

Confusion (rows gold, cols pred):
- CR→AMB **76** (κύριο λάθος): πραγματικά Clear Replies υποβαθμίζονται σε Ambivalent —
  το μοντέλο γίνεται υπερβολικά καχύποπτο όταν η απάντηση έχει και extra σχόλια.
- AMB→CNR **51**: υπερ-σκεπτικισμός — βλέπει υπεκφυγή ως πλήρη μη-εμπλοκή.
- CNR over-predicted (precision 0.33).
- **Η CoT reasoning quality είναι όντως καλή** — τα traces εντοπίζουν σωστά: "avoids a
  direct yes/no", "pivots", "addresses only half of the dual question", "no specific
  date". Παράδειγμα ([0], gold=AMB ✓): _"provides general economic optimism... explicitly
  avoids making a direct yes/no... talks around the issue"_.
- Annotation subjectivity: κάποια gold labels είναι αμφιλεγόμενα (π.χ. [332] "Current
  concerns about US-Russia" — η απάντηση όντως συζητά concerns ουσιαστικά· το μοντέλο
  το είπε Clear Reply, gold Ambivalent — βάσιμα διφορούμενο). Υπάρχει ceiling λόγω
  υποκειμενικότητας.

## Subgroup analysis (best = 4B + CoT)

- Ανά μήκος ερώτησης: short 0.607 / medium 0.535 / long 0.486 — μεγαλύτερες ερωτήσεις
  λίγο πιο δύσκολες.
- Ανά μήκος απάντησης: short 0.563 / medium 0.583 / **long 0.455** — οι μεγάλες
  απαντήσεις παραμένουν το δυσκολότερο subgroup, αλλά ΠΟΛΥ καλύτερα από Phase 2 (0.17):
  το CoT αναγκάζει το μοντέλο να διαβάσει τη μεγάλη απάντηση.

## Input formulation

- Question / Answer plain format σε όλα. Δεν χρειάστηκε αλλαγή — το CoT έλυσε το πρόβλημα
  μήκους χωρίς αλλαγή input format.

## Modifications tried (incl. what failed and why)

- Phase 3 few-shot με κοντά demos: απέτυχε (μη αντιπροσωπευτικά).
- Phase 4 few-shot με demos **αντιπροσωπευτικού μήκους** (truncated στις ~220 λέξεις):
  βελτιώθηκε αλλά **παραμένει κάτω από το CoT σε κάθε μέγεθος**. Συμπέρασμα: το task
  θέλει reasoning, όχι pattern-matching από παραδείγματα.
- **Phase 5 — refinement του 4B CoT prompt: precision/recall tradeoff.** Δύο βελτιωμένα
  prompts (v5a, v5b) στόχευσαν τα 2 λάθη του cot_v4. Αποτέλεσμα: **διόρθωσαν όντως** το
  CR over-suspicion (CR F1 0.557→0.613) αλλά **υπερ-κατέστειλαν** τη μειοψηφική CNR
  (CNR F1 0.394→0.289, γιατί το prompt είπε "CNR μόνο για πραγματική μη-εμπλοκή" → το
  μοντέλο σχεδόν δεν προβλέπει CNR). Net: macroF1 ελαφρώς κάτω (0.548→0.535), accuracy
  και weighted-F1 πάνω. Δίδαγμα: τα prompt fixes έχουν παρενέργειες σε άλλες κλάσεις —
  κλασικό precision/recall tradeoff. Phase 5.5 (`cot_v5c`) κρατά τη διόρθωση CR και
  επαναφέρει τον ορισμό CNR του v4 (best-of-both).

- **Phase 5.5 — `cot_v5c` (best-of-both) + self-consistency: ΑΠΕΤΥΧΕ.** Στο dev το v5c
  πέτυχε το intended best-of-both: macroF1 0.563 (καλύτερο από όλα τα Phase 5 prompts),
  με CR F1 0.625 (διατήρησε τη διόρθωση του v5b) και CNR F1 0.387 (επανέφερε σχεδόν
  πλήρως το v4 level). **Όμως το dev gain ΔΕΝ μεταφέρθηκε στο test**: test macroF1
  0.560 < cot_v4 test 0.575. Πιθανότατα slight dev overfitting του prompt-tuning, ή
  test-set noise — και τα δύο σύμβατα με το ότι δοκιμάσαμε τρεις παραλλαγές prompt στο
  ίδιο dev subset.

- **Self-consistency curve (N=1..8, temp=0.7, top_p=0.9, majority vote, ties→greedy):**

  | N        | 1     | 2     | 3     | 4     | 5     | 6     | 7     | 8     |
  |----------|-------|-------|-------|-------|-------|-------|-------|-------|
  | macroF1  | 0.553 | **0.581** | 0.561 | 0.544 | 0.550 | 0.542 | 0.547 | 0.538 |
  | wF1      | 0.670 | **0.692** | 0.674 | 0.672 | 0.667 | 0.669 | 0.664 | 0.664 |

  Η καμπύλη κορυφώνεται στο N=2 και **φθίνει μονοτονικά μετά** — αντίθετο από το
  canonical SC pattern (monotonic με diminishing returns). Ερμηνεία: το μοντέλο είναι
  ήδη confident στο greedy· το sampling δεν "averages out" thinking noise — απλώς
  perturbs την απάντηση. Στο N=2 πολλά ties → πέφτουν σε greedy, οπότε το "spike" είναι
  κυρίως tie-breaking noise. **SC δεν είναι το lever για αυτό το task σε αυτό το μοντέλο.**
  Είναι ένα από τα πιο χρήσιμα διδάγματα της εργασίας: η self-consistency, παρότι "the
  standard CoT booster" στη βιβλιογραφία, δεν δουλεύει όταν το base model είναι ήδη
  decisive· κάτω από τέτοιες συνθήκες, more samples = more noise, not more signal.

- **Phase 5.5 verdict:** Final system παραμένει το **cot_v4** (Kaggle LB 0.725, 2η θέση).
  Το cot_v5c+SC test weighted-F1 (0.692) είναι κάτω από το cot_v4 LB → no submission swap.

## Comparison with Assignment 1 & 2

- HW2 fine-tuned DeBERTa: 0.70 Kaggle macroF1. HW3 prompting Qwen3.5-4B + CoT: 0.575
  test macroF1. Το prompting ενός frozen 4B φτάνει ~80% του fine-tuned ceiling.
- **Kaggle public leaderboard: 0.725, 2η θέση** (το submission του 4B+CoT). Η μετρική
  του Kaggle είναι weighted-F1 (> accuracy 0.682 > macro-F1 0.575) — η μειοψηφική CNR
  μετράει ελάχιστα. Το report χρησιμοποιεί macro-F1 ως κύρια μετρική (όπως λέει η εκφώνηση).
- Discord: όλοι ~0.27–0.42 — το 4B+CoT (0.55–0.58) είναι σαφώς πάνω από το πακέτο.
- Failure modes: HW2 (fine-tuned) και HW3 (prompted) — και τα δύο δυσκολεύονται στο
  AMB/CR boundary και στη μειοψηφική CNR (να αναπτυχθεί στο report με τα confusion matrices).

## Reproducibility note

Greedy decoding (deterministic). ΟΜΩΣ το batched LLM inference με left-padding έχει μικρή
μη-ντετερμινιστικότητα: η σύνθεση του batch (π.χ. length-sorted batching στο Phase 4 vs
unsorted στο Phase 3) αλλάζει το padding → bf16 numerical noise → ~0.02 macroF1 διαφορά
για "ίδιο" config. Κάθε grid είναι εσωτερικά συνεπές (ίδια methodology σε όλα τα cells).

## Plots / tables to produce

- Heatmap / grouped bar: macroF1 του 3×3 grid (model size × strategy) — η κεντρική εικόνα.
- Line: CoT macroF1 vs model size (0.42 → 0.49 → 0.55) — δείχνει το scaling.
- Confusion matrices: 4B-zero (collapse) vs 4B-cot — δείχνει τι λύνει το CoT.
- Bar: subgroup macroF1 ανά answer-length, Phase 2 v3 (long 0.17) vs Phase 4 4B-cot (0.46).
- Comparison table HW1/HW2/HW3.

## Surprises & decisions log

- 2026-05-21: Competition δίνει μόνο `sample_solution.csv`· δεδομένα = HF QEvasion.
- 2026-05-21: CoT είναι το breakthrough lever· few-shot με κακά demos χειροτερεύει.
- 2026-05-22: **CoT κλιμακώνεται με το μέγεθος**· **4B zero-shot καταρρέει** (χειρότερο
  από 2B) — ισχυρή size×strategy αλληλεπίδραση.
- 2026-05-22: Best system = 4B + CoT (dev 0.548 / test 0.575). `submission.csv` έτοιμο.
- 2026-05-22: Το QEvasion test split έχει gold labels — μετράμε test τοπικά.
- 2026-05-22: Submission 4B+CoT → **Kaggle public LB 0.725, 2η θέση**. Επιβεβαιώθηκε
  ότι το Id mapping (QEvasion test order) είναι σωστό (αλλιώς θα είχαμε ~random score).
- 2026-05-26: Phase 5.5 (`cot_v5c` + SC, ~10h overnight σε 4B). cot_v5c πέτυχε το
  best-of-both στο dev (macroF1 0.563 — best dev) αλλά **έχασε στο test** (0.560 < 0.575).
  **Self-consistency curve N=1..8 = NON-MONOTONIC** (peak στο N=2 = 0.581, μετά decay
  στο 0.538) → SC δεν δουλεύει εδώ· το μοντέλο είναι ήδη confident. Final system
  παραμένει **cot_v4**. Phase 5.5 = clean negative result για το report (failed prompt
  refinement + failed SC, και τα δύο με ξεκάθαρη ερμηνεία).
