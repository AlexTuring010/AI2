# EXPERIMENTS.md

| id | model | input_fmt | lr | batch | epochs | max_len | seed | val_F1_macro | val_acc | notes |
|----|-------|-----------|-----|-------|--------|---------|------|-------------|---------|-------|
| distilbert-base-uncased_two_segment_lr2e-05_bs16_ep1_ml128_seed42 | distilbert-base-uncased | two_segment | 2e-05 | 16 | 1 | 128 | 42 | 0.1754 | 0.3333 | smoke |
| distilbert-base-uncased_two_segment_lr2e-05_bs16_ep1_ml128_seed42 | distilbert-base-uncased | two_segment | 2e-05 | 16 | 1 | 128 | 42 | 0.5492 | 0.6000 | smoke |
