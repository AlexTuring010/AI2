"""Dev baseline: DistilBERT στο full cleaned set (3410 rows), max_length=256, 3 epochs.
Αυτό είναι το πρώτο «πραγματικό» experiment — ο στόχος είναι να δούμε ρεαλιστικό val F1
πριν αρχίσουμε hyperparam sweep και input_fmt ablation.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ExperimentConfig
from src.experiment import append_to_experiments_md, run_experiment


def main() -> None:
    cfg = ExperimentConfig(
        model_name="distilbert-base-uncased",
        input_fmt="two_segment",
        max_length=256,
        lr=2e-5,
        batch_size=16,
        epochs=3,
        warmup_steps=100,
        mode="dev",
        seed=42,
        data_dir=str(PROJECT_ROOT / "data"),
        models_dir=str(PROJECT_ROOT / "models"),
    )
    final_metrics = run_experiment(cfg)
    append_to_experiments_md(
        cfg, final_metrics, md_path=str(PROJECT_ROOT / "EXPERIMENTS.md")
    )
    print(f"[done] run_id: {cfg.run_id}")


if __name__ == "__main__":
    main()
