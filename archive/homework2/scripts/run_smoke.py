"""Smoke test: DistilBERT, 50 samples/class, 1 epoch. Verifies the full pipeline end-to-end."""
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
        max_length=128,
        lr=2e-5,
        batch_size=16,
        epochs=1,
        mode="smoke",
        smoke_n=50,
        seed=42,
        data_dir=str(PROJECT_ROOT / "data"),
        models_dir=str(PROJECT_ROOT / "models"),
    )
    history, *_ = run_experiment(cfg)
    append_to_experiments_md(
        cfg, history, md_path=str(PROJECT_ROOT / "EXPERIMENTS.md")
    )
    print(f"[done] run_id: {cfg.run_id}")


if __name__ == "__main__":
    main()
