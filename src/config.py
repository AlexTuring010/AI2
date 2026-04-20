from dataclasses import dataclass, field, asdict
from typing import Literal
import json


@dataclass
class ExperimentConfig:
    # --- model ---
    model_name: str = "distilbert-base-uncased"

    # --- input ---
    input_fmt: Literal["two_segment", "concat_sep"] = "two_segment"
    max_length: int = 128

    # --- training ---
    lr: float = 2e-5
    batch_size: int = 32
    epochs: int = 3
    warmup_steps: int = 0
    grad_clip: float = 1.0

    # --- class imbalance ---
    use_class_weights: bool = False

    # --- reproducibility ---
    seed: int = 42
    split_id: int = 0

    # --- run mode ---
    mode: Literal["smoke", "dev", "confirm", "final"] = "dev"
    smoke_n: int = 50  # samples per class in smoke mode

    # --- paths ---
    data_dir: str = "data"
    models_dir: str = "models"

    # derived fields (not constructor args)
    run_id: str = field(init=False)

    def __post_init__(self):
        short_model = self.model_name.split("/")[-1]
        self.run_id = (
            f"{short_model}_{self.input_fmt}_lr{self.lr}_bs{self.batch_size}"
            f"_ep{self.epochs}_ml{self.max_length}_seed{self.seed}"
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        with open(path) as f:
            d = json.load(f)
        d.pop("run_id", None)
        return cls(**d)
