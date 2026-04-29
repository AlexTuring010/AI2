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
    # Αν True, προσθέτω [AFFIRM_Q] και [MULTI_Q] special tokens στην αρχή του
    # question string, based on dataset booleans. Ο model καλεί resize_token_embeddings.
    use_question_tokens: bool = False
    # Αν True, προσθέτω [NEG] token πριν από negated verbs στα answers (spaCy parse).
    use_negation_markers: bool = False

    # --- training ---
    lr: float = 2e-5
    batch_size: int = 32
    epochs: int = 3
    warmup_steps: int = 0
    grad_clip: float = 1.0

    # --- class imbalance ---
    use_class_weights: bool = False

    # --- loss ---
    # "ce" = standard cross-entropy
    # "focal" = focal loss με focusing parameter gamma
    loss: Literal["ce", "focal"] = "ce"
    focal_gamma: float = 2.0

    # --- training target ---
    # "clarity" = κλασικό 3-class training
    # "evasion" = training σε 9 classes, mapping σε 3 στο eval via taxonomy
    train_target: Literal["clarity", "evasion"] = "clarity"

    # --- multi-task auxiliary evasion loss ---
    # > 0 ενεργοποιεί dual-head multi-task model: primary=clarity (3-class),
    # auxiliary=evasion (9-class), combined loss = CE_clarity + λ * CE_evasion.
    # Mutually exclusive με train_target="evasion". 0.0 απενεργοποιεί.
    aux_evasion_weight: float = 0.0

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
        cw_tag = "_cw" if self.use_class_weights else ""
        target_tag = "" if self.train_target == "clarity" else f"_tgt{self.train_target}"
        loss_tag = "" if self.loss == "ce" else f"_focal{self.focal_gamma}"
        qtok_tag = "_qtoks" if self.use_question_tokens else ""
        neg_tag = "_neg" if self.use_negation_markers else ""
        aux_tag = f"_aux{self.aux_evasion_weight}" if self.aux_evasion_weight > 0 else ""
        self.run_id = (
            f"{short_model}_{self.input_fmt}_lr{self.lr}_bs{self.batch_size}"
            f"_ep{self.epochs}_ml{self.max_length}_seed{self.seed}"
            f"{cw_tag}{target_tag}{loss_tag}{qtok_tag}{neg_tag}{aux_tag}"
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
