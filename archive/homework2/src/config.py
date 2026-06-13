from dataclasses import dataclass, field, asdict
from typing import Literal
import json


@dataclass
class ExperimentConfig:
    # --- model ---
    model_name: str = "distilbert-base-uncased"

    # --- input ---
    input_fmt: Literal["two_segment", "concat_sep"] = "two_segment"
    answer_view: Literal["full", "focused"] = "full"
    use_dual_view: bool = False
    max_length: int = 128
    # Αν True, προσθέτω [AFFIRM_Q] και [MULTI_Q] special tokens στην αρχή του
    # question string, based on dataset booleans. Ο model καλεί resize_token_embeddings.
    use_question_tokens: bool = False
    # Αν True, προσθέτω [NEG] token πριν από negated verbs στα answers (spaCy parse).
    use_negation_markers: bool = False
    # Αν True, προσθέτω compact cue tokens από HW1-style feature engineering:
    # question intent, answer evidence/style, and Q-A lexical overlap.
    use_engineered_cue_tokens: bool = False
    # If True, replace concrete surface forms with ordinary English words that
    # DeBERTa already knows: 1981 -> year, $3m -> money amount, 45% -> percent.
    use_surface_word_normalization: bool = False
    # If True, prepend a short natural-language evidence/style summary built
    # from regex cues. This deliberately uses normal words, not new tokens.
    use_evidence_word_prefix: bool = False
    # Αν True, κρατάω το Q/A text καθαρό και περνάω engineered numeric features
    # ως δεύτερο input branch που γίνεται concat με το pooled transformer vector.
    use_numeric_features: bool = False
    numeric_feature_hidden: int = 64
    numeric_feature_profile: Literal["base", "coverage_evidence", "long_mixed", "rich"] = "base"
    use_evasion_proba_features: bool = False

    # --- training ---
    lr: float = 2e-5
    batch_size: int = 32
    epochs: int = 3
    warmup_steps: int = 0
    grad_clip: float = 1.0
    weight_decay: float = 0.0

    # --- class imbalance ---
    use_class_weights: bool = False

    # --- loss ---
    # "ce" = standard cross-entropy
    # "focal" = focal loss με focusing parameter gamma
    loss: Literal["ce", "focal"] = "ce"
    focal_gamma: float = 2.0
    label_smoothing: float = 0.0

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
        view_tag = "" if self.answer_view == "full" else f"_{self.answer_view}"
        dual_tag = "_dualview" if self.use_dual_view else ""
        cw_tag = "_cw" if self.use_class_weights else ""
        target_tag = "" if self.train_target == "clarity" else f"_tgt{self.train_target}"
        loss_tag = "" if self.loss == "ce" else f"_focal{self.focal_gamma}"
        smooth_tag = f"_ls{self.label_smoothing}" if self.label_smoothing > 0 else ""
        wd_tag = f"_wd{self.weight_decay}" if self.weight_decay > 0 else ""
        qtok_tag = "_qtoks" if self.use_question_tokens else ""
        neg_tag = "_neg" if self.use_negation_markers else ""
        cue_tag = "_cues" if self.use_engineered_cue_tokens else ""
        norm_tag = "_wordnorm" if self.use_surface_word_normalization else ""
        evid_tag = "_evidprefix" if self.use_evidence_word_prefix else ""
        numfeat_tag = "_numfeat" if self.use_numeric_features else ""
        profile_tag = "" if self.numeric_feature_profile == "base" else f"_{self.numeric_feature_profile}"
        evprob_tag = "_evprob" if self.use_evasion_proba_features else ""
        aux_tag = f"_aux{self.aux_evasion_weight}" if self.aux_evasion_weight > 0 else ""
        self.run_id = (
            f"{short_model}_{self.input_fmt}_lr{self.lr}_bs{self.batch_size}"
            f"_ep{self.epochs}_ml{self.max_length}_seed{self.seed}"
            f"{view_tag}{dual_tag}{cw_tag}{target_tag}{loss_tag}{smooth_tag}{wd_tag}{qtok_tag}{neg_tag}"
            f"{cue_tag}{norm_tag}{evid_tag}{numfeat_tag}{profile_tag}{evprob_tag}{aux_tag}"
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
