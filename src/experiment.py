import json
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup

from .config import ExperimentConfig
from .data import (
    EVASION_ID_TO_CLARITY_ID,
    NUM_CLARITY,
    NUM_EVASION,
    clean_data,
    create_split,
    encode_labels,
    load_clarity,
    smoke_subset,
)
from sklearn.utils.class_weight import compute_class_weight

from .evaluate import evaluate
from .features import (
    fit_numeric_feature_transform,
    save_feature_stats,
    transform_numeric_features,
)
from .features_head import load_dual_view_feature_model_and_tokenizer, load_feature_model_and_tokenizer
from .evasion_stack import fit_evasion_stack_features, save_evasion_stack_model
from .model import load_model_and_tokenizer
from .multitask import load_multitask_model
from .tokenization import (
    ENGINEERED_CUE_TOKENS,
    NEGATION_SPECIAL_TOKEN,
    QUESTION_SPECIAL_TOKENS,
    apply_engineered_cue_tokens,
    apply_evidence_word_prefix,
    apply_focused_answer_view,
    apply_negation_markers,
    apply_question_prefix,
    apply_surface_word_normalization,
    ensure_special_tokens,
    tokenize_pairs,
    tokenize_dual_view_pairs_with_features,
    tokenize_pairs_with_features,
    tokenize_pairs_multitask,
)
from .train import set_seed, train_one_epoch


def special_tokens_for_config(cfg: ExperimentConfig) -> list:
    tokens = []
    if cfg.use_question_tokens:
        tokens.extend(QUESTION_SPECIAL_TOKENS)
    if cfg.use_negation_markers:
        tokens.append(NEGATION_SPECIAL_TOKEN)
    if cfg.use_engineered_cue_tokens:
        tokens.extend(ENGINEERED_CUE_TOKENS)
    return tokens


def apply_text_transforms_for_config(df, cfg: ExperimentConfig, answer_view: str = None):
    """Apply the same deterministic text transforms used during training."""
    answer_view = answer_view or cfg.answer_view
    out = df.copy()
    if cfg.use_question_tokens:
        out = apply_question_prefix(out)
    if cfg.use_negation_markers:
        out = apply_negation_markers(out)
    if cfg.use_engineered_cue_tokens:
        out = apply_engineered_cue_tokens(out)
    if cfg.use_surface_word_normalization:
        out = apply_surface_word_normalization(out)
    if answer_view == "focused":
        out = apply_focused_answer_view(out)
    if cfg.use_evidence_word_prefix:
        out = apply_evidence_word_prefix(out)
    return out


def run_experiment(cfg: ExperimentConfig):
    """Επιστρέφει: (history, val_logits, val_labels, test_df, tokenizer, model, device).
    Το Cell 4 του notebook χρησιμοποιεί αυτό το 7-tuple για submission generation
    και GPU cleanup μεταξύ runs.
    """
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device: {device}")

    train_df, test_df = load_clarity(cfg.data_dir)
    cleaned = clean_data(train_df)
    cleaned = encode_labels(cleaned)
    print(f"[info] cleaned: {len(cleaned)} rows (original: {len(train_df)})")

    if cfg.mode == "smoke":
        cleaned = smoke_subset(cleaned, n_per_class=cfg.smoke_n, seed=cfg.seed)
        print(f"[info] smoke subset: {len(cleaned)} rows")

    train_idx, val_idx = create_split(cleaned, val_size=0.1, seed=cfg.split_id)
    train_df_split = cleaned.iloc[train_idx].reset_index(drop=True)
    val_df_split = cleaned.iloc[val_idx].reset_index(drop=True)
    print(f"[info] train: {len(train_df_split)}, val: {len(val_df_split)}")

    # Multi-task και evasion-target είναι mutually exclusive
    multitask = cfg.aux_evasion_weight > 0
    if multitask and cfg.train_target == "evasion":
        raise ValueError(
            "aux_evasion_weight>0 και train_target='evasion' είναι incompatible - "
            "το multi-task έχει ήδη evasion ως αυξιλιαρικό. Διάλεξε ένα."
        )

    # Train target: clarity (3-class) ή evasion (9-class, collapse σε 3 στο eval)
    if cfg.train_target == "evasion":
        num_labels = NUM_EVASION
        label_col = "evasion_id"
        evasion_to_clarity = np.array(
            [EVASION_ID_TO_CLARITY_ID[i] for i in range(NUM_EVASION)], dtype=np.int64
        )
        print(f"[info] train_target=evasion -> {NUM_EVASION}-class training, eval collapsed to 3 clarity classes")
    else:
        num_labels = NUM_CLARITY
        label_col = "label_id"
        evasion_to_clarity = None

    # Class weights (only on clarity target; για evasion δεν τα υπολογίζω στο 9-class)
    class_weights = None
    if cfg.use_class_weights and cfg.train_target == "clarity":
        weights = compute_class_weight(
            "balanced",
            classes=np.array([0, 1, 2]),
            y=train_df_split["label_id"].values,
        )
        class_weights = torch.tensor(weights, dtype=torch.float).to(device)
        print(
            f"[info] class weights: CR={weights[0]:.3f}  AMB={weights[1]:.3f}  CNR={weights[2]:.3f}"
        )

    if cfg.use_numeric_features and cfg.train_target != "clarity":
        raise ValueError("use_numeric_features currently supports train_target='clarity' only.")
    if cfg.use_numeric_features and multitask:
        raise ValueError("use_numeric_features and aux_evasion_weight are not combined in this branch.")
    if cfg.use_dual_view and not cfg.use_numeric_features:
        raise ValueError("use_dual_view currently requires use_numeric_features=True.")

    train_features = val_features = feature_stats = evasion_stack_model = None
    if cfg.use_numeric_features:
        train_features, feature_stats = fit_numeric_feature_transform(
            train_df_split, profile=cfg.numeric_feature_profile
        )
        val_features = transform_numeric_features(val_df_split, feature_stats)
        if cfg.use_evasion_proba_features:
            print("[info] fitting evasion-probability stack features (OOF train + final val)")
            train_ev, val_ev, evasion_stack_model = fit_evasion_stack_features(
                train_df_split, val_df_split, seed=cfg.seed
            )
            train_features = np.concatenate([train_features, train_ev], axis=1)
            val_features = np.concatenate([val_features, val_ev], axis=1)
            feature_stats["evasion_proba_features"] = True
            feature_stats["feature_names"] = feature_stats["feature_names"] + [
                f"evprob_{i}" for i in range(train_ev.shape[1])
            ]
        print(
            f"[info] numeric features: {train_features.shape[1]} dims, "
            f"side_hidden={cfg.numeric_feature_hidden}"
        )

    if cfg.use_dual_view:
        model, tokenizer = load_dual_view_feature_model_and_tokenizer(
            cfg.model_name,
            num_labels=num_labels,
            num_features=train_features.shape[1],
            feature_hidden=cfg.numeric_feature_hidden,
        )
    elif cfg.use_numeric_features:
        model, tokenizer = load_feature_model_and_tokenizer(
            cfg.model_name,
            num_labels=num_labels,
            num_features=train_features.shape[1],
            feature_hidden=cfg.numeric_feature_hidden,
        )
    elif multitask:
        print(f"[info] multi-task mode: clarity head (3) + evasion head (9), aux_weight={cfg.aux_evasion_weight}")
        model, tokenizer = load_multitask_model(
            cfg.model_name, num_clarity=NUM_CLARITY, num_evasion=NUM_EVASION
        )
    else:
        model, tokenizer = load_model_and_tokenizer(cfg.model_name, num_labels=num_labels)
    model.to(device)

    extra_tokens = special_tokens_for_config(cfg)
    if extra_tokens:
        added = ensure_special_tokens(extra_tokens, tokenizer, model)
        print(f"[info] added {added} special tokens: {extra_tokens}")

    if cfg.use_negation_markers:
        print("[info] applying spaCy negation markers to answers (train + val)")
    if cfg.use_engineered_cue_tokens:
        print("[info] applying engineered cue tokens to answers (train + val)")
    if cfg.use_surface_word_normalization:
        print("[info] applying aggressive real-word surface normalization (train + val)")
    if cfg.answer_view == "focused":
        print("[info] applying focused answer view: opening + overlap sentences + closing")
    if cfg.use_dual_view:
        print("[info] dual-view mode: full Q/A + focused Q/A through shared backbone")
    if cfg.use_evidence_word_prefix:
        print("[info] applying natural-language evidence/style prefix (train + val)")
    train_df_split = apply_text_transforms_for_config(train_df_split, cfg)
    val_df_split = apply_text_transforms_for_config(val_df_split, cfg)

    if cfg.use_dual_view:
        train_df_second = apply_text_transforms_for_config(
            cleaned.iloc[train_idx].reset_index(drop=True), cfg, answer_view="focused"
        )
        val_df_second = apply_text_transforms_for_config(
            cleaned.iloc[val_idx].reset_index(drop=True), cfg, answer_view="focused"
        )
        train_ds = tokenize_dual_view_pairs_with_features(
            train_df_split,
            train_df_second,
            tokenizer,
            cfg.max_length,
            cfg.input_fmt,
            train_features,
            label_col=label_col,
        )
        val_ds = tokenize_dual_view_pairs_with_features(
            val_df_split,
            val_df_second,
            tokenizer,
            cfg.max_length,
            cfg.input_fmt,
            val_features,
            label_col=label_col,
        )
    elif cfg.use_numeric_features:
        train_ds = tokenize_pairs_with_features(
            train_df_split,
            tokenizer,
            cfg.max_length,
            cfg.input_fmt,
            train_features,
            label_col=label_col,
        )
        val_ds = tokenize_pairs_with_features(
            val_df_split,
            tokenizer,
            cfg.max_length,
            cfg.input_fmt,
            val_features,
            label_col=label_col,
        )
    elif multitask:
        train_ds = tokenize_pairs_multitask(
            train_df_split, tokenizer, cfg.max_length, cfg.input_fmt
        )
        val_ds = tokenize_pairs_multitask(
            val_df_split, tokenizer, cfg.max_length, cfg.input_fmt
        )
    else:
        train_ds = tokenize_pairs(
            train_df_split, tokenizer, cfg.max_length, cfg.input_fmt, label_col=label_col
        )
        val_ds = tokenize_pairs(
            val_df_split, tokenizer, cfg.max_length, cfg.input_fmt, label_col=label_col
        )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, sampler=RandomSampler(train_ds)
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, sampler=SequentialSampler(val_ds)
    )

    # DeBERTa-v3: give classifier+pooler head a 10x higher lr than the backbone.
    # The randomly-init head needs stronger signal; backbone gradients get clipped too
    # aggressively at uniform lr because DeBERTa's disentangled attention produces
    # large grad norms (max ~10) which clip=1.0 reduces 10x.
    if "deberta" in cfg.model_name.lower():
        if cfg.use_dual_view:
            head_params = (
                list(model.feature_mlp.parameters())
                + list(model.pre_classifier.parameters())
                + list(model.classifier.parameters())
            )
            if hasattr(model.backbone, "pooler") and model.backbone.pooler is not None:
                head_params += list(model.backbone.pooler.parameters())
        elif cfg.use_numeric_features:
            head_params = (
                list(model.feature_mlp.parameters())
                + list(model.pre_classifier.parameters())
                + list(model.classifier.parameters())
            )
            if hasattr(model.backbone, "pooler") and model.backbone.pooler is not None:
                head_params += list(model.backbone.pooler.parameters())
        elif multitask:
            head_params = (
                list(model.pre_classifier.parameters())
                + list(model.clarity_head.parameters())
                + list(model.evasion_head.parameters())
            )
        else:
            head_params = list(model.pooler.parameters()) + list(model.classifier.parameters())
        head_ids = {id(p) for p in head_params}
        backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
        optimizer = AdamW([
            {"params": backbone_params, "lr": cfg.lr},
            {"params": head_params,     "lr": cfg.lr * 10},
        ], eps=1e-6, weight_decay=cfg.weight_decay)
    else:
        optimizer = AdamW(
            model.parameters(), lr=cfg.lr, eps=1e-6, weight_decay=cfg.weight_decay
        )
    total_steps = len(train_loader) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=total_steps,
    )

    run_dir = Path(cfg.models_dir) / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if feature_stats is not None:
        save_feature_stats(feature_stats, run_dir / "feature_stats.json")
    if evasion_stack_model is not None:
        save_evasion_stack_model(evasion_stack_model, run_dir / "evasion_stack.joblib")

    history = []
    val_logits: np.ndarray = np.array([])
    val_labels: np.ndarray = np.array([])
    best_f1 = -1.0
    for epoch in range(cfg.epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            cfg.grad_clip,
            class_weights=class_weights,
            loss_type=cfg.loss,
            focal_gamma=cfg.focal_gamma,
            label_smoothing=cfg.label_smoothing,
            aux_evasion_weight=cfg.aux_evasion_weight,
        )
        val_metrics, val_logits, val_labels = evaluate(
            model, val_loader, device, evasion_to_clarity=evasion_to_clarity
        )
        print(
            f"[epoch {epoch+1}/{cfg.epochs}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['val_loss']:.4f} "
            f"val_f1_macro={val_metrics['f1_macro']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )
        history.append({"epoch": epoch + 1, "train_loss": train_loss, **val_metrics})
        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            torch.save(model.state_dict(), run_dir / "best_model.pt")
            np.save(run_dir / "val_preds.npy", val_logits)
            np.save(run_dir / "val_labels.npy", val_labels)
            print(f"  * new best checkpoint (f1_macro={best_f1:.4f})")

    # restore best checkpoint so the returned model is ready for submission generation
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
    val_logits = np.load(run_dir / "val_preds.npy")
    val_labels = np.load(run_dir / "val_labels.npy")

    cfg.save(str(run_dir / "config.json"))
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"[info] saved run to {run_dir} (best f1_macro={best_f1:.4f})")

    return history, val_logits, val_labels, test_df, tokenizer, model, device


def append_to_experiments_md(
    cfg: ExperimentConfig, history: list, md_path: str
) -> None:
    best = max(history, key=lambda x: x["f1_macro"])
    row = (
        f"| {cfg.run_id} | {cfg.model_name} | {cfg.input_fmt} | {cfg.lr} | "
        f"{cfg.batch_size} | {cfg.epochs} | {cfg.max_length} | {cfg.seed} | "
        f"{best['f1_macro']:.4f} | {best['accuracy']:.4f} | "
        f"{cfg.mode} |\n"
    )
    placeholder = (
        "| — | — | — | — | — | — | — | — | — | — | No experiments run yet |\n"
    )
    with open(md_path, "r") as f:
        content = f.read()
    if placeholder in content:
        content = content.replace(placeholder, "")
        with open(md_path, "w") as f:
            f.write(content)
    with open(md_path, "a") as f:
        f.write(row)
