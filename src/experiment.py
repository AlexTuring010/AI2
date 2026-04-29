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
from .model import load_model_and_tokenizer
from .multitask import load_multitask_model
from .tokenization import (
    NEGATION_SPECIAL_TOKEN,
    QUESTION_SPECIAL_TOKENS,
    apply_negation_markers,
    apply_question_prefix,
    ensure_special_tokens,
    tokenize_pairs,
    tokenize_pairs_multitask,
)
from .train import set_seed, train_one_epoch


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

    if multitask:
        print(f"[info] multi-task mode: clarity head (3) + evasion head (9), aux_weight={cfg.aux_evasion_weight}")
        model, tokenizer = load_multitask_model(
            cfg.model_name, num_clarity=NUM_CLARITY, num_evasion=NUM_EVASION
        )
    else:
        model, tokenizer = load_model_and_tokenizer(cfg.model_name, num_labels=num_labels)
    model.to(device)

    extra_tokens = []
    if cfg.use_question_tokens:
        extra_tokens.extend(QUESTION_SPECIAL_TOKENS)
    if cfg.use_negation_markers:
        extra_tokens.append(NEGATION_SPECIAL_TOKEN)
    if extra_tokens:
        added = ensure_special_tokens(extra_tokens, tokenizer, model)
        print(f"[info] added {added} special tokens: {extra_tokens}")

    if cfg.use_question_tokens:
        train_df_split = apply_question_prefix(train_df_split)
        val_df_split = apply_question_prefix(val_df_split)
    if cfg.use_negation_markers:
        print("[info] applying spaCy negation markers to answers (train + val)")
        train_df_split = apply_negation_markers(train_df_split)
        val_df_split = apply_negation_markers(val_df_split)

    if multitask:
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
        if multitask:
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
        ], eps=1e-6)
    else:
        optimizer = AdamW(model.parameters(), lr=cfg.lr, eps=1e-6)
    total_steps = len(train_loader) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=total_steps,
    )

    run_dir = Path(cfg.models_dir) / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

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
