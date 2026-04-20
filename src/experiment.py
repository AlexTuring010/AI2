import json
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup

from .config import ExperimentConfig
from .data import (
    NUM_LABELS,
    clean_data,
    create_split,
    encode_labels,
    load_clarity,
    smoke_subset,
)
from .evaluate import evaluate
from .model import load_model_and_tokenizer
from .tokenization import tokenize_pairs
from .train import set_seed, train_one_epoch


def run_experiment(cfg: ExperimentConfig) -> dict:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device: {device}")

    train_df, _ = load_clarity(cfg.data_dir)
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

    model, tokenizer = load_model_and_tokenizer(cfg.model_name, num_labels=NUM_LABELS)
    model.to(device)

    train_ds = tokenize_pairs(train_df_split, tokenizer, cfg.max_length, cfg.input_fmt)
    val_ds = tokenize_pairs(val_df_split, tokenizer, cfg.max_length, cfg.input_fmt)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, sampler=RandomSampler(train_ds)
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, sampler=SequentialSampler(val_ds)
    )

    optimizer = AdamW(model.parameters(), lr=cfg.lr, eps=1e-8)
    total_steps = len(train_loader) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=total_steps,
    )

    history = []
    val_logits: np.ndarray = np.array([])
    val_labels: np.ndarray = np.array([])
    for epoch in range(cfg.epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, cfg.grad_clip
        )
        val_metrics, val_logits, val_labels = evaluate(model, val_loader, device)
        print(
            f"[epoch {epoch+1}/{cfg.epochs}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['val_loss']:.4f} "
            f"val_f1_macro={val_metrics['f1_macro']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )
        history.append({"epoch": epoch + 1, "train_loss": train_loss, **val_metrics})

    run_dir = Path(cfg.models_dir) / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.save(str(run_dir / "config.json"))
    torch.save(model.state_dict(), run_dir / "best_model.pt")
    np.save(run_dir / "val_preds.npy", val_logits)
    np.save(run_dir / "val_labels.npy", val_labels)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"[info] saved run to {run_dir}")

    return history[-1]


def append_to_experiments_md(
    cfg: ExperimentConfig, final_metrics: dict, md_path: str
) -> None:
    row = (
        f"| {cfg.run_id} | {cfg.model_name} | {cfg.input_fmt} | {cfg.lr} | "
        f"{cfg.batch_size} | {cfg.epochs} | {cfg.max_length} | {cfg.seed} | "
        f"{final_metrics['f1_macro']:.4f} | {final_metrics['accuracy']:.4f} | "
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
