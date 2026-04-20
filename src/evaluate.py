from typing import Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(model, loader: DataLoader, device) -> Tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0
    for batch in loader:
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += out.loss.item()
        all_logits.append(out.logits.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    preds = logits.argmax(axis=1)

    metrics = {
        "val_loss": total_loss / max(len(loader), 1),
        "accuracy": float(accuracy_score(labels_np, preds)),
        "f1_macro": float(f1_score(labels_np, preds, average="macro", zero_division=0)),
        "precision_macro": float(
            precision_score(labels_np, preds, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(labels_np, preds, average="macro", zero_division=0)
        ),
        "f1_per_class": f1_score(
            labels_np, preds, average=None, labels=[0, 1, 2], zero_division=0
        ).tolist(),
    }
    return metrics, logits, labels_np
