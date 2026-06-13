from typing import Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _collapse_to_clarity(
    evasion_logits: np.ndarray, evasion_to_clarity: np.ndarray, num_clarity: int = 3
) -> np.ndarray:
    """Sum evasion softmax probs στις clarity classes που αντιστοιχούν.
    Επιστρέφει (N, num_clarity) probabilities (όχι logits).
    """
    probs = _softmax(evasion_logits, axis=-1)
    collapsed = np.zeros((probs.shape[0], num_clarity), dtype=probs.dtype)
    for e_id in range(probs.shape[1]):
        c_id = int(evasion_to_clarity[e_id])
        collapsed[:, c_id] += probs[:, e_id]
    return collapsed


@torch.no_grad()
def evaluate(
    model,
    loader: DataLoader,
    device,
    evasion_to_clarity: Optional[np.ndarray] = None,
) -> Tuple[dict, np.ndarray, np.ndarray]:
    """Evaluate. Αν δοθεί `evasion_to_clarity` (shape [NUM_EVASION]), ο model
    θεωρείται 9-class evasion classifier: κάνω collapse τα probs σε 3 clarity
    classes (sum probs) και τα labels από evasion_ids σε clarity_ids, ώστε όλες
    οι reported metrics να είναι στο 3-class clarity level (comparable με τα
    clarity-target runs). Το `val_loss` μένει το 9-class training loss.
    """
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0
    for batch in loader:
        batch = [b.to(device) for b in batch]
        if len(batch) == 6:
            input_ids, attention_mask, input_ids_2, attention_mask_2, labels, features = batch
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_ids_2=input_ids_2,
                attention_mask_2=attention_mask_2,
                features=features,
                labels=labels,
            )
        else:
            input_ids, attention_mask, labels = batch[:3]
        if len(batch) == 4 and hasattr(model, "feature_mlp"):
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                features=batch[3],
            )
        elif len(batch) != 6:
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        # out.loss μπορεί να είναι None (π.χ. multi-task forward χωρίς evasion_labels).
        if out.loss is not None:
            total_loss += out.loss.item()
        all_logits.append(out.logits.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)

    if evasion_to_clarity is not None:
        clarity_probs = _collapse_to_clarity(logits, evasion_to_clarity, num_clarity=3)
        preds = clarity_probs.argmax(axis=1)
        labels_for_metrics = evasion_to_clarity[labels_np].astype(np.int64)
        # Σώζω log-probs ως "logits" ώστε downstream code (softmax-then-use) να
        # παράγει ξανά τα ίδια clarity_probs: softmax(log_probs) == probs (αφού
        # τα probs αθροίζουν ήδη σε 1).
        logits_out = np.log(np.clip(clarity_probs, 1e-12, None))
        labels_out = labels_for_metrics
    else:
        preds = logits.argmax(axis=1)
        labels_for_metrics = labels_np
        logits_out = logits
        labels_out = labels_np

    metrics = {
        "val_loss": total_loss / max(len(loader), 1),
        "accuracy": float(accuracy_score(labels_for_metrics, preds)),
        "f1_macro": float(
            f1_score(labels_for_metrics, preds, average="macro", zero_division=0)
        ),
        "precision_macro": float(
            precision_score(labels_for_metrics, preds, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(labels_for_metrics, preds, average="macro", zero_division=0)
        ),
        "f1_per_class": f1_score(
            labels_for_metrics, preds, average=None, labels=[0, 1, 2], zero_division=0
        ).tolist(),
    }
    return metrics, logits_out, labels_out
