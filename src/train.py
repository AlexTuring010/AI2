import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float = 2.0,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Multiclass focal loss. Down-weights well-classified examples με factor
    (1 - p_t)^gamma. Παράγει ίδιο mean reduction με cross_entropy.
    """
    ce = F.cross_entropy(logits, labels, weight=class_weights, reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


def train_one_epoch(
    model,
    loader: DataLoader,
    optimizer,
    scheduler,
    device,
    grad_clip: float = 1.0,
    class_weights: Optional[torch.Tensor] = None,
    loss_type: str = "ce",
    focal_gamma: float = 2.0,
    aux_evasion_weight: float = 0.0,
) -> float:
    """Κάθε batch είναι (input_ids, attn, labels) ή (input_ids, attn, clarity, evasion)
    για multi-task. Το multi-task μονοπάτι ενεργοποιείται μόνο όταν
    len(batch)==4 και aux_evasion_weight>0 - τότε `model` πρέπει να είναι
    MultiTaskClassifier που δέχεται evasion_labels.
    """
    model.train()
    use_ce_weight = class_weights is not None
    total_loss = 0.0
    for batch in loader:
        batch = [b.to(device) for b in batch]
        optimizer.zero_grad()

        if len(batch) == 4 and aux_evasion_weight > 0:
            input_ids, attention_mask, labels, evasion_labels = batch
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                evasion_labels=evasion_labels,
                aux_weight=aux_evasion_weight,
            )
            loss = out.loss
        else:
            input_ids, attention_mask, labels = batch[:3]
            if loss_type == "focal":
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = focal_loss(
                    out.logits, labels, gamma=focal_gamma, class_weights=class_weights
                )
            elif use_ce_weight:
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = F.cross_entropy(out.logits, labels, weight=class_weights)
            else:
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)
