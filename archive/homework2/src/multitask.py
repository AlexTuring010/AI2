"""Multi-task classifier: shared backbone + (clarity 3-class, evasion 9-class) heads.

Primary task = clarity. Evasion head ενεργοποιείται μόνο στο training ως auxiliary loss,
δίνοντας στο backbone fine-grained supervision. Στο eval, χρησιμοποιείται μόνο το clarity head.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer


@dataclass
class MultiTaskOutput:
    logits: torch.Tensor  # clarity logits (primary, used by evaluate)
    evasion_logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    clarity_loss: Optional[torch.Tensor] = None
    evasion_loss: Optional[torch.Tensor] = None


class MultiTaskClassifier(nn.Module):
    """Shared backbone + two linear heads. Pool [CLS] από last_hidden_state.

    Για DeBERTa χρησιμοποιούμε τον built-in ContextPooler όταν διαθέσιμος, αλλιώς
    πέφτουμε σε [CLS] hidden state. Το resize_token_embeddings κάνει passthrough
    στο backbone.
    """

    def __init__(
        self,
        model_name: str,
        num_clarity: int = 3,
        num_evasion: int = 9,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.config.hidden_size
        # Shared pre-classifier to mimic DistilBertForSequenceClassification's head:
        # Linear(hidden, hidden) -> ReLU -> Dropout -> head.
        # Without this, the multi-task model has a thinner head than the baseline
        # and confounds any comparison against single-task DistilBert.
        self.pre_classifier = nn.Linear(hidden, hidden)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.clarity_head = nn.Linear(hidden, num_clarity)
        self.evasion_head = nn.Linear(hidden, num_evasion)

    def _pool(self, outputs) -> torch.Tensor:
        # HF models expose pooler_output for BERT/DeBERTa; DistilBERT doesn't.
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]
        return pooled

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        evasion_labels: Optional[torch.Tensor] = None,
        aux_weight: float = 0.3,
    ) -> MultiTaskOutput:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._pool(outputs)
        pooled = self.pre_classifier(pooled)
        pooled = self.activation(pooled)
        pooled = self.dropout(pooled)
        clarity_logits = self.clarity_head(pooled)
        evasion_logits = self.evasion_head(pooled)

        loss = clarity_loss = evasion_loss = None
        if labels is not None:
            clarity_loss = F.cross_entropy(clarity_logits, labels)
            loss = clarity_loss
            if evasion_labels is not None and aux_weight > 0:
                evasion_loss = F.cross_entropy(evasion_logits, evasion_labels)
                loss = clarity_loss + aux_weight * evasion_loss

        return MultiTaskOutput(
            logits=clarity_logits,
            evasion_logits=evasion_logits,
            loss=loss,
            clarity_loss=clarity_loss,
            evasion_loss=evasion_loss,
        )

    def resize_token_embeddings(self, new_size: int):
        return self.backbone.resize_token_embeddings(new_size)


def load_multitask_model(
    model_name: str,
    num_clarity: int = 3,
    num_evasion: int = 9,
    dropout: float = 0.1,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = MultiTaskClassifier(
        model_name,
        num_clarity=num_clarity,
        num_evasion=num_evasion,
        dropout=dropout,
    )
    return model, tokenizer
