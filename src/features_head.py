"""Transformer classifier with a numeric-feature side branch."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer


@dataclass
class FeatureClassifierOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None


class FeatureConcatClassifier(nn.Module):
    """Backbone text encoder + numeric feature MLP + classifier head."""

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        num_features: int,
        feature_hidden: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.config.hidden_size
        self.feature_mlp = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Linear(num_features, feature_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.pre_classifier = nn.Linear(hidden + feature_hidden, hidden)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def _pool(self, outputs) -> torch.Tensor:
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None and hasattr(self.backbone, "pooler") and self.backbone.pooler is not None:
            pooled = self.backbone.pooler(outputs.last_hidden_state)
        elif pooled is None:
            pooled = outputs.last_hidden_state[:, 0]
        return pooled

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> FeatureClassifierOutput:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._pool(outputs)
        feat_repr = self.feature_mlp(features.float())
        combined = torch.cat([pooled, feat_repr], dim=1)
        x = self.pre_classifier(combined)
        x = self.activation(x)
        x = self.dropout(x)
        logits = self.classifier(x)
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return FeatureClassifierOutput(logits=logits, loss=loss)

    def resize_token_embeddings(self, new_size: int):
        return self.backbone.resize_token_embeddings(new_size)


def load_feature_model_and_tokenizer(
    model_name: str,
    num_labels: int,
    num_features: int,
    feature_hidden: int = 64,
) -> Tuple[FeatureConcatClassifier, object]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = FeatureConcatClassifier(
        model_name,
        num_labels=num_labels,
        num_features=num_features,
        feature_hidden=feature_hidden,
    )
    return model, tokenizer


class DualViewFeatureConcatClassifier(nn.Module):
    """Shared DeBERTa over full and focused Q/A views, then concat pooled states."""

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        num_features: int,
        feature_hidden: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.config = self.backbone.config
        hidden = self.config.hidden_size
        self.feature_mlp = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Linear(num_features, feature_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.pre_classifier = nn.Linear(hidden * 2 + feature_hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def _pooled(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        mask = attention_mask.unsqueeze(-1).float()
        return (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        input_ids_2: torch.Tensor,
        attention_mask_2: torch.Tensor,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> FeatureClassifierOutput:
        pooled_a = self._pooled(input_ids, attention_mask)
        pooled_b = self._pooled(input_ids_2, attention_mask_2)
        feat = self.feature_mlp(features.float())
        x = torch.cat([pooled_a, pooled_b, feat], dim=1)
        x = torch.relu(self.pre_classifier(x))
        x = self.dropout(x)
        logits = self.classifier(x)
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return FeatureClassifierOutput(loss=loss, logits=logits)

    def resize_token_embeddings(self, new_size: int):
        return self.backbone.resize_token_embeddings(new_size)


def load_dual_view_feature_model_and_tokenizer(
    model_name: str,
    num_labels: int,
    num_features: int,
    feature_hidden: int = 64,
) -> tuple:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = DualViewFeatureConcatClassifier(
        model_name,
        num_labels=num_labels,
        num_features=num_features,
        feature_hidden=feature_hidden,
    )
    return model, tokenizer
