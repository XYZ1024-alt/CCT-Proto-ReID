"""Loss functions used by the T2C-CLIP training design."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from t2c_clip.features import l2_normalize

DEFAULT_LOGIT_SCALE = 1.0
DEFAULT_MARGIN = 0.3


def bidirectional_contrastive_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale: float = DEFAULT_LOGIT_SCALE,
) -> torch.Tensor:
    if image_features.shape != text_features.shape:
        raise ValueError("image_features and text_features must have identical shapes")
    image = l2_normalize(image_features)
    text = l2_normalize(text_features)
    logits = logit_scale * image @ text.T
    targets = torch.arange(logits.shape[0], device=logits.device)
    image_loss = F.cross_entropy(logits, targets)
    text_loss = F.cross_entropy(logits.T, targets)
    return (image_loss + text_loss) / 2.0


def batch_hard_triplet_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    margin: float = DEFAULT_MARGIN,
) -> torch.Tensor:
    _validate_triplet_inputs(features, labels)
    distances = 1.0 - l2_normalize(features) @ l2_normalize(features).T
    losses = _valid_anchor_losses(distances, labels, margin)
    if not losses:
        raise ValueError("batch_hard_triplet_loss requires at least one valid positive and negative")
    return torch.stack(losses).mean()


def _valid_anchor_losses(distances: torch.Tensor, labels: torch.Tensor, margin: float) -> list[torch.Tensor]:
    losses: list[torch.Tensor] = []
    for index in range(labels.shape[0]):
        positive_mask = labels == labels[index]
        negative_mask = labels != labels[index]
        positive_mask[index] = False
        if bool(torch.any(positive_mask)) and bool(torch.any(negative_mask)):
            hardest_positive = torch.max(distances[index][positive_mask])
            hardest_negative = torch.min(distances[index][negative_mask])
            losses.append(F.relu(hardest_positive - hardest_negative + margin))
    return losses


def _validate_triplet_inputs(features: torch.Tensor, labels: torch.Tensor) -> None:
    if features.ndim != 2:
        raise ValueError("features must be a rank-2 tensor")
    if labels.dtype != torch.long or labels.ndim != 1:
        raise ValueError("labels must be a rank-1 torch.long tensor")
    if features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels must have the same batch size")
