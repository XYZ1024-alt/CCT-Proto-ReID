"""Stage-1 and Stage-2 loss wiring for T2C-CLIP."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch.nn import functional as F

from t2c_clip.losses import batch_hard_triplet_loss, bidirectional_contrastive_loss
from t2c_clip.model import T2CClipModel
from t2c_clip.tfc import TFCCenterBank

DEFAULT_LOGIT_SCALE = 1.0
DEFAULT_TRIPLET_MARGIN = 0.3
DEFAULT_TFC_WEIGHT = 1.0


@dataclass(frozen=True)
class TrainingBatch:
    images: torch.Tensor
    camera_ids: torch.Tensor
    person_ids: torch.Tensor


@dataclass(frozen=True)
class Stage1LossConfig:
    logit_scale: float = DEFAULT_LOGIT_SCALE


@dataclass(frozen=True)
class Stage2LossConfig:
    logit_scale: float = DEFAULT_LOGIT_SCALE
    triplet_margin: float = DEFAULT_TRIPLET_MARGIN
    tfc_weight: float = DEFAULT_TFC_WEIGHT


@dataclass(frozen=True)
class Stage2LossInputs:
    classifier: torch.nn.Module
    tfc_bank: TFCCenterBank
    config: Stage2LossConfig = field(default_factory=Stage2LossConfig)


@dataclass(frozen=True)
class Stage2LossBreakdown:
    clip_dual: torch.Tensor
    identity: torch.Tensor
    triplet: torch.Tensor
    tfc: torch.Tensor
    tfc_weight: float

    @property
    def total(self) -> torch.Tensor:
        return self.clip_dual + self.identity + self.triplet + self.tfc_weight * self.tfc


def stage1_alignment_loss(
    model: T2CClipModel,
    batch: TrainingBatch,
    config: Stage1LossConfig,
) -> torch.Tensor:
    visual = model.encode_visual(batch.images)
    prompts = model.prompt_bank.training_prompts(batch.camera_ids, batch.person_ids)
    text = model.encode_text(prompts)
    return bidirectional_contrastive_loss(visual, text, logit_scale=config.logit_scale)


def stage2_loss_breakdown(
    model: T2CClipModel,
    batch: TrainingBatch,
    inputs: Stage2LossInputs,
) -> Stage2LossBreakdown:
    outputs = model.forward_training(batch.images, batch.camera_ids, batch.person_ids)
    retrieval = outputs["retrieval"]
    logits = inputs.classifier(retrieval)
    return Stage2LossBreakdown(
        clip_dual=bidirectional_contrastive_loss(outputs["visual"], outputs["text"], inputs.config.logit_scale),
        identity=F.cross_entropy(logits, batch.person_ids),
        triplet=batch_hard_triplet_loss(retrieval, batch.person_ids, inputs.config.triplet_margin),
        tfc=inputs.tfc_bank.loss(retrieval, batch.person_ids),
        tfc_weight=inputs.config.tfc_weight,
    )
