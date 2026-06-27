"""Injectable dual-stream T2C-CLIP model wiring.

The model has three explicit forward paths:

- ``forward_stage1``: prompt alignment with identity-aware training text.
- ``forward_stage2``: ReID training. ReID losses use the same retrieval
  feature as inference; identity-aware text is kept for CLIP alignment only.
- ``encode_retrieval``: validation and inference retrieval with either fused
  global + camera prompts or image-only features.
"""

from __future__ import annotations

import torch

from t2c_clip.features import fuse_features, l2_normalize
from t2c_clip.prompts import PromptBank
from t2c_clip.retrieval import FUSED_RETRIEVAL, IMAGE_ONLY_RETRIEVAL, require_retrieval_mode


class T2CClipModel(torch.nn.Module):
    def __init__(
        self,
        image_encoder: torch.nn.Module,
        text_encoder: torch.nn.Module,
        prompt_bank: PromptBank,
        beta: float,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.prompt_bank = prompt_bank
        self.beta = float(beta)

    def encode_retrieval(
        self,
        images: torch.Tensor,
        camera_ids: torch.Tensor,
        retrieval_mode: str = FUSED_RETRIEVAL,
    ) -> torch.Tensor:
        """Inference / validation retrieval feature."""
        visual = self.encode_visual(images)
        mode = require_retrieval_mode(retrieval_mode)
        if mode == IMAGE_ONLY_RETRIEVAL:
            return visual
        text = self.encode_inference_text(camera_ids)
        return fuse_features(visual, text, self.beta)

    def encode_visual(self, images: torch.Tensor) -> torch.Tensor:
        image_features = self.image_encoder(images)
        return l2_normalize(image_features)

    def encode_inference_text(self, camera_ids: torch.Tensor) -> torch.Tensor:
        prompts = self.prompt_bank.inference_prompts(camera_ids)
        return self.encode_text(prompts)

    def encode_training_text(self, camera_ids: torch.Tensor, person_ids: torch.Tensor) -> torch.Tensor:
        prompts = self.prompt_bank.training_prompts(camera_ids, person_ids)
        return self.encode_text(prompts)

    def forward_stage1(
        self,
        images: torch.Tensor,
        camera_ids: torch.Tensor,
        person_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Stage-1 prompt alignment forward."""
        visual = self.encode_visual(images)
        text = self.encode_training_text(camera_ids, person_ids)
        return {"visual": visual, "text": text}

    def forward_stage2(
        self,
        images: torch.Tensor,
        camera_ids: torch.Tensor,
        person_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Stage-2 ReID training forward."""
        visual = self.encode_visual(images)
        text = self.encode_training_text(camera_ids, person_ids)
        retrieval_text = self.encode_inference_text(camera_ids)
        retrieval = fuse_features(visual, retrieval_text, self.beta)
        return {"visual": visual, "text": text, "retrieval": retrieval}

    def forward_training(
        self,
        images: torch.Tensor,
        camera_ids: torch.Tensor,
        person_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Backwards-compatible alias for Stage-2 training forward."""
        return self.forward_stage2(images, camera_ids, person_ids)

    def encode_text(self, prompts: torch.Tensor) -> torch.Tensor:
        """Encode prompt embeddings through the configured text encoder."""
        text_features = self.text_encoder(prompts)
        return l2_normalize(text_features)
