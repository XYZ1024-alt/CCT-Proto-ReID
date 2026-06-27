"""Injectable dual-stream T2C-CLIP model wiring.

The model has three explicit forward paths:

- :meth:`T2CClipModel.forward_stage1` — Stage-1 prompt alignment. Only the
  CLIP image feature and the training identity prompt's text feature are
  returned; no ReID/TFC gradients flow here.
- :meth:`T2CClipModel.forward_stage2` — Stage-2 ReID training. Build the
  retrieval feature ``f = normalize(f_v + beta * f_t_train)`` using training
  prompts (global + camera + identity).
- :meth:`T2CClipModel.encode_retrieval` — Inference / validation. Build the
  retrieval feature ``f = normalize(f_v + beta * f_t_eval)`` using inference
  prompts (global + camera) only.

Identity prompts are never used in :meth:`encode_retrieval`.
"""

from __future__ import annotations

import torch

from t2c_clip.features import fuse_features, l2_normalize
from t2c_clip.prompts import PromptBank


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

    def encode_retrieval(self, images: torch.Tensor, camera_ids: torch.Tensor) -> torch.Tensor:
        """Inference / validation retrieval feature.

        Uses ``global + camera`` prompts only — never identity prompts.
        """
        visual = self.encode_visual(images)
        prompts = self.prompt_bank.inference_prompts(camera_ids)
        text = self.encode_text(prompts)
        return fuse_features(visual, text, self.beta)

    def encode_visual(self, images: torch.Tensor) -> torch.Tensor:
        image_features = self.image_encoder(images)
        return l2_normalize(image_features)

    def forward_stage1(
        self,
        images: torch.Tensor,
        camera_ids: torch.Tensor,
        person_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Stage-1 prompt alignment forward.

        Returns normalized visual and identity-aware text features. Stage-1
        computes no ReID/Triplet/TFC; those are Stage-2 only.
        """
        visual = self.encode_visual(images)
        prompts = self.prompt_bank.training_prompts(camera_ids, person_ids)
        text = self.encode_text(prompts)
        return {"visual": visual, "text": text}

    def forward_stage2(
        self,
        images: torch.Tensor,
        camera_ids: torch.Tensor,
        person_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Stage-2 ReID training forward.

        Uses training prompts (global + camera + identity). The retrieval
        feature ``f = normalize(f_v + beta * f_t_train)`` is what classifier,
        triplet, and TFC losses act on.
        """
        visual = self.encode_visual(images)
        prompts = self.prompt_bank.training_prompts(camera_ids, person_ids)
        text = self.encode_text(prompts)
        retrieval = fuse_features(visual, text, self.beta)
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
        """Encode prompts through the real CLIP text branch.

        ``prompts`` is [batch, context_length, hidden_dim] where ``hidden_dim``
        is the CLIP text token embedding dimension. The text encoder injects
        the prompts into fixed context slots and runs the CLIP text
        transformer + ``text_projection``, returning L2-normalized text
        features in the shared projection space.
        """
        text_features = self.text_encoder(prompts)
        return l2_normalize(text_features)