"""Injectable dual-stream T2C-CLIP model wiring."""

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
        self.beta = beta

    def encode_retrieval(self, images: torch.Tensor, camera_ids: torch.Tensor) -> torch.Tensor:
        visual = self.encode_visual(images)
        prompts = self.prompt_bank.inference_prompts(camera_ids)
        text = self.encode_text(prompts)
        return fuse_features(visual, text, self.beta)

    def forward_training(
        self,
        images: torch.Tensor,
        camera_ids: torch.Tensor,
        person_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        visual = self.encode_visual(images)
        prompts = self.prompt_bank.training_prompts(camera_ids, person_ids)
        text = self.encode_text(prompts)
        retrieval = fuse_features(visual, text, self.beta)
        return {"visual": visual, "text": text, "retrieval": retrieval}

    def encode_visual(self, images: torch.Tensor) -> torch.Tensor:
        return l2_normalize(self.image_encoder(images))

    def encode_text(self, prompts: torch.Tensor) -> torch.Tensor:
        return l2_normalize(self.text_encoder(prompts))
