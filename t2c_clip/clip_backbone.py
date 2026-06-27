"""Transformers CLIP adapters for T2C-CLIP training."""

from __future__ import annotations

from typing import Any

import torch


class TransformersCLIPImageEncoder(torch.nn.Module):
    def __init__(self, clip_model: torch.nn.Module):
        super().__init__()
        self.clip_model = clip_model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        output = self.clip_model.get_image_features(pixel_values=images)
        return _image_features_tensor(output)


class PromptTextEncoder(torch.nn.Module):
    def __init__(self, prompt_embedding_dim: int, output_dim: int):
        super().__init__()
        _require_positive(prompt_embedding_dim, "prompt_embedding_dim")
        _require_positive(output_dim, "output_dim")
        self.projection = torch.nn.Linear(prompt_embedding_dim, output_dim)

    def forward(self, prompts: torch.Tensor) -> torch.Tensor:
        if prompts.ndim != 3:
            raise ValueError("prompts must have shape [batch, context_length, embedding_dim]")
        return self.projection(prompts.mean(dim=1))


def clip_projection_dim(clip_model: Any) -> int:
    projection_dim = getattr(getattr(clip_model, "config", None), "projection_dim", None)
    if not isinstance(projection_dim, int):
        raise ValueError("CLIP model config must expose integer projection_dim")
    _require_positive(projection_dim, "projection_dim")
    return projection_dim


def _image_features_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    pooler_output = getattr(output, "pooler_output", None)
    if isinstance(pooler_output, torch.Tensor):
        return pooler_output
    raise TypeError("CLIP image features must be a tensor or expose tensor pooler_output")


def _require_positive(value: int, name: str) -> None:
    if value < 1:
        raise ValueError(f"{name} must be positive")
