"""Image transforms derived from CLIP processors."""

from __future__ import annotations

from typing import Any

from PIL import Image
import torch


class CLIPImageTransform:
    def __init__(self, image_processor: Any):
        self.image_processor = image_processor

    def __call__(self, image: Image.Image) -> torch.Tensor:
        encoded = self.image_processor(images=image, return_tensors="pt")
        return encoded["pixel_values"][0]
