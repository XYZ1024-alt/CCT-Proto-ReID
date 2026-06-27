"""PyTorch dataset helpers for Image-to-Image ReID samples."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from PIL import Image
import torch

from t2c_clip.data import ReIDSample

ImageTransform = Callable[[Image.Image], torch.Tensor]


@dataclass(frozen=True)
class ReIDImageDatasetConfig:
    samples: Sequence[ReIDSample]
    person_id_map: Mapping[int, int]
    camera_id_map: Mapping[int, int]
    transform: ImageTransform


@dataclass(frozen=True)
class ReIDImageItem:
    image: torch.Tensor
    person_id: int
    camera_id: int
    original_person_id: int
    original_camera_id: int


@dataclass(frozen=True)
class ReIDImageBatch:
    images: torch.Tensor
    person_ids: torch.Tensor
    camera_ids: torch.Tensor
    original_person_ids: tuple[int, ...]
    original_camera_ids: tuple[int, ...]


class ReIDImageDataset(torch.utils.data.Dataset):
    def __init__(self, config: ReIDImageDatasetConfig):
        self._config = config

    def __len__(self) -> int:
        return len(self._config.samples)

    def __getitem__(self, index: int) -> ReIDImageItem:
        sample = self._config.samples[index]
        return ReIDImageItem(
            image=self._load_image(sample),
            person_id=_map_value(self._config.person_id_map, sample.person_id, "person_id"),
            camera_id=_map_value(self._config.camera_id_map, sample.camera_id, "camera_id"),
            original_person_id=sample.person_id,
            original_camera_id=sample.camera_id,
        )

    def _load_image(self, sample: ReIDSample) -> torch.Tensor:
        with Image.open(sample.image_path) as image:
            return self._config.transform(image.convert("RGB"))


def build_person_id_map(samples: Sequence[ReIDSample]) -> dict[int, int]:
    return _build_index_map(sample.person_id for sample in samples)


def build_camera_id_map(samples: Sequence[ReIDSample]) -> dict[int, int]:
    return _build_index_map(sample.camera_id for sample in samples)


def collate_reid_batches(items: Sequence[ReIDImageItem]) -> ReIDImageBatch:
    return ReIDImageBatch(
        images=torch.stack([item.image for item in items]),
        person_ids=torch.tensor([item.person_id for item in items], dtype=torch.long),
        camera_ids=torch.tensor([item.camera_id for item in items], dtype=torch.long),
        original_person_ids=tuple(item.original_person_id for item in items),
        original_camera_ids=tuple(item.original_camera_id for item in items),
    )


def _build_index_map(values) -> dict[int, int]:
    return {value: index for index, value in enumerate(sorted(set(values)))}


def _map_value(mapping: Mapping[int, int], value: int, name: str) -> int:
    if value not in mapping:
        raise KeyError(f"{name} {value} is missing from the index map")
    return mapping[value]
