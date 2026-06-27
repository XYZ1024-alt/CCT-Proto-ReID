from collections import Counter
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import torch

from t2c_clip.data import ReIDSample
from t2c_clip.datasets import (
    IdentityBalancedBatchSampler,
    ReIDImageDatasetConfig,
    ReIDImageDataset,
    build_camera_id_map,
    build_person_id_map,
    collate_reid_batches,
)


class ReIDImageDatasetTest(unittest.TestCase):
    def test_reid_image_dataset_returns_remapped_and_original_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "0002_c3s1_000551_01.jpg"
            Image.new("RGB", (2, 2), color="red").save(image_path)
            sample = ReIDSample(image_path, 42, 7, "market1501", "train")
            dataset = ReIDImageDataset(_dataset_config([sample], {42: 0}, {7: 0}))

            batch = dataset[0]

        self.assertTrue(torch.equal(batch.image, torch.ones(3, 2, 2)))
        self.assertEqual(batch.person_id, 0)
        self.assertEqual(batch.camera_id, 0)
        self.assertEqual(batch.original_person_id, 42)
        self.assertEqual(batch.original_camera_id, 7)

    def test_build_index_maps_sorts_unique_values(self):
        samples = [
            ReIDSample(Path("a.jpg"), 9, 3, "market1501", "train"),
            ReIDSample(Path("b.jpg"), 4, 1, "market1501", "train"),
        ]

        self.assertEqual(build_person_id_map(samples), {4: 0, 9: 1})
        self.assertEqual(build_camera_id_map(samples), {1: 0, 3: 1})

    def test_collate_reid_batches_stacks_images_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = _sample(Path(tmp) / "a.jpg", 9, 3)
            second = _sample(Path(tmp) / "b.jpg", 4, 1)
            dataset = ReIDImageDataset(_dataset_config([first, second], {4: 0, 9: 1}, {1: 0, 3: 1}))

            batch = collate_reid_batches([dataset[0], dataset[1]])

        self.assertEqual(batch.images.shape, (2, 3, 2, 2))
        self.assertTrue(torch.equal(batch.person_ids, torch.tensor([1, 0])))
        self.assertTrue(torch.equal(batch.camera_ids, torch.tensor([1, 0])))
        self.assertEqual(batch.original_person_ids, (9, 4))
        self.assertEqual(batch.original_camera_ids, (3, 1))

    def test_identity_balanced_batch_sampler_groups_positive_and_negative_pairs(self):
        labels = [0, 0, 0, 1, 1, 1, 2, 2]
        sampler = IdentityBalancedBatchSampler(labels, batch_size=4, instances_per_identity=2)

        batch = next(iter(sampler))
        counts = Counter(labels[index] for index in batch)

        self.assertEqual(len(counts), 2)
        self.assertEqual(set(counts.values()), {2})

    def test_identity_balanced_batch_sampler_rejects_missing_positive_pairs(self):
        labels = [0, 1, 2, 3]

        with self.assertRaises(ValueError):
            IdentityBalancedBatchSampler(labels, batch_size=4, instances_per_identity=2)


def _tensor_transform(image: Image.Image) -> torch.Tensor:
    return torch.ones(3, image.height, image.width)


def _sample(path: Path, person_id: int, camera_id: int) -> ReIDSample:
    Image.new("RGB", (2, 2), color="blue").save(path)
    return ReIDSample(path, person_id, camera_id, "market1501", "train")


def _dataset_config(samples, person_id_map, camera_id_map) -> ReIDImageDatasetConfig:
    return ReIDImageDatasetConfig(samples, person_id_map, camera_id_map, _tensor_transform)
