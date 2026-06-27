from argparse import Namespace
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import torch

from t2c_clip.jobs.clip_reid import (
    CLIPLoadResult,
    JobDataConfig,
    build_training_job,
    load_dataset_bundle,
)


class CLIPReIDJobTest(unittest.TestCase):
    def test_load_dataset_bundle_rejects_missing_root(self):
        config = JobDataConfig("market1501", Path("missing"))

        with self.assertRaises(FileNotFoundError):
            load_dataset_bundle(config, FakeImageProcessor())

    def test_build_training_job_returns_real_callbacks_with_fake_clip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            job = build_training_job(_training_args(root), clip_loader=_load_fake_clip)
            reporter = TrainBatchReporterRecorder()

            train_metrics = job.train_one_epoch(1, reporter)
            metrics = job.validate(1)

        self.assertEqual(len(reporter.batch_reports), 1)
        self.assertIn("loss", train_metrics)
        self.assertIn("clip_loss", train_metrics)
        self.assertIn("reid_loss", train_metrics)
        self.assertIn("triplet_loss", train_metrics)
        self.assertIn("tfc_loss", train_metrics)
        self.assertIn("lr", train_metrics)
        self.assertGreaterEqual(metrics.map, 0.0)
        self.assertIn(1, metrics.cmc)

    def test_build_training_job_rejects_training_split_without_positive_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture_without_positive_pairs(Path(tmp))

            with self.assertRaises(ValueError):
                build_training_job(_training_args(root), clip_loader=_load_fake_clip)


class FakeCLIP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = Namespace(projection_dim=2)
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def get_image_features(self, pixel_values):
        pooled = pixel_values.mean(dim=(2, 3))[:, :2]
        return pooled * self.scale


class FakeImageProcessor:
    def __call__(self, images, return_tensors):
        pixel = torch.tensor(images.getpixel((0, 0)), dtype=torch.float32) / 255.0
        values = pixel.view(1, 3, 1, 1).expand(1, 3, 2, 2)
        return {"pixel_values": values}


def _load_fake_clip(model_name: str) -> CLIPLoadResult:
    return CLIPLoadResult(FakeCLIP(), FakeImageProcessor())


class TrainBatchReporterRecorder:
    def __init__(self):
        self.batch_reports: list[dict[str, float]] = []

    def batches(self, iterable):
        return iterable

    def report_batch(self, metrics):
        self.batch_reports.append(dict(metrics))


def _training_args(root: Path) -> Namespace:
    return Namespace(
        dataset="market1501",
        data_root=root,
        clip_model_name="fake-clip",
        batch_size=4,
        num_workers=0,
        lr=0.001,
        device="cpu",
        beta=0.1,
        context_length=2,
        tfc_momentum=0.5,
        triplet_margin=0.3,
        tfc_weight=1.0,
    )


def _build_market_fixture(root: Path) -> Path:
    _write_market_image(root / "bounding_box_train" / "0001_c1s1_000001_01.jpg", "red")
    _write_market_image(root / "bounding_box_train" / "0001_c2s1_000002_01.jpg", "red")
    _write_market_image(root / "bounding_box_train" / "0002_c1s1_000003_01.jpg", "blue")
    _write_market_image(root / "bounding_box_train" / "0002_c2s1_000004_01.jpg", "blue")
    _write_market_image(root / "query" / "0003_c1s1_000004_01.jpg", "green")
    _write_market_image(root / "bounding_box_test" / "0003_c2s1_000005_01.jpg", "green")
    _write_market_image(root / "bounding_box_test" / "0004_c1s1_000006_01.jpg", "blue")
    return root


def _build_market_fixture_without_positive_pairs(root: Path) -> Path:
    _write_market_image(root / "bounding_box_train" / "0001_c1s1_000001_01.jpg", "red")
    _write_market_image(root / "bounding_box_train" / "0002_c1s1_000002_01.jpg", "blue")
    _write_market_image(root / "query" / "0003_c1s1_000003_01.jpg", "green")
    _write_market_image(root / "bounding_box_test" / "0003_c2s1_000004_01.jpg", "green")
    return root


def _write_market_image(path: Path, color: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2), color=color).save(path)
