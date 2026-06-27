from argparse import Namespace
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import torch

from t2c_clip.datasets import ReIDImageBatch
from t2c_clip.jobs.clip_reid import (
    CLIPLoadResult,
    JobDataConfig,
    _extract_features,
    build_training_job,
    load_dataset_bundle,
)
from t2c_clip.retrieval import IMAGE_ONLY_RETRIEVAL
from tests._clip_fakes import FakeCLIP, ImageAwareFakeImageProcessor


class CLIPReIDJobTest(unittest.TestCase):
    def test_load_dataset_bundle_rejects_missing_root(self):
        config = JobDataConfig("market1501", Path("missing"))

        with self.assertRaises(FileNotFoundError):
            load_dataset_bundle(config, ImageAwareFakeImageProcessor())

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

    def test_build_training_job_returns_two_stage_job_when_stage1_epochs_positive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture(Path(tmp))
            args = _training_args(root)
            args.stage1_epochs = 1
            job = build_training_job(args, clip_loader=_load_fake_clip)

        from scripts.train import TwoStageTrainingJob
        self.assertIsInstance(job, TwoStageTrainingJob)

    def test_build_training_job_rejects_training_split_without_positive_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = _build_market_fixture_without_positive_pairs(Path(tmp))

            with self.assertRaises(ValueError):
                build_training_job(_training_args(root), clip_loader=_load_fake_clip)

    def test_extract_features_passes_configured_retrieval_mode(self):
        model = RetrievalModeRecorder()
        batch = ReIDImageBatch(
            images=torch.ones(2, 3, 2, 2),
            person_ids=torch.tensor([0, 1]),
            camera_ids=torch.tensor([1, 2]),
            original_person_ids=(10, 20),
            original_camera_ids=(1, 2),
        )

        features = _extract_features(
            model,
            [batch],
            torch.device("cpu"),
            retrieval_mode=IMAGE_ONLY_RETRIEVAL,
        )

        self.assertEqual(model.retrieval_modes, [IMAGE_ONLY_RETRIEVAL])
        self.assertEqual(features.person_ids, (10, 20))
        self.assertEqual(features.camera_ids, (1, 2))
        self.assertEqual(tuple(features.features.shape), (2, 4))


def _load_fake_clip(model_name: str) -> CLIPLoadResult:
    return CLIPLoadResult(FakeCLIP(hidden_size=8, projection_dim=4), ImageAwareFakeImageProcessor(), tokenizer=None)


class TrainBatchReporterRecorder:
    def __init__(self):
        self.batch_reports: list[dict[str, float]] = []

    def batches(self, iterable):
        return iterable

    def report_batch(self, metrics):
        self.batch_reports.append(dict(metrics))


class RetrievalModeRecorder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.retrieval_modes: list[str] = []

    def encode_retrieval(
        self,
        images: torch.Tensor,
        camera_ids: torch.Tensor,
        retrieval_mode: str = "fused",
    ) -> torch.Tensor:
        self.retrieval_modes.append(retrieval_mode)
        return torch.ones(images.shape[0], 4)


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
        clip_weight=0.1,
        stage1_epochs=0,
        epochs=1,
        validation_interval=1,
        freeze_image_encoder_stage1=True,
        freeze_image_encoder_stage2=True,
        freeze_text_encoder=True,
        retrieval_mode="fused",
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
