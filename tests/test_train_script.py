from pathlib import Path
import contextlib
import io
import tempfile
import unittest

from mlflow.tracking import MlflowClient
import torch

from scripts.train import TrainingJob, main
from t2c_clip.evaluation import ReIDMetrics
from t2c_clip.mlflow import sqlite_tracking_uri

RECORDED_ARGS = None


class TrainScriptTest(unittest.TestCase):
    def test_main_requires_job_builder(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as context:
                main([])

        self.assertNotEqual(context.exception.code, 0)

    def test_main_runs_builder_training_job_and_saves_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / "checkpoints"
            exit_code = main(
                [
                    "--job-builder",
                    "tests.test_train_script:build_training_job",
                    "--epochs",
                    "2",
                    "--validation-interval",
                    "1",
                    "--checkpoint-dir",
                    str(checkpoint_dir),
                ],
                progress_factory=lambda iterable, **kwargs: iterable,
            )
            best_payload = torch.load(checkpoint_dir / "best.pth", map_location="cpu", weights_only=True)
            last_payload = torch.load(checkpoint_dir / "last.pth", map_location="cpu", weights_only=True)

        self.assertEqual(exit_code, 0)
        self.assertEqual(best_payload["epoch"], 2)
        self.assertEqual(last_payload["epoch"], 2)
        self.assertEqual(last_payload["metrics"]["mAP"], 0.2)

    def test_main_passes_project_training_args_to_builder(self):
        global RECORDED_ARGS
        RECORDED_ARGS = None
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp) / "checkpoints"
            exit_code = main(
                _recording_job_args(checkpoint_dir),
                progress_factory=lambda iterable, **kwargs: iterable,
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(RECORDED_ARGS.dataset, "msmt17")
        self.assertEqual(RECORDED_ARGS.data_root, Path("MSMT17_V1"))
        self.assertEqual(RECORDED_ARGS.clip_model_name, "openai/clip-vit-base-patch16")
        self.assertEqual(RECORDED_ARGS.batch_size, 8)
        self.assertEqual(RECORDED_ARGS.num_workers, 2)
        self.assertEqual(RECORDED_ARGS.lr, 0.001)
        self.assertEqual(RECORDED_ARGS.device, "cpu")

    def test_main_logs_validation_metrics_when_mlflow_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            tracking_db = Path(tmp) / "mlflow" / "tracking.db"
            artifact_root = Path(tmp) / "mlruns"
            checkpoint_dir = Path(tmp) / "checkpoints"
            exit_code = main(
                [
                    "--job-builder",
                    f"{__name__}:build_training_job",
                    "--epochs",
                    "1",
                    "--validation-interval",
                    "1",
                    "--checkpoint-dir",
                    str(checkpoint_dir),
                    "--enable-mlflow",
                    "--tracking-db",
                    str(tracking_db),
                    "--artifact-root",
                    str(artifact_root),
                    "--experiment-name",
                    "T2C-CLIP-TrainScript-Test",
                    "--run-name",
                    "train-script-test",
                ],
                progress_factory=lambda iterable, **kwargs: iterable,
            )
            runs = _runs_for_experiment(tracking_db, "T2C-CLIP-TrainScript-Test")

        self.assertEqual(exit_code, 0)
        self.assertEqual(runs[0].data.metrics["mAP"], 0.1)
        self.assertEqual(runs[0].data.metrics["rank_1"], 0.1)


def build_training_job(args) -> TrainingJob:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    def train_one_epoch(epoch: int) -> None:
        optimizer.zero_grad()
        loss = model(torch.eye(2)).sum() * epoch
        loss.backward()
        optimizer.step()

    def validate(epoch: int) -> ReIDMetrics:
        return ReIDMetrics(map=epoch / 10.0, cmc={1: epoch / 10.0})

    return TrainingJob(model, optimizer, train_one_epoch, validate)


def recording_training_job(args) -> TrainingJob:
    global RECORDED_ARGS
    RECORDED_ARGS = args
    return build_training_job(args)


def _recording_job_args(checkpoint_dir: Path) -> list[str]:
    return [
        "--job-builder",
        f"{__name__}:recording_training_job",
        "--epochs",
        "1",
        "--validation-interval",
        "1",
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--dataset",
        "msmt17",
        "--data-root",
        "MSMT17_V1",
        "--clip-model-name",
        "openai/clip-vit-base-patch16",
        "--batch-size",
        "8",
        "--num-workers",
        "2",
        "--lr",
        "0.001",
        "--device",
        "cpu",
    ]


def _runs_for_experiment(tracking_db: Path, experiment_name: str):
    client = MlflowClient(tracking_uri=sqlite_tracking_uri(tracking_db))
    experiment = client.get_experiment_by_name(experiment_name)
    return client.search_runs([experiment.experiment_id])
