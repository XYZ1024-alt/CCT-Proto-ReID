from pathlib import Path
import contextlib
import io
import tempfile
import unittest

import torch

from scripts.train import TrainingJob, main
from t2c_clip.evaluation import ReIDMetrics


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
