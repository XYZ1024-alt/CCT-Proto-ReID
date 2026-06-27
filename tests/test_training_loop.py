from pathlib import Path
import tempfile
import unittest

import torch

from t2c_clip.evaluation import ReIDMetrics
from t2c_clip.loops import TrainingLoopConfig, run_training_loop


class ProgressRecorder:
    def __init__(self):
        self.descriptions: list[str] = []
        self.items: list[int] = []

    def __call__(self, iterable, **kwargs):
        self.descriptions.append(kwargs["desc"])
        self.items = list(iterable)
        return self.items


class TqdmLikeProgressRecorder:
    def __init__(self):
        self.items: list[int] = []
        self.messages: list[str] = []
        self.postfixes: list[dict[str, str]] = []

    def __call__(self, iterable, **kwargs):
        self.items = list(iterable)
        return self

    def __iter__(self):
        return iter(self.items)

    def set_postfix(self, values):
        self.postfixes.append(values)

    def write(self, message):
        self.messages.append(message)


class TrainingLoopTest(unittest.TestCase):
    def test_default_interval_validates_every_five_epochs_and_saves_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = torch.nn.Linear(2, 2)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            progress = ProgressRecorder()
            trained_epochs: list[int] = []
            validated_epochs: list[int] = []

            result = run_training_loop(
                model=model,
                optimizer=optimizer,
                config=TrainingLoopConfig(total_epochs=10, checkpoint_dir=Path(tmp)),
                train_one_epoch=lambda epoch: trained_epochs.append(epoch),
                validate=lambda epoch: _metric(epoch, validated_epochs, {5: 0.3, 10: 0.2}),
                progress_factory=progress,
            )
            best_payload = _load_checkpoint(Path(tmp) / "best.pth")
            last_payload = _load_checkpoint(Path(tmp) / "last.pth")

        self.assertEqual(trained_epochs, list(range(1, 11)))
        self.assertEqual(validated_epochs, [5, 10])
        self.assertEqual(progress.descriptions, ["training"])
        self.assertEqual(progress.items, list(range(1, 11)))
        self.assertEqual(result.best_map, 0.3)
        self.assertEqual(best_payload["epoch"], 5)
        self.assertEqual(last_payload["epoch"], 10)
        self.assertEqual(last_payload["best_map"], 0.3)

    def test_custom_interval_updates_best_only_when_map_improves(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = torch.nn.Linear(2, 2)
            result = run_training_loop(
                model=model,
                optimizer=None,
                config=TrainingLoopConfig(total_epochs=6, validation_interval=2, checkpoint_dir=Path(tmp)),
                train_one_epoch=lambda epoch: None,
                validate=lambda epoch: ReIDMetrics(map={2: 0.2, 4: 0.5, 6: 0.4}[epoch], cmc={1: 0.0}),
                progress_factory=lambda iterable, **kwargs: iterable,
            )
            best_payload = _load_checkpoint(Path(tmp) / "best.pth")
            last_payload = _load_checkpoint(Path(tmp) / "last.pth")

        self.assertEqual([row.is_best for row in result.history if row.metrics is not None], [True, True, False])
        self.assertEqual(result.best_map, 0.5)
        self.assertEqual(best_payload["epoch"], 4)
        self.assertEqual(last_payload["epoch"], 6)
        self.assertEqual(last_payload["metrics"]["mAP"], 0.4)

    def test_validation_interval_must_be_positive(self):
        with self.assertRaises(ValueError):
            TrainingLoopConfig(total_epochs=1, validation_interval=0)

    def test_validation_metrics_are_reported_to_progress_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            progress = TqdmLikeProgressRecorder()
            run_training_loop(
                model=torch.nn.Linear(2, 2),
                optimizer=None,
                config=TrainingLoopConfig(total_epochs=1, validation_interval=1, checkpoint_dir=Path(tmp)),
                train_one_epoch=lambda epoch: None,
                validate=lambda epoch: ReIDMetrics(map=0.25, cmc={1: 0.5}),
                progress_factory=progress,
            )

        self.assertEqual(progress.postfixes, [{"mAP": "0.2500", "best_mAP": "0.2500", "rank1": "0.5000"}])
        self.assertEqual(progress.messages, ["epoch=1 mAP=0.2500 rank1=0.5000 best_mAP=0.2500 best=True"])

    def test_each_epoch_is_reported_even_without_validation_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            progress = TqdmLikeProgressRecorder()
            run_training_loop(
                model=torch.nn.Linear(2, 2),
                optimizer=None,
                config=TrainingLoopConfig(total_epochs=3, validation_interval=2, checkpoint_dir=Path(tmp)),
                train_one_epoch=lambda epoch: None,
                validate=lambda epoch: ReIDMetrics(map=0.25, cmc={1: 0.5}),
                progress_factory=progress,
            )

        self.assertEqual(
            progress.messages,
            [
                "epoch=1 done",
                "epoch=2 mAP=0.2500 rank1=0.5000 best_mAP=0.2500 best=True",
                "epoch=3 done",
            ],
        )

    def test_training_metrics_are_reported_each_epoch(self):
        logged: list[tuple[int, dict[str, float]]] = []
        with tempfile.TemporaryDirectory() as tmp:
            progress = TqdmLikeProgressRecorder()
            run_training_loop(
                model=torch.nn.Linear(2, 2),
                optimizer=None,
                config=TrainingLoopConfig(total_epochs=2, validation_interval=2, checkpoint_dir=Path(tmp)),
                train_one_epoch=lambda epoch: {"loss": float(epoch), "lr": 0.01},
                validate=lambda epoch: ReIDMetrics(map=0.25, cmc={1: 0.5}),
                progress_factory=progress,
                train_metric_logger=lambda epoch, metrics: logged.append((epoch, dict(metrics))),
            )

        self.assertEqual(
            progress.postfixes,
            [
                {"loss": "1.0000", "lr": "0.0100"},
                {"loss": "2.0000", "lr": "0.0100"},
                {
                    "loss": "2.0000",
                    "lr": "0.0100",
                    "mAP": "0.2500",
                    "best_mAP": "0.2500",
                    "rank1": "0.5000",
                },
            ],
        )
        self.assertEqual(
            progress.messages,
            [
                "epoch=1 loss=1.0000 lr=0.0100",
                "epoch=2 loss=2.0000 lr=0.0100 mAP=0.2500 rank1=0.5000 best_mAP=0.2500 best=True",
            ],
        )
        self.assertEqual(logged, [(1, {"loss": 1.0, "lr": 0.01}), (2, {"loss": 2.0, "lr": 0.01})])

    def test_validation_metrics_are_sent_to_metric_logger(self):
        logged: list[tuple[int, ReIDMetrics, float | None, bool]] = []
        with tempfile.TemporaryDirectory() as tmp:
            run_training_loop(
                model=torch.nn.Linear(2, 2),
                optimizer=None,
                config=TrainingLoopConfig(total_epochs=1, validation_interval=1, checkpoint_dir=Path(tmp)),
                train_one_epoch=lambda epoch: None,
                validate=lambda epoch: ReIDMetrics(map=0.25, cmc={1: 0.5}),
                progress_factory=lambda iterable, **kwargs: iterable,
                metric_logger=lambda epoch, metrics, best_map, is_best: logged.append(
                    (epoch, metrics, best_map, is_best)
                ),
            )

        self.assertEqual(logged, [(1, ReIDMetrics(map=0.25, cmc={1: 0.5}), 0.25, True)])


def _metric(epoch: int, validated_epochs: list[int], values: dict[int, float]) -> ReIDMetrics:
    validated_epochs.append(epoch)
    return ReIDMetrics(map=values[epoch], cmc={1: values[epoch]})


def _load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)
