"""Reusable training loop with validation scheduling and checkpoints."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from t2c_clip.evaluation import ReIDMetrics

DEFAULT_VALIDATION_INTERVAL = 5
DEFAULT_CHECKPOINT_DIR = Path("checkpoints")
BEST_CHECKPOINT_NAME = "best.pth"
LAST_CHECKPOINT_NAME = "last.pth"
DEFAULT_PROGRESS_DESCRIPTION = "training"

TrainOneEpoch = Callable[[int], Any]
ValidateEpoch = Callable[[int], ReIDMetrics]
ProgressFactory = Callable[[Iterable[int]], Iterable[int]]


@dataclass(frozen=True)
class TrainingLoopConfig:
    total_epochs: int
    validation_interval: int = DEFAULT_VALIDATION_INTERVAL
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR
    first_epoch: int = 1
    progress_description: str = DEFAULT_PROGRESS_DESCRIPTION

    def __post_init__(self) -> None:
        _require_positive(self.total_epochs, "total_epochs")
        _require_positive(self.validation_interval, "validation_interval")
        _require_positive(self.first_epoch, "first_epoch")


@dataclass(frozen=True)
class EpochResult:
    epoch: int
    metrics: ReIDMetrics | None
    best_map: float | None
    is_best: bool


@dataclass(frozen=True)
class TrainingLoopResult:
    best_map: float | None
    history: tuple[EpochResult, ...] = field(default_factory=tuple)


def run_training_loop(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    config: TrainingLoopConfig,
    train_one_epoch: TrainOneEpoch,
    validate: ValidateEpoch,
    progress_factory: ProgressFactory = tqdm,
) -> TrainingLoopResult:
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_map: float | None = None
    history: list[EpochResult] = []
    for epoch in _progress_epochs(config, progress_factory):
        train_one_epoch(epoch)
        metrics = validate(epoch) if should_validate_epoch(epoch, config.validation_interval) else None
        is_best = _is_best_metric(metrics, best_map)
        best_map = metrics.map if is_best and metrics is not None else best_map
        _save_epoch_checkpoints(model, optimizer, config, epoch, metrics, best_map, is_best)
        history.append(EpochResult(epoch, metrics, best_map, is_best))
    return TrainingLoopResult(best_map=best_map, history=tuple(history))


def should_validate_epoch(epoch: int, validation_interval: int) -> bool:
    _require_positive(validation_interval, "validation_interval")
    return epoch % validation_interval == 0


def _progress_epochs(config: TrainingLoopConfig, progress_factory: ProgressFactory) -> Iterable[int]:
    epochs = range(config.first_epoch, config.first_epoch + config.total_epochs)
    return progress_factory(epochs, desc=config.progress_description)


def _is_best_metric(metrics: ReIDMetrics | None, best_map: float | None) -> bool:
    if metrics is None:
        return False
    if best_map is None:
        return True
    return metrics.map > best_map


def _save_epoch_checkpoints(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    config: TrainingLoopConfig,
    epoch: int,
    metrics: ReIDMetrics | None,
    best_map: float | None,
    is_best: bool,
) -> None:
    payload = _checkpoint_payload(model, optimizer, epoch, metrics, best_map)
    torch.save(payload, config.checkpoint_dir / LAST_CHECKPOINT_NAME)
    if is_best:
        torch.save(payload, config.checkpoint_dir / BEST_CHECKPOINT_NAME)


def _checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    metrics: ReIDMetrics | None,
    best_map: float | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "epoch": epoch,
        "best_map": best_map,
        "metrics": _metrics_payload(metrics),
        "model_state": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    return payload


def _metrics_payload(metrics: ReIDMetrics | None) -> dict[str, Any] | None:
    if metrics is None:
        return None
    return {"mAP": metrics.map, "cmc": dict(metrics.cmc)}


def _require_positive(value: int, name: str) -> None:
    if value < 1:
        raise ValueError(f"{name} must be positive")
