"""Reusable training loop with validation scheduling and checkpoints."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
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
VALIDATION_MESSAGE_KEYS = ("mAP", "rank1", "best_mAP")

TrainMetrics = Mapping[str, float]
TrainOneEpoch = Callable[[int], TrainMetrics | None]
ValidateEpoch = Callable[[int], ReIDMetrics]
ProgressFactory = Callable[[Iterable[int]], Iterable[int]]
MetricLogger = Callable[[int, ReIDMetrics, float | None, bool], None]
TrainMetricLogger = Callable[[int, TrainMetrics], None]


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
    train_metrics: TrainMetrics = field(default_factory=dict)


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
    metric_logger: MetricLogger | None = None,
    train_metric_logger: TrainMetricLogger | None = None,
) -> TrainingLoopResult:
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_map: float | None = None
    history: list[EpochResult] = []
    progress = _progress_epochs(config, progress_factory)
    for epoch in progress:
        train_metrics = _train_metrics(train_one_epoch(epoch))
        _report_training_progress(progress, train_metrics)
        _log_train_metrics(epoch, train_metrics, train_metric_logger)
        metrics = validate(epoch) if should_validate_epoch(epoch, config.validation_interval) else None
        is_best = _is_best_metric(metrics, best_map)
        best_map = metrics.map if is_best and metrics is not None else best_map
        _report_metrics(progress, epoch, train_metrics, metrics, best_map, is_best, metric_logger)
        _save_epoch_checkpoints(model, optimizer, config, epoch, metrics, best_map, is_best)
        history.append(EpochResult(epoch, metrics, best_map, is_best, train_metrics))
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


def _report_metrics(
    progress: Iterable[int],
    epoch: int,
    train_metrics: TrainMetrics,
    metrics: ReIDMetrics | None,
    best_map: float | None,
    is_best: bool,
    metric_logger: MetricLogger | None,
) -> None:
    if metrics is None:
        _write_epoch_progress_message(progress, epoch, train_metrics, None, best_map, is_best)
        return
    _write_epoch_progress_metrics(progress, epoch, train_metrics, metrics, best_map, is_best)
    if metric_logger is not None:
        metric_logger(epoch, metrics, best_map, is_best)


def _report_training_progress(progress: Iterable[int], train_metrics: TrainMetrics) -> None:
    if train_metrics:
        _write_progress_values(progress, _train_metric_strings(train_metrics))


def _log_train_metrics(
    epoch: int,
    train_metrics: TrainMetrics,
    train_metric_logger: TrainMetricLogger | None,
) -> None:
    if train_metrics and train_metric_logger is not None:
        train_metric_logger(epoch, train_metrics)


def _write_epoch_progress_metrics(
    progress: Iterable[int],
    epoch: int,
    train_metrics: TrainMetrics,
    metrics: ReIDMetrics,
    best_map: float | None,
    is_best: bool,
) -> None:
    values = _epoch_metric_strings(train_metrics, metrics, best_map)
    _write_progress_values(progress, values)
    _write_progress_message(progress, _epoch_progress_message(epoch, values, is_best))


def _write_epoch_progress_message(
    progress: Iterable[int],
    epoch: int,
    train_metrics: TrainMetrics,
    metrics: ReIDMetrics | None,
    best_map: float | None,
    is_best: bool,
) -> None:
    values = _epoch_metric_strings(train_metrics, metrics, best_map)
    message = _epoch_progress_message(epoch, values, is_best) if values else f"epoch={epoch} done"
    _write_progress_message(progress, message)


def _write_progress_values(progress: Iterable[int], values: Mapping[str, str]) -> None:
    set_postfix = getattr(progress, "set_postfix", None)
    if callable(set_postfix):
        set_postfix(values)


def _write_progress_message(progress: Iterable[int], message: str) -> None:
    write = getattr(progress, "write", None)
    if callable(write):
        write(message)


def _train_metrics(result: TrainMetrics | None) -> TrainMetrics:
    if result is None:
        return {}
    return {name: float(value) for name, value in result.items()}


def _epoch_metric_strings(
    train_metrics: TrainMetrics,
    metrics: ReIDMetrics | None,
    best_map: float | None,
) -> dict[str, str]:
    values = _train_metric_strings(train_metrics)
    if metrics is not None:
        values.update(_metric_strings(metrics, best_map))
    return values


def _train_metric_strings(train_metrics: TrainMetrics) -> dict[str, str]:
    return {name: _format_metric(value) for name, value in train_metrics.items()}


def _metric_strings(metrics: ReIDMetrics, best_map: float | None) -> dict[str, str]:
    values = {"mAP": _format_metric(metrics.map), "best_mAP": _format_optional_metric(best_map)}
    if 1 in metrics.cmc:
        values["rank1"] = _format_metric(metrics.cmc[1])
    return values


def _epoch_progress_message(epoch: int, values: Mapping[str, str], is_best: bool) -> str:
    suffix = f" best={is_best}" if "mAP" in values else ""
    return f"epoch={epoch} {_metric_message(values)}{suffix}"


def _metric_message(values: Mapping[str, str]) -> str:
    return " ".join(f"{key}={values[key]}" for key in _message_metric_keys(values))


def _message_metric_keys(values: Mapping[str, str]) -> tuple[str, ...]:
    train_keys = tuple(key for key in values if key not in VALIDATION_MESSAGE_KEYS)
    validation_keys = tuple(key for key in VALIDATION_MESSAGE_KEYS if key in values)
    return train_keys + validation_keys


def _format_optional_metric(value: float | None) -> str:
    if value is None:
        return "nan"
    return _format_metric(value)


def _format_metric(value: float) -> str:
    return f"{value:.4f}"


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
