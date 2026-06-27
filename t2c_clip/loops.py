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
DEFAULT_PROGRESS_DESCRIPTION = ""
STAGE2 = "stage2"
VALIDATION_METRIC_KEYS = ("mAP", "best_mAP")

TrainMetrics = Mapping[str, float]
TrainOneEpoch = Callable[[int, "TrainingEpochReporter"], TrainMetrics | None]
ValidateEpoch = Callable[[int], ReIDMetrics]
ProgressFactory = Callable[[Iterable[int]], Iterable[int]]
MetricLogger = Callable[[int, ReIDMetrics, float | None, bool], None]
TrainMetricLogger = Callable[[int, TrainMetrics], None]
TrainStepMetricLogger = Callable[[int, TrainMetrics], None]


@dataclass(frozen=True)
class TrainingLoopConfig:
    total_epochs: int
    validation_interval: int = DEFAULT_VALIDATION_INTERVAL
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR
    first_epoch: int = 1
    progress_description: str = DEFAULT_PROGRESS_DESCRIPTION
    checkpoint_prefix: str = ""
    stage: str = STAGE2

    def __post_init__(self) -> None:
        _require_positive(self.total_epochs, "total_epochs")
        _require_positive(self.validation_interval, "validation_interval")
        _require_positive(self.first_epoch, "first_epoch")
        if self.checkpoint_prefix and not all(c.isalnum() or c == "_" for c in self.checkpoint_prefix):
            raise ValueError("checkpoint_prefix must be alphanumeric or underscore only")


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


@dataclass(frozen=True)
class TrainingEpochReporterConfig:
    epoch_position: int
    total_epochs: int
    first_train_step: int
    progress_description: str = ""

    @property
    def progress_label(self) -> str:
        prefix = self.progress_description.strip()
        if prefix:
            return f"{prefix} epoch {self.epoch_position}/{self.total_epochs}"
        return f"epoch {self.epoch_position}/{self.total_epochs}"


class TrainingEpochReporter:
    def __init__(
        self,
        config: TrainingEpochReporterConfig,
        progress_factory: ProgressFactory,
        train_step_metric_logger: TrainStepMetricLogger | None,
    ):
        self._config = config
        self._progress_factory = progress_factory
        self._train_step_metric_logger = train_step_metric_logger
        self._last_train_step = config.first_train_step
        self._progress = None

    @property
    def progress(self):
        return self._progress

    @property
    def last_train_step(self) -> int:
        return self._last_train_step

    def batches(self, iterable):
        self._progress = self._progress_factory(iterable, desc=self._description(), leave=False)
        return self._progress

    def report_batch(self, metrics: TrainMetrics) -> None:
        if self._progress is None:
            raise RuntimeError("report_batch requires batches() to be called first")
        train_metrics = _train_metrics(metrics)
        _write_progress_values(self._progress, _train_metric_strings(train_metrics))
        self._last_train_step += 1
        if self._train_step_metric_logger is not None:
            self._train_step_metric_logger(self._last_train_step, train_metrics)

    def _description(self) -> str:
        return self._config.progress_label


def run_training_loop(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    config: TrainingLoopConfig,
    train_one_epoch: TrainOneEpoch,
    validate: ValidateEpoch,
    progress_factory: ProgressFactory = tqdm,
    metric_logger: MetricLogger | None = None,
    train_metric_logger: TrainMetricLogger | None = None,
    train_step_metric_logger: TrainStepMetricLogger | None = None,
) -> TrainingLoopResult:
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_map: float | None = None
    history: list[EpochResult] = []
    train_step = 0
    for epoch in _epoch_numbers(config):
        reporter_config = _epoch_reporter_config(config, epoch, train_step)
        reporter = _epoch_reporter(reporter_config, progress_factory, train_step_metric_logger)
        train_metrics = _train_metrics(train_one_epoch(epoch, reporter))
        train_step = reporter.last_train_step
        _report_training_progress(reporter.progress, train_metrics)
        _log_train_metrics(epoch, train_metrics, train_metric_logger)
        metrics = validate(epoch) if should_validate_epoch(epoch, config.validation_interval) else None
        is_best = _is_best_metric(metrics, best_map)
        best_map = metrics.map if is_best and metrics is not None else best_map
        _report_metrics(reporter.progress, epoch, train_metrics, metrics, best_map, is_best, metric_logger)
        _save_epoch_checkpoints(model, optimizer, config, epoch, metrics, best_map, is_best)
        history.append(EpochResult(epoch, metrics, best_map, is_best, train_metrics))
    return TrainingLoopResult(best_map=best_map, history=tuple(history))


def should_validate_epoch(epoch: int, validation_interval: int) -> bool:
    _require_positive(validation_interval, "validation_interval")
    return epoch % validation_interval == 0


def _epoch_numbers(config: TrainingLoopConfig) -> Iterable[int]:
    return range(config.first_epoch, config.first_epoch + config.total_epochs)


def _epoch_reporter_config(
    config: TrainingLoopConfig,
    epoch: int,
    first_train_step: int,
) -> TrainingEpochReporterConfig:
    epoch_position = epoch - config.first_epoch + 1
    return TrainingEpochReporterConfig(
        epoch_position=epoch_position,
        total_epochs=config.total_epochs,
        first_train_step=first_train_step,
        progress_description=config.progress_description,
    )


def _epoch_reporter(
    config: TrainingEpochReporterConfig,
    progress_factory: ProgressFactory,
    train_step_metric_logger: TrainStepMetricLogger | None,
) -> TrainingEpochReporter:
    return TrainingEpochReporter(
        config=config,
        progress_factory=progress_factory,
        train_step_metric_logger=train_step_metric_logger,
    )


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


def _write_progress_values(progress: Iterable[int] | None, values: Mapping[str, str]) -> None:
    set_postfix = getattr(progress, "set_postfix", None) if progress is not None else None
    if callable(set_postfix):
        set_postfix(values)


def _write_progress_message(progress: Iterable[int] | None, message: str) -> None:
    write = getattr(progress, "write", None) if progress is not None else None
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
    for rank, value in sorted(metrics.cmc.items()):
        values[f"rank{rank}"] = _format_metric(value)
    return values


def _epoch_progress_message(epoch: int, values: Mapping[str, str], is_best: bool) -> str:
    suffix = f" best={is_best}" if "mAP" in values else ""
    return f"epoch={epoch} {_metric_message(values)}{suffix}"


def _metric_message(values: Mapping[str, str]) -> str:
    return " ".join(f"{key}={values[key]}" for key in _message_metric_keys(values))


def _message_metric_keys(values: Mapping[str, str]) -> tuple[str, ...]:
    train_keys = tuple(key for key in values if not _is_validation_metric_key(key))
    return train_keys + _validation_message_keys(values)


def _validation_message_keys(values: Mapping[str, str]) -> tuple[str, ...]:
    rank_keys = tuple(sorted((key for key in values if key.startswith("rank")), key=_rank_number))
    prefix_keys = tuple(key for key in ("mAP",) if key in values)
    suffix_keys = tuple(key for key in ("best_mAP",) if key in values)
    return prefix_keys + rank_keys + suffix_keys


def _is_validation_metric_key(key: str) -> bool:
    return key in VALIDATION_METRIC_KEYS or key.startswith("rank")


def _rank_number(key: str) -> int:
    return int(key.removeprefix("rank"))


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
    payload = _checkpoint_payload(model, optimizer, epoch, metrics, best_map, config.stage)
    last_name = _prefixed_name(LAST_CHECKPOINT_NAME, config.checkpoint_prefix)
    torch.save(payload, config.checkpoint_dir / last_name)
    if is_best:
        best_name = _prefixed_name(BEST_CHECKPOINT_NAME, config.checkpoint_prefix)
        torch.save(payload, config.checkpoint_dir / best_name)


def _prefixed_name(base: str, prefix: str) -> str:
    if not prefix:
        return base
    return f"{prefix}_{base}"


def _checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    metrics: ReIDMetrics | None,
    best_map: float | None,
    stage: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "stage": stage,
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
