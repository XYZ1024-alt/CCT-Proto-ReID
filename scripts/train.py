"""Training entrypoint for project-specific T2C-CLIP jobs."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable
from contextlib import nullcontext
from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Sequence

import torch

from t2c_clip.evaluation import ReIDMetrics
from t2c_clip.loops import MetricLogger, TrainMetricLogger, TrainingLoopConfig, run_training_loop
from t2c_clip.mlflow import (
    MLflowSQLiteConfig,
    log_reid_metrics_to_mlflow,
    log_training_metrics_to_mlflow,
    start_mlflow_sqlite_run,
)

DEFAULT_TOTAL_EPOCHS = 120
DEFAULT_VALIDATION_INTERVAL = 5
DEFAULT_CHECKPOINT_DIR = Path("checkpoints")
DEFAULT_TRACKING_DB = Path("mlflow") / "t2c_clip.db"
DEFAULT_ARTIFACT_ROOT = Path("mlruns")
DEFAULT_EXPERIMENT_NAME = "T2C-CLIP"
DEFAULT_RUN_NAME = "train"
DEFAULT_CLIP_MODEL_NAME = "openai/clip-vit-base-patch16"
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_WORKERS = 4
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_DEVICE = "cuda"
DEFAULT_BETA = 0.1
DEFAULT_CONTEXT_LENGTH = 4
DEFAULT_TFC_MOMENTUM = 0.5
DEFAULT_TRIPLET_MARGIN = 0.3
DEFAULT_TFC_WEIGHT = 1.0
SUPPORTED_DATASETS = ("market1501", "msmt17")

TrainOneEpoch = Callable[[int], dict[str, float] | None]
ValidateEpoch = Callable[[int], ReIDMetrics]
JobBuilder = Callable[[argparse.Namespace], "TrainingJob"]
ProgressFactory = Callable[[Iterable[int]], Iterable[int]]


@dataclass(frozen=True)
class TrainingJob:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer | None
    train_one_epoch: TrainOneEpoch
    validate: ValidateEpoch


def main(argv: Sequence[str] | None = None, progress_factory: ProgressFactory | None = None) -> int:
    args = _build_parser().parse_args(argv)
    with _mlflow_context_if_requested(args):
        job = _load_job_builder(args.job_builder)(args)
        config = TrainingLoopConfig(
            total_epochs=args.epochs,
            validation_interval=args.validation_interval,
            checkpoint_dir=args.checkpoint_dir,
        )
        _run_loop(job, config, progress_factory, _metric_logger_if_requested(args), _train_logger_if_requested(args))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a T2C-CLIP training job.")
    parser.add_argument("--job-builder", required=True)
    parser.add_argument("--epochs", type=int, default=DEFAULT_TOTAL_EPOCHS)
    parser.add_argument("--validation-interval", type=int, default=DEFAULT_VALIDATION_INTERVAL)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--enable-mlflow", action="store_true")
    parser.add_argument("--tracking-db", type=Path, default=DEFAULT_TRACKING_DB)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    _add_project_training_args(parser)
    return parser


def _add_project_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--clip-model-name", default=DEFAULT_CLIP_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    parser.add_argument("--tfc-momentum", type=float, default=DEFAULT_TFC_MOMENTUM)
    parser.add_argument("--triplet-margin", type=float, default=DEFAULT_TRIPLET_MARGIN)
    parser.add_argument("--tfc-weight", type=float, default=DEFAULT_TFC_WEIGHT)


def _mlflow_context_if_requested(args: argparse.Namespace):
    if not args.enable_mlflow:
        return nullcontext()
    config = MLflowSQLiteConfig(args.tracking_db, args.artifact_root, args.experiment_name)
    return start_mlflow_sqlite_run(config, run_name=args.run_name)


def _metric_logger_if_requested(args: argparse.Namespace) -> MetricLogger | None:
    if not args.enable_mlflow:
        return None
    return log_reid_metrics_to_mlflow


def _train_logger_if_requested(args: argparse.Namespace) -> TrainMetricLogger | None:
    if not args.enable_mlflow:
        return None
    return log_training_metrics_to_mlflow


def _run_loop(
    job: TrainingJob,
    config: TrainingLoopConfig,
    progress_factory: ProgressFactory | None,
    metric_logger: MetricLogger | None,
    train_metric_logger: TrainMetricLogger | None,
) -> None:
    if progress_factory is None:
        run_training_loop(
            job.model,
            job.optimizer,
            config,
            job.train_one_epoch,
            job.validate,
            metric_logger=metric_logger,
            train_metric_logger=train_metric_logger,
        )
        return
    run_training_loop(
        job.model,
        job.optimizer,
        config,
        job.train_one_epoch,
        job.validate,
        progress_factory,
        metric_logger=metric_logger,
        train_metric_logger=train_metric_logger,
    )


def _load_job_builder(spec: str) -> JobBuilder:
    module_name, function_name = _split_builder_spec(spec)
    module = importlib.import_module(module_name)
    builder = getattr(module, function_name)
    if not callable(builder):
        raise TypeError(f"Job builder is not callable: {spec}")
    return builder


def _split_builder_spec(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        raise ValueError("job builder must use 'module:function' format")
    module_name, function_name = spec.split(":", maxsplit=1)
    if not module_name or not function_name:
        raise ValueError("job builder must include both module and function")
    return module_name, function_name


if __name__ == "__main__":
    raise SystemExit(main())
