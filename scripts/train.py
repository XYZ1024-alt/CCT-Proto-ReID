"""Training entrypoint for project-specific T2C-CLIP jobs."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable
from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Sequence

import torch

from t2c_clip.evaluation import ReIDMetrics
from t2c_clip.loops import TrainingLoopConfig, run_training_loop
from t2c_clip.mlflow import MLflowSQLiteConfig, initialize_mlflow_sqlite

DEFAULT_TOTAL_EPOCHS = 120
DEFAULT_VALIDATION_INTERVAL = 5
DEFAULT_CHECKPOINT_DIR = Path("checkpoints")
DEFAULT_TRACKING_DB = Path("mlflow") / "t2c_clip.db"
DEFAULT_ARTIFACT_ROOT = Path("mlruns")
DEFAULT_EXPERIMENT_NAME = "T2C-CLIP"
DEFAULT_RUN_NAME = "train"

TrainOneEpoch = Callable[[int], None]
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
    _initialize_mlflow_if_requested(args)
    job = _load_job_builder(args.job_builder)(args)
    config = TrainingLoopConfig(
        total_epochs=args.epochs,
        validation_interval=args.validation_interval,
        checkpoint_dir=args.checkpoint_dir,
    )
    _run_loop(job, config, progress_factory)
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
    return parser


def _initialize_mlflow_if_requested(args: argparse.Namespace) -> None:
    if not args.enable_mlflow:
        return
    config = MLflowSQLiteConfig(args.tracking_db, args.artifact_root, args.experiment_name)
    initialize_mlflow_sqlite(config, run_name=args.run_name)


def _run_loop(
    job: TrainingJob,
    config: TrainingLoopConfig,
    progress_factory: ProgressFactory | None,
) -> None:
    if progress_factory is None:
        run_training_loop(job.model, job.optimizer, config, job.train_one_epoch, job.validate)
        return
    run_training_loop(job.model, job.optimizer, config, job.train_one_epoch, job.validate, progress_factory)


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
