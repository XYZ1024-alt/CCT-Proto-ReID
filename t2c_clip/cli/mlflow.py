"""Initialize T2C-CLIP MLflow SQLite tracking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from t2c_clip.mlflow import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_INIT_RUN_NAME,
    DEFAULT_MLFLOW_UI_HOST,
    DEFAULT_MLFLOW_UI_PORT,
    DEFAULT_TRACKING_DB,
    MLflowInitialization,
    MLflowSQLiteConfig,
    initialize_mlflow_sqlite,
    mlflow_ui_command,
)


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = MLflowSQLiteConfig(args.tracking_db, args.artifact_root, args.experiment_name)
    result = initialize_mlflow_sqlite(config, run_name=args.run_name)
    payload = _payload(result, mlflow_ui_command(config, args.host, args.port))
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is None:
        print(text)
        return 0
    args.output.write_text(text + "\n", encoding="utf-8")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Initialize MLflow with a SQLite backend store.")
    parser.add_argument("--tracking-db", type=Path, default=DEFAULT_TRACKING_DB)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--run-name", default=DEFAULT_INIT_RUN_NAME)
    parser.add_argument("--host", default=DEFAULT_MLFLOW_UI_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_MLFLOW_UI_PORT)
    parser.add_argument("--output", type=Path)
    return parser


def _payload(result: MLflowInitialization, ui_command: str) -> dict[str, str]:
    return {
        "artifact_uri": result.artifact_uri,
        "experiment_id": result.experiment_id,
        "experiment_name": result.experiment_name,
        "run_id": result.run_id,
        "tracking_uri": result.tracking_uri,
        "ui_command": ui_command,
    }


if __name__ == "__main__":
    raise SystemExit(main())
