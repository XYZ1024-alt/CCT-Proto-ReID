"""Train2Central-CLIP core package."""

from t2c_clip.data import ReIDSample
from t2c_clip.evaluation import ReIDMetrics, evaluate_reid
from t2c_clip.features import fuse_features, l2_normalize
from t2c_clip.losses import batch_hard_triplet_loss, bidirectional_contrastive_loss
from t2c_clip.loops import (
    DEFAULT_VALIDATION_INTERVAL,
    EpochResult,
    TrainingLoopConfig,
    TrainingLoopResult,
    run_training_loop,
    should_validate_epoch,
)
from t2c_clip.model import T2CClipModel
from t2c_clip.mlflow import (
    DEFAULT_MLFLOW_UI_PORT,
    MLflowInitialization,
    MLflowSQLiteConfig,
    initialize_mlflow_sqlite,
    mlflow_ui_command,
    sqlite_tracking_uri,
)
from t2c_clip.prompts import PromptBank, PromptConfig
from t2c_clip.tfc import TFCCenterBank
from t2c_clip.training import (
    Stage1LossConfig,
    Stage2LossBreakdown,
    Stage2LossConfig,
    Stage2LossInputs,
    TrainingBatch,
    stage1_alignment_loss,
    stage2_loss_breakdown,
)

__all__ = [
    "PromptBank",
    "PromptConfig",
    "DEFAULT_MLFLOW_UI_PORT",
    "DEFAULT_VALIDATION_INTERVAL",
    "EpochResult",
    "MLflowInitialization",
    "MLflowSQLiteConfig",
    "ReIDMetrics",
    "ReIDSample",
    "Stage1LossConfig",
    "Stage2LossBreakdown",
    "Stage2LossConfig",
    "Stage2LossInputs",
    "T2CClipModel",
    "TFCCenterBank",
    "TrainingLoopConfig",
    "TrainingLoopResult",
    "TrainingBatch",
    "batch_hard_triplet_loss",
    "bidirectional_contrastive_loss",
    "evaluate_reid",
    "fuse_features",
    "initialize_mlflow_sqlite",
    "l2_normalize",
    "mlflow_ui_command",
    "run_training_loop",
    "should_validate_epoch",
    "sqlite_tracking_uri",
    "stage1_alignment_loss",
    "stage2_loss_breakdown",
]
