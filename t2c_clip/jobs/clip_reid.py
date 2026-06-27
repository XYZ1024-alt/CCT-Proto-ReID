"""Real CLIP-backed T2C-CLIP training job builder."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from scripts.train import TrainingJob
from t2c_clip.clip_backbone import PromptTextEncoder, TransformersCLIPImageEncoder, clip_projection_dim
from t2c_clip.data import ReIDSample, load_market_split, load_msmt17_manifest
from t2c_clip.datasets import (
    ReIDImageBatch,
    ReIDImageDataset,
    ReIDImageDatasetConfig,
    build_camera_id_map,
    build_person_id_map,
    collate_reid_batches,
)
from t2c_clip.evaluation import ReIDMetrics, evaluate_reid
from t2c_clip.model import T2CClipModel
from t2c_clip.prompts import PromptBank, PromptConfig
from t2c_clip.tfc import TFCCenterBank
from t2c_clip.training import Stage2LossConfig, Stage2LossInputs, TrainingBatch, stage2_loss_breakdown
from t2c_clip.transforms import CLIPImageTransform

DEFAULT_RANKS = (1, 5, 10)
SUPPORTED_DATASETS = ("market1501", "msmt17")

ClipLoader = Callable[[str], "CLIPLoadResult"]


@dataclass(frozen=True)
class CLIPLoadResult:
    model: torch.nn.Module
    image_processor: Any


@dataclass(frozen=True)
class JobDataConfig:
    dataset: str
    root: Path


@dataclass(frozen=True)
class CLIPReIDJobConfig:
    dataset: str
    data_root: Path
    clip_model_name: str
    batch_size: int
    num_workers: int
    lr: float
    device: torch.device
    beta: float
    context_length: int
    tfc_momentum: float
    triplet_margin: float
    tfc_weight: float


@dataclass(frozen=True)
class DatasetBundle:
    train: ReIDImageDataset
    query: ReIDImageDataset
    gallery: ReIDImageDataset
    num_train_ids: int
    num_cameras: int


@dataclass(frozen=True)
class SplitSamples:
    train: Sequence[ReIDSample]
    query: Sequence[ReIDSample]
    gallery: Sequence[ReIDSample]


@dataclass(frozen=True)
class LoaderBundle:
    train: DataLoader
    query: DataLoader
    gallery: DataLoader


@dataclass(frozen=True)
class TrainingRuntime:
    model: "CLIPReIDTrainingModel"
    loaders: LoaderBundle
    optimizer: torch.optim.Optimizer
    loss_config: Stage2LossConfig
    device: torch.device


@dataclass(frozen=True)
class ValidationRuntime:
    model: "CLIPReIDTrainingModel"
    loaders: LoaderBundle
    device: torch.device


class CLIPReIDTrainingModel(torch.nn.Module):
    def __init__(
        self,
        retrieval_model: T2CClipModel,
        classifier: torch.nn.Module,
        tfc_bank: TFCCenterBank,
    ):
        super().__init__()
        self.retrieval_model = retrieval_model
        self.classifier = classifier
        self.tfc_bank = tfc_bank


def build_training_job(
    args: Any,
    clip_loader: ClipLoader = lambda model_name: load_transformers_clip(model_name),
) -> TrainingJob:
    config = _job_config_from_args(args)
    loaded_clip = clip_loader(config.clip_model_name)
    transform = CLIPImageTransform(loaded_clip.image_processor)
    data = load_dataset_bundle(JobDataConfig(config.dataset, config.data_root), transform)
    model = _build_training_model(config, loaded_clip.model, data).to(config.device)
    loaders = _build_loaders(data, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    loss_config = Stage2LossConfig(triplet_margin=config.triplet_margin, tfc_weight=config.tfc_weight)
    training_runtime = TrainingRuntime(model, loaders, optimizer, loss_config, config.device)
    validation_runtime = ValidationRuntime(model, loaders, config.device)
    return TrainingJob(
        model=model,
        optimizer=optimizer,
        train_one_epoch=_train_one_epoch(training_runtime),
        validate=_validate(validation_runtime),
    )


def load_transformers_clip(model_name: str) -> CLIPLoadResult:
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise ImportError("transformers is required for the CLIP ReID training job") from exc
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    return CLIPLoadResult(model, getattr(processor, "image_processor", processor))


def load_dataset_bundle(config: JobDataConfig, transform) -> DatasetBundle:
    splits = _load_split_samples(config)
    _require_non_empty_splits(splits)
    camera_map = build_camera_id_map([*splits.train, *splits.query, *splits.gallery])
    train_person_map = build_person_id_map(splits.train)
    eval_person_map = build_person_id_map([*splits.query, *splits.gallery])
    return DatasetBundle(
        train=ReIDImageDataset(ReIDImageDatasetConfig(splits.train, train_person_map, camera_map, transform)),
        query=ReIDImageDataset(ReIDImageDatasetConfig(splits.query, eval_person_map, camera_map, transform)),
        gallery=ReIDImageDataset(ReIDImageDatasetConfig(splits.gallery, eval_person_map, camera_map, transform)),
        num_train_ids=len(train_person_map),
        num_cameras=len(camera_map),
    )


def _job_config_from_args(args: Any) -> CLIPReIDJobConfig:
    if args.dataset is None:
        raise ValueError("--dataset is required for t2c_clip.jobs.clip_reid")
    if args.data_root is None:
        raise ValueError("--data-root is required for t2c_clip.jobs.clip_reid")
    return CLIPReIDJobConfig(
        dataset=args.dataset,
        data_root=args.data_root,
        clip_model_name=args.clip_model_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        device=torch.device(args.device),
        beta=args.beta,
        context_length=args.context_length,
        tfc_momentum=args.tfc_momentum,
        triplet_margin=args.triplet_margin,
        tfc_weight=args.tfc_weight,
    )


def _load_split_samples(config: JobDataConfig) -> SplitSamples:
    if not config.root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {config.root}")
    if config.dataset == "market1501":
        return SplitSamples(
            train=load_market_split(config.root, "train"),
            query=load_market_split(config.root, "query"),
            gallery=load_market_split(config.root, "gallery"),
        )
    if config.dataset == "msmt17":
        return SplitSamples(
            train=load_msmt17_manifest(config.root, "train"),
            query=load_msmt17_manifest(config.root, "query"),
            gallery=load_msmt17_manifest(config.root, "gallery"),
        )
    raise ValueError(f"Unsupported dataset: {config.dataset}")


def _require_non_empty_splits(splits: SplitSamples) -> None:
    if not splits.train:
        raise ValueError("training split is empty")
    if not splits.query:
        raise ValueError("query split is empty")
    if not splits.gallery:
        raise ValueError("gallery split is empty")


def _build_training_model(
    config: CLIPReIDJobConfig,
    clip_model: torch.nn.Module,
    data: DatasetBundle,
) -> CLIPReIDTrainingModel:
    feature_dim = clip_projection_dim(clip_model)
    prompt_bank = PromptBank(
        PromptConfig(
            num_cameras=data.num_cameras,
            num_train_ids=data.num_train_ids,
            context_length=config.context_length,
            embedding_dim=feature_dim,
        )
    )
    retrieval = T2CClipModel(
        image_encoder=TransformersCLIPImageEncoder(clip_model),
        text_encoder=PromptTextEncoder(feature_dim, feature_dim),
        prompt_bank=prompt_bank,
        beta=config.beta,
    )
    classifier = torch.nn.Linear(feature_dim, data.num_train_ids)
    tfc_bank = TFCCenterBank(data.num_train_ids, feature_dim, config.tfc_momentum)
    return CLIPReIDTrainingModel(retrieval, classifier, tfc_bank)


def _build_loaders(data: DatasetBundle, config: CLIPReIDJobConfig) -> LoaderBundle:
    return LoaderBundle(
        train=_loader(data.train, config, shuffle=True),
        query=_loader(data.query, config, shuffle=False),
        gallery=_loader(data.gallery, config, shuffle=False),
    )


def _loader(dataset: ReIDImageDataset, config: CLIPReIDJobConfig, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        collate_fn=collate_reid_batches,
    )


def _train_one_epoch(runtime: TrainingRuntime):
    def train(epoch: int) -> None:
        runtime.model.train()
        for batch in runtime.loaders.train:
            training_batch = _training_batch(batch, runtime.device)
            runtime.optimizer.zero_grad()
            _update_tfc_centers(runtime.model, training_batch)
            inputs = Stage2LossInputs(runtime.model.classifier, runtime.model.tfc_bank, runtime.loss_config)
            loss = stage2_loss_breakdown(runtime.model.retrieval_model, training_batch, inputs).total
            loss.backward()
            runtime.optimizer.step()

    return train


def _validate(runtime: ValidationRuntime):
    def validate(epoch: int) -> ReIDMetrics:
        runtime.model.eval()
        query = _extract_features(runtime.model.retrieval_model, runtime.loaders.query, runtime.device)
        gallery = _extract_features(runtime.model.retrieval_model, runtime.loaders.gallery, runtime.device)
        return evaluate_reid(
            query.features,
            gallery.features,
            query.person_ids,
            gallery.person_ids,
            query.camera_ids,
            gallery.camera_ids,
            ranks=DEFAULT_RANKS,
        )

    return validate


@dataclass(frozen=True)
class FeatureSet:
    features: torch.Tensor
    person_ids: tuple[int, ...]
    camera_ids: tuple[int, ...]


def _extract_features(model: T2CClipModel, loader: DataLoader, device: torch.device) -> FeatureSet:
    feature_parts: list[torch.Tensor] = []
    person_ids: list[int] = []
    camera_ids: list[int] = []
    with torch.no_grad():
        for batch in loader:
            images = batch.images.to(device)
            cameras = batch.camera_ids.to(device)
            feature_parts.append(model.encode_retrieval(images, cameras).cpu())
            person_ids.extend(batch.original_person_ids)
            camera_ids.extend(batch.original_camera_ids)
    return FeatureSet(torch.cat(feature_parts), tuple(person_ids), tuple(camera_ids))


def _training_batch(batch: ReIDImageBatch, device: torch.device) -> TrainingBatch:
    return TrainingBatch(
        images=batch.images.to(device),
        camera_ids=batch.camera_ids.to(device),
        person_ids=batch.person_ids.to(device),
    )


def _update_tfc_centers(model: CLIPReIDTrainingModel, batch: TrainingBatch) -> None:
    with torch.no_grad():
        outputs = model.retrieval_model.forward_training(batch.images, batch.camera_ids, batch.person_ids)
        model.tfc_bank.update(outputs["retrieval"], batch.person_ids)
