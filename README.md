# T2C-CLIP

Train2Central-CLIP foundation for Image-to-Image person ReID experiments.

The current implementation follows `docs/2026-06-27-t2c-clip-design.md` for the testable core:

- Market-1501 and MSMT17 person/camera parsing.
- Global, camera, and training-ID learnable prompt composition.
- Retrieval feature fusion: `f = normalize(f_v + beta * f_t)`.
- Training-time Feature Centralization (TFC) EMA centers and loss.
- Bidirectional image-text contrastive loss and batch-hard triplet loss.
- Stage-1 prompt alignment loss and Stage-2 CLIP/ReID/TFC loss breakdown.
- tqdm-based training loop scheduling with configurable mAP validation intervals.
- `best.pth` and `last.pth` checkpoint saving.
- MLflow tracking with a SQLite backend store.
- No-rerank cosine ReID evaluation.
- Injectable dual-stream model wiring for real CLIP image/text encoders.
- `.npz` no-rerank evaluation CLI.

## Environment

Use the WSL conda environment named `reid`:

```bash
wsl --cd /mnt/d/Code/T2C-CLIP /home/xyz10/miniconda3/bin/conda run -n reid python -m unittest discover -s tests -v
```

## Evaluate Extracted Features

The evaluator expects a `.npz` file with these arrays:

- `query_features`: float matrix `[num_query, dim]`
- `gallery_features`: float matrix `[num_gallery, dim]`
- `query_ids`: integer vector `[num_query]`
- `gallery_ids`: integer vector `[num_gallery]`
- `query_cams`: integer vector `[num_query]`
- `gallery_cams`: integer vector `[num_gallery]`

Run:

```bash
wsl --cd /mnt/d/Code/T2C-CLIP /home/xyz10/miniconda3/bin/conda run -n reid python -m t2c_clip.cli.evaluate features.npz --output metrics.json --ranks 1 5 10
```

The evaluator performs standard Image-to-Image ReID scoring without rerank. Same-identity same-camera gallery samples are excluded for each query.

## Training Loop Checkpoints

`run_training_loop` provides reusable epoch scheduling around project-specific training and validation callables:

- `TrainingLoopConfig(validation_interval=5)` validates mAP every 5 epochs by default.
- Set `validation_interval` to another positive integer to validate at a different cadence.
- The loop wraps epochs with `tqdm`.
- `last.pth` is saved after every epoch.
- `best.pth` is saved only when the latest validation mAP is better than the previous best.

The loop calls real `torch.save` and lets training, validation, tqdm, and checkpoint errors surface directly.

## MLflow SQLite Tracking

Initialize the local SQLite-backed MLflow store:

```bash
wsl --cd /mnt/d/Code/T2C-CLIP /home/xyz10/miniconda3/bin/conda run -n reid python -m t2c_clip.cli.mlflow --tracking-db mlflow/t2c_clip.db --artifact-root mlruns --experiment-name T2C-CLIP
```

Start the MLflow UI on port 6006:

```bash
wsl --cd /mnt/d/Code/T2C-CLIP /home/xyz10/miniconda3/bin/conda run -n reid mlflow ui --backend-store-uri sqlite:///mlflow/t2c_clip.db --default-artifact-root mlruns --host 127.0.0.1 --port 6006
```

The SQLite database and artifact directory are local runtime outputs and are ignored by git.

## CLIP Encoder Boundary

`T2CClipModel` requires real image and text encoder modules to be injected. It does not download weights, invent mock metrics, or return fake training success. A Stage-1 or Stage-2 trainer should pass concrete CLIP-compatible encoders into this model and let dependency or checkpoint failures surface directly.
