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
- Transformers CLIP-backed Market-1501/MSMT17 training job builder.
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

## Train Entrypoint

`scripts/train.py` wires a project-specific training job into `run_training_loop`.
The recommended real training builder is `t2c_clip.jobs.clip_reid:build_training_job`.
It loads Market-1501 or MSMT17 images, wraps a Transformers CLIP image encoder, trains the T2C prompt/fusion branch, evaluates no-rerank mAP, and lets dependency, data, and checkpoint failures surface directly.

Market-1501 expects the standard directories under `--data-root`:

- `bounding_box_train/`
- `query/`
- `bounding_box_test/`

MSMT17 expects the standard manifests and image folders under `--data-root`:

- `list_train.txt`
- `list_query.txt`
- `list_gallery.txt`
- `train/`
- `test/`

Run Stage-2 training:

```bash
wsl --cd /mnt/d/Code/T2C-CLIP /home/xyz10/miniconda3/bin/conda run -n reid python scripts/train.py \
  --job-builder t2c_clip.jobs.clip_reid:build_training_job \
  --dataset msmt17 \
  --data-root MSMT17_V1 \
  --clip-model-name openai/clip-vit-base-patch16 \
  --epochs 120 \
  --validation-interval 5 \
  --checkpoint-dir checkpoints \
  --batch-size 64 \
  --num-workers 4 \
  --lr 0.0001 \
  --device cuda
```

Useful training arguments:

- `--dataset market1501|msmt17`
- `--data-root PATH`
- `--clip-model-name openai/clip-vit-base-patch16`
- `--batch-size 64`
- `--num-workers 4`
- `--lr 0.0001`
- `--device cuda`
- `--beta 0.1`
- `--context-length 4`
- `--tfc-momentum 0.5`
- `--triplet-margin 0.3`
- `--tfc-weight 1.0`

Add `--enable-mlflow` to initialize the SQLite-backed MLflow store before training:

```bash
wsl --cd /mnt/d/Code/T2C-CLIP /home/xyz10/miniconda3/bin/conda run -n reid python scripts/train.py \
  --job-builder t2c_clip.jobs.clip_reid:build_training_job \
  --dataset msmt17 \
  --data-root MSMT17_V1 \
  --epochs 120 \
  --validation-interval 5 \
  --checkpoint-dir checkpoints \
  --enable-mlflow \
  --tracking-db mlflow/t2c_clip.db \
  --artifact-root mlruns \
  --experiment-name T2C-CLIP \
  --run-name msmt17-stage2
```

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

`T2CClipModel` requires concrete encoder modules to be injected. The recommended `clip_reid` builder uses a real Transformers CLIP image encoder and a learnable prompt projection module for the text branch. It does not invent mock metrics, return fake training success, silently switch devices, or hide missing dependency and dataset errors.
