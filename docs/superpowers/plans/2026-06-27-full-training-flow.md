# Full Training Flow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a real `t2c_clip.jobs.clip_reid:build_training_job` training path that can train T2C-CLIP on Market-1501 or MSMT17 with Transformers CLIP, validation mAP, and existing checkpoints.

**Architecture:** Keep `scripts/train.py` as the generic runner and put project-specific training wiring in `t2c_clip.jobs.clip_reid`. Add small focused modules for image datasets, CLIP preprocessing, and Transformers CLIP wrappers so each piece is independently testable.

**Tech Stack:** Python, PyTorch, Transformers CLIP, Pillow, unittest, WSL conda env `reid`.

---

### Task 1: Training CLI Arguments

**Files:**
- Modify: `scripts/train.py`
- Modify: `tests/test_train_script.py`

- [ ] **Step 1: Write the failing test**

```python
def test_main_passes_project_training_args_to_builder(self):
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint_dir = Path(tmp) / "checkpoints"
        exit_code = main(
            [
                "--job-builder",
                "tests.test_train_script:recording_training_job",
                "--epochs",
                "1",
                "--validation-interval",
                "1",
                "--checkpoint-dir",
                str(checkpoint_dir),
                "--dataset",
                "msmt17",
                "--data-root",
                "MSMT17_V1",
                "--clip-model-name",
                "openai/clip-vit-base-patch16",
                "--batch-size",
                "8",
                "--num-workers",
                "2",
                "--lr",
                "0.001",
                "--device",
                "cpu",
            ],
            progress_factory=lambda iterable, **kwargs: iterable,
        )

    self.assertEqual(exit_code, 0)
    self.assertEqual(RECORDED_ARGS.dataset, "msmt17")
    self.assertEqual(RECORDED_ARGS.data_root, Path("MSMT17_V1"))
    self.assertEqual(RECORDED_ARGS.clip_model_name, "openai/clip-vit-base-patch16")
    self.assertEqual(RECORDED_ARGS.batch_size, 8)
    self.assertEqual(RECORDED_ARGS.num_workers, 2)
    self.assertEqual(RECORDED_ARGS.lr, 0.001)
    self.assertEqual(RECORDED_ARGS.device, "cpu")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `wsl --cd /mnt/d/Code/T2C-CLIP /home/xyz10/miniconda3/bin/conda run -n reid python -m unittest tests.test_train_script.TrainScriptTest.test_main_passes_project_training_args_to_builder -v`

Expected: FAIL because `scripts/train.py` does not recognize the new arguments.

- [ ] **Step 3: Write minimal implementation**

Add parser arguments:

```python
parser.add_argument("--dataset", choices=("market1501", "msmt17"))
parser.add_argument("--data-root", type=Path)
parser.add_argument("--clip-model-name", default="openai/clip-vit-base-patch16")
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--num-workers", type=int, default=4)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--device", default="cuda")
parser.add_argument("--beta", type=float, default=0.1)
parser.add_argument("--context-length", type=int, default=4)
parser.add_argument("--tfc-momentum", type=float, default=0.5)
parser.add_argument("--triplet-margin", type=float, default=0.3)
parser.add_argument("--tfc-weight", type=float, default=1.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run the same unittest command. Expected: PASS.

### Task 2: ReID Image Dataset

**Files:**
- Create: `t2c_clip/datasets.py`
- Create: `tests/test_datasets.py`

- [ ] **Step 1: Write failing dataset tests**

```python
def test_reid_image_dataset_returns_remapped_and_original_metadata(self):
    with tempfile.TemporaryDirectory() as tmp:
        image_path = Path(tmp) / "0002_c3s1_000551_01.jpg"
        Image.new("RGB", (2, 2), color="red").save(image_path)
        sample = ReIDSample(image_path, 42, 7, "market1501", "train")
        dataset = ReIDImageDataset([sample], {42: 0}, {7: 0}, tensor_transform)

        batch = dataset[0]

    self.assertTrue(torch.equal(batch.image, torch.ones(3, 2, 2)))
    self.assertEqual(batch.person_id, 0)
    self.assertEqual(batch.camera_id, 0)
    self.assertEqual(batch.original_person_id, 42)
    self.assertEqual(batch.original_camera_id, 7)
```

```python
def test_build_index_maps_sorts_unique_values(self):
    samples = [
        ReIDSample(Path("a.jpg"), 9, 3, "market1501", "train"),
        ReIDSample(Path("b.jpg"), 4, 1, "market1501", "train"),
    ]

    self.assertEqual(build_person_id_map(samples), {4: 0, 9: 1})
    self.assertEqual(build_camera_id_map(samples), {1: 0, 3: 1})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `wsl --cd /mnt/d/Code/T2C-CLIP /home/xyz10/miniconda3/bin/conda run -n reid python -m unittest tests.test_datasets -v`

Expected: FAIL because `t2c_clip.datasets` does not exist.

- [ ] **Step 3: Implement dataset module**

Create frozen `ReIDImageBatch`, `ReIDImageDataset`, `build_person_id_map`, `build_camera_id_map`, and `collate_reid_batches`.

- [ ] **Step 4: Run tests to verify they pass**

Run the same unittest command. Expected: PASS.

### Task 3: CLIP Transform And Backbones

**Files:**
- Create: `t2c_clip/transforms.py`
- Create: `t2c_clip/clip_backbone.py`
- Create: `tests/test_clip_training_components.py`

- [ ] **Step 1: Write failing component tests**

```python
def test_clip_image_transform_returns_first_pixel_values_row(self):
    processor = FakeImageProcessor()
    transform = CLIPImageTransform(processor)

    output = transform(Image.new("RGB", (2, 2)))

    self.assertTrue(torch.equal(output, torch.ones(3, 2, 2)))
    self.assertEqual(processor.call_count, 1)
```

```python
def test_transformers_clip_image_encoder_calls_get_image_features(self):
    clip = FakeCLIP(projection_dim=2)
    encoder = TransformersCLIPImageEncoder(clip)

    output = encoder(torch.ones(2, 3, 2, 2))

    self.assertTrue(torch.equal(output, torch.full((2, 2), 2.0)))
    self.assertTrue(clip.image_called)
```

```python
def test_prompt_text_encoder_projects_mean_prompt(self):
    encoder = PromptTextEncoder(prompt_embedding_dim=3, output_dim=2)
    with torch.no_grad():
        encoder.projection.weight.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
        encoder.projection.bias.zero_()

    output = encoder(torch.tensor([[[1.0, 2.0, 9.0], [3.0, 4.0, 9.0]]]))

    self.assertTrue(torch.equal(output, torch.tensor([[2.0, 3.0]])))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `wsl --cd /mnt/d/Code/T2C-CLIP /home/xyz10/miniconda3/bin/conda run -n reid python -m unittest tests.test_clip_training_components -v`

Expected: FAIL because modules do not exist.

- [ ] **Step 3: Implement transform and backbones**

`CLIPImageTransform` wraps a CLIP image processor. `TransformersCLIPImageEncoder` wraps `get_image_features`. `PromptTextEncoder` mean-pools prompt embeddings and projects to CLIP output dimension.

- [ ] **Step 4: Run tests to verify they pass**

Run the same unittest command. Expected: PASS.

### Task 4: CLIP ReID Job Builder

**Files:**
- Create: `t2c_clip/jobs/__init__.py`
- Create: `t2c_clip/jobs/clip_reid.py`
- Create: `tests/test_clip_reid_job.py`

- [ ] **Step 1: Write failing job tests**

```python
def test_load_dataset_bundle_rejects_missing_root(self):
    with self.assertRaises(FileNotFoundError):
        load_dataset_bundle(JobDataConfig("market1501", Path("missing")), tensor_transform)
```

```python
def test_build_training_job_returns_real_callbacks_with_fake_clip(self):
    with tempfile.TemporaryDirectory() as tmp:
        root = build_market_fixture(Path(tmp))
        args = training_args(root)
        job = build_training_job(args, clip_loader=load_fake_clip)

        job.train_one_epoch(1)
        metrics = job.validate(1)

    self.assertGreaterEqual(metrics.map, 0.0)
    self.assertIn(1, metrics.cmc)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `wsl --cd /mnt/d/Code/T2C-CLIP /home/xyz10/miniconda3/bin/conda run -n reid python -m unittest tests.test_clip_reid_job -v`

Expected: FAIL because `t2c_clip.jobs.clip_reid` does not exist.

- [ ] **Step 3: Implement job builder**

Implement dataset loading, shared ID/camera maps, DataLoaders, CLIP loader, model construction, optimizer, `train_one_epoch`, `validate`, and feature extraction helpers.

- [ ] **Step 4: Run tests to verify they pass**

Run the same unittest command. Expected: PASS.

### Task 5: Documentation And Full Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README**

Document the recommended training command using `t2c_clip.jobs.clip_reid:build_training_job`, dataset root expectations, MLflow UI on port `6006`, and no-fallback failure behavior.

- [ ] **Step 2: Run all verification**

Run:

```bash
wsl --cd /mnt/d/Code/T2C-CLIP /home/xyz10/miniconda3/bin/conda run -n reid python -m unittest discover -s tests -v
wsl --cd /mnt/d/Code/T2C-CLIP /home/xyz10/miniconda3/bin/conda run -n reid python -m compileall -q t2c_clip tests scripts
```

Expected: all tests pass and compileall exits with code 0.
