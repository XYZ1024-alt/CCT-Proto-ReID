# T2C-CLIP Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the testable engineering foundation described by `docs/2026-06-27-t2c-clip-design.md`: dataset camera parsing, prompt composition, fusion, TFC, losses, no-rerank evaluation, and injectable CLIP-style model wiring.

**Architecture:** Keep CLIP-specific loading behind dependency injection so the core method is testable without downloading weights. The package exposes small PyTorch modules/functions for each research component and refuses invalid protocol states with explicit errors.

**Tech Stack:** Python 3.12, PyTorch, NumPy, standard-library `unittest` in the WSL `reid` conda environment.

---

### Task 1: Dataset Protocol Parsers

**Files:**
- Create: `tests/test_data.py`
- Create: `t2c_clip/data.py`
- Modify: `t2c_clip/__init__.py`

- [ ] **Step 1: Write failing tests**

```python
from pathlib import Path
import tempfile
import unittest

from t2c_clip.data import (
    ReIDSample,
    load_market_split,
    load_msmt17_manifest,
    parse_market_filename,
    parse_msmt17_filename,
)


class DataParsingTest(unittest.TestCase):
    def test_parse_market_filename_reads_person_and_camera(self):
        self.assertEqual(parse_market_filename("0002_c3s1_000551_01.jpg"), (2, 3))

    def test_parse_msmt17_filename_reads_camera_token(self):
        self.assertEqual(parse_msmt17_filename("0000_045_12_0303morning_0006_2.jpg"), 12)

    def test_load_market_split_skips_junk_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            split_dir = Path(tmp) / "bounding_box_train"
            split_dir.mkdir()
            (split_dir / "0002_c3s1_000551_01.jpg").touch()
            (split_dir / "-1_c1s1_000401_03.jpg").touch()

            samples = load_market_split(Path(tmp), "train")

        self.assertEqual(samples, [ReIDSample(split_dir / "0002_c3s1_000551_01.jpg", 2, 3, "market1501", "train")])

    def test_load_msmt17_manifest_uses_manifest_label_and_camera(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "list_query.txt").write_text("0000/0000_000_01_0303morning_0015_0.jpg 7\n", encoding="utf-8")

            samples = load_msmt17_manifest(root, "query")

        expected_path = root / "test" / "0000" / "0000_000_01_0303morning_0015_0.jpg"
        self.assertEqual(samples, [ReIDSample(expected_path, 7, 1, "msmt17", "query")])
```

- [ ] **Step 2: Run tests to verify failure**

Run: `/home/xyz10/miniconda3/bin/conda run -n reid python -m unittest tests.test_data -v`

Expected: FAIL with `ModuleNotFoundError: No module named 't2c_clip'`.

- [ ] **Step 3: Implement parser module**

Implement `ReIDSample`, `parse_market_filename`, `parse_msmt17_filename`, `load_market_split`, and `load_msmt17_manifest` with explicit `ValueError` for unsupported file names and split names.

- [ ] **Step 4: Run tests to verify pass**

Run: `/home/xyz10/miniconda3/bin/conda run -n reid python -m unittest tests.test_data -v`

Expected: PASS.

### Task 2: Prompt Composition and Fusion

**Files:**
- Create: `tests/test_features_prompts.py`
- Create: `t2c_clip/features.py`
- Create: `t2c_clip/prompts.py`
- Modify: `t2c_clip/__init__.py`

- [ ] **Step 1: Write failing tests**

```python
import unittest
import torch

from t2c_clip.features import fuse_features, l2_normalize
from t2c_clip.prompts import PromptBank, PromptConfig


class FeaturePromptTest(unittest.TestCase):
    def test_l2_normalize_returns_unit_rows(self):
        output = l2_normalize(torch.tensor([[3.0, 4.0]]))
        self.assertTrue(torch.allclose(output.norm(dim=1), torch.ones(1)))

    def test_fuse_features_uses_beta_and_normalizes(self):
        visual = torch.tensor([[1.0, 0.0]])
        text = torch.tensor([[0.0, 1.0]])
        fused = fuse_features(visual, text, beta=1.0)
        expected = torch.tensor([[2 ** -0.5, 2 ** -0.5]])
        self.assertTrue(torch.allclose(fused, expected))

    def test_prompt_bank_training_adds_identity_prompt(self):
        bank = PromptBank(PromptConfig(num_cameras=2, num_train_ids=3, context_length=2, embedding_dim=2))
        with torch.no_grad():
            bank.global_prompt.fill_(1.0)
            bank.camera_prompts.zero_()
            bank.identity_prompts.zero_()
            bank.camera_prompts[1].fill_(2.0)
            bank.identity_prompts[2].fill_(4.0)

        prompt = bank.training_prompts(torch.tensor([1]), torch.tensor([2]))

        self.assertTrue(torch.equal(prompt, torch.full((1, 2, 2), 7.0)))

    def test_prompt_bank_inference_excludes_identity_prompt(self):
        bank = PromptBank(PromptConfig(num_cameras=2, num_train_ids=3, context_length=1, embedding_dim=2))
        with torch.no_grad():
            bank.global_prompt.fill_(1.0)
            bank.camera_prompts[1].fill_(2.0)
            bank.identity_prompts[2].fill_(4.0)

        prompt = bank.inference_prompts(torch.tensor([1]))

        self.assertTrue(torch.equal(prompt, torch.full((1, 1, 2), 3.0)))
```

- [ ] **Step 2: Run tests to verify failure**

Run: `/home/xyz10/miniconda3/bin/conda run -n reid python -m unittest tests.test_features_prompts -v`

Expected: FAIL because `t2c_clip.features` and `t2c_clip.prompts` do not exist.

- [ ] **Step 3: Implement feature and prompt modules**

Implement row-wise normalization, scalar feature fusion, immutable prompt config, and a `torch.nn.Module` prompt bank with separate training and inference composition.

- [ ] **Step 4: Run tests to verify pass**

Run: `/home/xyz10/miniconda3/bin/conda run -n reid python -m unittest tests.test_features_prompts -v`

Expected: PASS.

### Task 3: TFC and Losses

**Files:**
- Create: `tests/test_tfc_losses.py`
- Create: `t2c_clip/tfc.py`
- Create: `t2c_clip/losses.py`
- Modify: `t2c_clip/__init__.py`

- [ ] **Step 1: Write failing tests**

```python
import unittest
import torch

from t2c_clip.losses import batch_hard_triplet_loss, bidirectional_contrastive_loss
from t2c_clip.tfc import TFCCenterBank


class TFCLossTest(unittest.TestCase):
    def test_tfc_updates_identity_centers_with_ema(self):
        bank = TFCCenterBank(num_train_ids=2, feature_dim=2, momentum=0.5)
        bank.update(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([0, 1]))
        bank.update(torch.tensor([[0.0, 1.0]]), torch.tensor([0]))

        expected = torch.tensor([2 ** -0.5, 2 ** -0.5])
        self.assertTrue(torch.allclose(bank.centers[0], expected, atol=1e-6))

    def test_tfc_loss_uses_existing_centers(self):
        bank = TFCCenterBank(num_train_ids=1, feature_dim=2, momentum=0.5)
        bank.update(torch.tensor([[1.0, 0.0]]), torch.tensor([0]))

        loss = bank.loss(torch.tensor([[0.0, 1.0]]), torch.tensor([0]))

        self.assertTrue(torch.allclose(loss, torch.tensor(1.0)))

    def test_bidirectional_contrastive_loss_is_low_for_matching_pairs(self):
        image = torch.eye(2)
        text = torch.eye(2)
        loss = bidirectional_contrastive_loss(image, text, logit_scale=10.0)
        self.assertLess(float(loss), 0.01)

    def test_batch_hard_triplet_loss_penalizes_close_negative(self):
        features = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]])
        labels = torch.tensor([0, 0, 1])
        loss = batch_hard_triplet_loss(features, labels, margin=0.3)
        self.assertGreater(float(loss), 0.0)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `/home/xyz10/miniconda3/bin/conda run -n reid python -m unittest tests.test_tfc_losses -v`

Expected: FAIL because `t2c_clip.tfc` and `t2c_clip.losses` do not exist.

- [ ] **Step 3: Implement TFC and loss modules**

Implement EMA center updates without gradient tracking, explicit missing-center errors, dual contrastive loss, and batch-hard triplet loss.

- [ ] **Step 4: Run tests to verify pass**

Run: `/home/xyz10/miniconda3/bin/conda run -n reid python -m unittest tests.test_tfc_losses -v`

Expected: PASS.

### Task 4: No-Rerank Evaluation and Model Wiring

**Files:**
- Create: `tests/test_evaluation_model.py`
- Create: `t2c_clip/evaluation.py`
- Create: `t2c_clip/model.py`
- Modify: `t2c_clip/__init__.py`

- [ ] **Step 1: Write failing tests**

```python
import unittest
import torch

from t2c_clip.evaluation import evaluate_reid
from t2c_clip.model import T2CClipModel
from t2c_clip.prompts import PromptBank, PromptConfig


class IdentityEncoder(torch.nn.Module):
    def forward(self, inputs):
        return inputs


class PromptMeanEncoder(torch.nn.Module):
    def forward(self, prompts):
        return prompts.mean(dim=1)


class EvaluationModelTest(unittest.TestCase):
    def test_evaluate_reid_excludes_same_camera_matches(self):
        query = torch.tensor([[1.0, 0.0]])
        gallery = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        metrics = evaluate_reid(query, gallery, [1], [1, 2], [1], [1, 2], ranks=(1,))
        self.assertEqual(metrics.map, 0.0)
        self.assertEqual(metrics.cmc[1], 0.0)

    def test_evaluate_reid_reports_rank1_and_map(self):
        query = torch.tensor([[1.0, 0.0]])
        gallery = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        metrics = evaluate_reid(query, gallery, [1], [2, 1], [1], [2, 2], ranks=(1,))
        self.assertEqual(metrics.map, 0.5)
        self.assertEqual(metrics.cmc[1], 0.0)

    def test_model_inference_uses_global_and_camera_prompts(self):
        prompt_bank = PromptBank(PromptConfig(num_cameras=1, num_train_ids=1, context_length=1, embedding_dim=2))
        with torch.no_grad():
            prompt_bank.global_prompt.zero_()
            prompt_bank.camera_prompts[0] = torch.tensor([[0.0, 1.0]])
            prompt_bank.identity_prompts[0] = torch.tensor([[10.0, 0.0]])
        model = T2CClipModel(IdentityEncoder(), PromptMeanEncoder(), prompt_bank, beta=1.0)

        output = model.encode_retrieval(torch.tensor([[1.0, 0.0]]), torch.tensor([0]))

        expected = torch.tensor([[2 ** -0.5, 2 ** -0.5]])
        self.assertTrue(torch.allclose(output, expected))
```

- [ ] **Step 2: Run tests to verify failure**

Run: `/home/xyz10/miniconda3/bin/conda run -n reid python -m unittest tests.test_evaluation_model -v`

Expected: FAIL because `t2c_clip.evaluation` and `t2c_clip.model` do not exist.

- [ ] **Step 3: Implement evaluation and model modules**

Implement cosine retrieval metrics with Market/MSMT-style same-ID same-camera exclusion, and an injectable dual-encoder wrapper that uses inference prompts without ID prompts.

- [ ] **Step 4: Run tests to verify pass**

Run: `/home/xyz10/miniconda3/bin/conda run -n reid python -m unittest tests.test_evaluation_model -v`

Expected: PASS.

### Task 5: CLI Documentation

**Files:**
- Create: `README.md`

- [ ] **Step 1: Document real execution status**

Document the WSL `reid` commands, implemented modules, and the explicit boundary that CLIP weight loading/training loops require a real encoder adapter and must not return mock metrics.

- [ ] **Step 2: Run full verification**

Run: `/home/xyz10/miniconda3/bin/conda run -n reid python -m unittest discover -s tests -v`

Expected: PASS.
