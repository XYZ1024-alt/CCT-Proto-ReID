# Training Loop Checkpoints Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reusable training loop that uses tqdm progress, validates mAP every configurable number of epochs, and saves `best.pth` plus `last.pth`.

**Architecture:** Keep the loop generic and dependency-injected: callers provide `train_one_epoch` and `validate` callables. The loop owns epoch scheduling and checkpoint persistence, while validation computes real `ReIDMetrics`.

**Tech Stack:** Python 3.12, PyTorch, tqdm, standard-library `unittest`.

---

### Task 1: Training Loop Scheduling and Checkpoints

**Files:**
- Create: `tests/test_training_loop.py`
- Create: `t2c_clip/loops.py`
- Modify: `t2c_clip/__init__.py`
- Modify: `README.md`

- [ ] Write tests for default validation interval `5`, custom validation interval, tqdm wrapper usage, `best.pth`, and `last.pth`.
- [ ] Run `python -m unittest tests.test_training_loop -v` and confirm it fails because `t2c_clip.loops` does not exist.
- [ ] Implement `TrainingLoopConfig`, `TrainingLoopResult`, `EpochResult`, `run_training_loop`, and checkpoint helpers using real `torch.save`.
- [ ] Export the loop API and document the default `validation_interval=5`.
- [ ] Run full verification in the WSL `reid` environment.
