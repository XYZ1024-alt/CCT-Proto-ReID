# Train Script Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `scripts/train.py` as a real training entrypoint that wires project-specific training jobs into the reusable loop.

**Architecture:** The script parses CLI arguments and dynamically imports a user-provided `--job-builder module:function`. The builder returns concrete model, optimizer, training, and validation callables; the script then calls `run_training_loop`.

**Tech Stack:** Python 3.12, PyTorch, MLflow optional SQLite initialization, standard-library `unittest`.

---

### Task 1: Train Entrypoint

**Files:**
- Create: `scripts/train.py`
- Create: `scripts/__init__.py`
- Create: `tests/test_train_script.py`
- Modify: `README.md`

- [ ] Write RED tests for requiring `--job-builder` and executing a tiny real training job.
- [ ] Run `python -m unittest tests.test_train_script -v` and confirm it fails because `scripts.train` does not exist.
- [ ] Implement dynamic builder loading, typed `TrainingJob`, argument parsing, optional MLflow SQLite initialization, and `run_training_loop` invocation.
- [ ] Document how to start training by providing a project job builder.
- [ ] Run full verification in WSL `reid`.
