# MLflow SQLite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add real MLflow tracking support using a SQLite backend and a documented UI command pinned to port 6006.

**Architecture:** Keep MLflow integration in a dedicated module. Training code can call the helper explicitly, while core model/loss modules remain free of global MLflow state.

**Tech Stack:** Python 3.12, MLflow 3.14.0 in the WSL `reid` conda environment, SQLite backend store, standard-library `unittest`.

---

### Task 1: MLflow SQLite Helper

**Files:**
- Create: `tests/test_mlflow_sqlite.py`
- Create: `t2c_clip/mlflow.py`
- Modify: `t2c_clip/__init__.py`

- [ ] Write tests for SQLite URI construction, real MLflow initialization, and the default UI command port.
- [ ] Run `python -m unittest tests.test_mlflow_sqlite -v` and confirm it fails because the helper does not exist.
- [ ] Implement immutable config, SQLite tracking URI creation, artifact URI creation, real experiment/run initialization, and UI command construction.
- [ ] Run `python -m unittest tests.test_mlflow_sqlite -v` and confirm it passes.

### Task 2: MLflow CLI and Docs

**Files:**
- Create: `t2c_clip/cli/mlflow.py`
- Modify: `tests/test_mlflow_sqlite.py`
- Modify: `.gitignore`
- Modify: `README.md`

- [ ] Write a CLI test that initializes a temporary SQLite store and emits JSON containing a `--port 6006` UI command.
- [ ] Run the CLI test and confirm it fails because the CLI does not exist.
- [ ] Implement the CLI with explicit arguments for tracking DB, artifact root, experiment name, run name, output path, host, and port.
- [ ] Ignore MLflow SQLite databases and artifact directories.
- [ ] Document initialization and UI startup with `--port 6006`.
- [ ] Run the full test suite and compile check in WSL `reid`.
