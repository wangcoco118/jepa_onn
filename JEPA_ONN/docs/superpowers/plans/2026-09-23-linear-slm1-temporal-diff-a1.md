# Linear SLM1 Temporal-Difference Alpha 1 Implementation Plan

> **For agentic workers:** Execute this task with a red-green verification cycle.

**Goal:** Add a guarded fresh-training pipeline for SLM1 alpha 1 followed by resumable complete Test.

**Architecture:** A Bash launcher extracts only configuration from the alpha 0.5 best checkpoint, validates the source identity, changes alpha and identifiers, starts fresh training, and runs complete Test from the newly trained validation-best checkpoint.

**Tech Stack:** Bash, Python 3.10, PyTorch, YAML, JEPA_ONN IntPhys evaluators.

## Global Constraints

- Source experiment remains read-only.
- Never pass --resume for training.
- Preserve SLM1, feedback index 0, and every non-alpha model setting.
- Train on physical GPU0 for 25 epochs with batch size 25.
- Complete Test uses batch size 15 and rolling resume.
- Do not launch training during implementation.

---

### Task 1: Add and verify the SLM1 alpha 1 pipeline

**Files:**
- Modify: tests/test_training_pipeline_scripts.py
- Create: scripts/run_linear_slm1_temporal_diff_a1_25ep_bs25_then_test_gpu0_bs15.sh

- [x] Add a contract test and confirm failure while the launcher is absent.
- [x] Validate source SLM1, feedback index 0, empty intervals, and alpha 0.5.
- [x] Change only alpha to 1.0 plus identifying metadata.
- [x] Start fresh training without --resume.
- [x] Run resumable complete O1/O2/O3 Test at batch size 15.
- [x] Require RUN_DONE markers, result artifacts, and 12,960 answer rows.
- [x] Run contract tests and Bash syntax validation.
