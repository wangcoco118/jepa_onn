# Linear SLM1 Negative-Feedback Implementation Plan

> **For agentic workers:** Execute this task with a red-green verification cycle.

**Goal:** Add a guarded fresh-training pipeline for one-SLM negative feedback followed by complete IntPhys Test.

**Architecture:** A Bash launcher extracts only configuration from the source SLM3 negative-feedback best checkpoint, validates its identity, applies the approved SLM1 structure changes, starts fresh training, and runs complete Test from the newly trained validation-best checkpoint.

**Tech Stack:** Bash, Python 3.10, PyTorch, YAML, JEPA_ONN IntPhys evaluators.

## Global Constraints

- Source experiment remains read-only.
- Do not pass --resume or load source model weights.
- Preserve feedback_sign=-1 and all non-structural model settings.
- Train on physical GPU3 for 25 epochs with batch size 25.
- Run complete Test with batch size 15.
- Do not launch training during implementation.

---

### Task 1: Add and verify the SLM1 negative-feedback pipeline

**Files:**
- Modify: tests/test_training_pipeline_scripts.py
- Create: scripts/run_linear_slm1_negative_25ep_bs25_then_test_gpu3_bs15.sh

- [x] Add a contract test and confirm failure while the launcher is absent.
- [x] Validate the source is SLM3, feedback index 1, and feedback sign -1.
- [x] Generate a new config with SLM1, feedback index 0, and empty inter-SLM distances.
- [x] Start fresh training without --resume.
- [x] Run complete O1/O2/O3 Test after verified training completion.
- [x] Require RUN_DONE markers, result artifacts, and 12,960 answer rows.
- [x] Run the contract test and Bash syntax check.
