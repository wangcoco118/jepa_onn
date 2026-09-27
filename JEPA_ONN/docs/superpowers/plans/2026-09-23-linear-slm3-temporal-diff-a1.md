# Linear SLM3 Temporal-Difference Alpha 1 Implementation Plan

> **For agentic workers:** Execute this single task with a red-green verification cycle.

**Goal:** Add a guarded fresh-training pipeline for alpha 1.0 followed by complete IntPhys Test.

**Architecture:** A Bash launcher reads only the effective configuration from the alpha 0.5 source checkpoint, changes alpha and identifying metadata, starts fresh training, locates the best new checkpoint, and runs complete Test. Guard checks prevent Test from starting after failed or incomplete training.

**Tech Stack:** Bash, Python 3.10, PyTorch, YAML, JEPA_ONN IntPhys evaluators.

## Global Constraints

- Source experiment is read-only.
- Do not pass --resume.
- Change only temporal difference alpha from 0.5 to 1.0 plus output identifiers.
- Training uses physical GPU2, 25 epochs, and batch size 25.
- Test uses O1/O2/O3 and batch size 10.
- Do not launch training during implementation.

---

### Task 1: Add and verify the fresh train-to-Test pipeline

**Files:**
- Create: tests/test_training_pipeline_scripts.py
- Create: scripts/run_linear_slm3_temporal_diff_a1_25ep_bs25_then_test_gpu2_bs10.sh

- [x] Write a script contract test and confirm it fails because the launcher is absent.
- [x] Create the launcher with source-checkpoint configuration extraction and alpha 1.0.
- [x] Require no resume argument and verify source SLM3, feedback index 1, and source alpha 0.5.
- [x] Train with batch size 25 on physical GPU2.
- [x] Run complete Test with batch size 10 from the validation-best checkpoint.
- [x] Validate RUN_DONE markers, required artifacts, and 12,960 answer rows.
- [x] Run the contract test and bash syntax check.
