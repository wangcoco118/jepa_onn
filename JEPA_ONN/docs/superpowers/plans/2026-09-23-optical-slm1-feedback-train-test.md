# Optical SLM1 Feedback Train-Test Implementation Plan

> **For agentic workers:** Execute the two tasks in order and verify every command before launching training.

**Goal:** Add an isolated one-SLM feedback experiment derived from the existing three-SLM optical-output experiment, then run complete O1/O2/O3 Test automatically on physical GPU1.

**Architecture:** A dedicated YAML preserves all source settings except the ONN core depth, feedback index, core inter-SLM distances, and identifying tags. A dedicated shell launcher trains first, verifies the best checkpoint and RUN_DONE marker, then generates and runs a guarded real Python Test runner and validates all 12,960-row outputs.

**Tech Stack:** YAML, Bash, Python 3.10, PyTorch, JEPA_ONN IntPhys evaluators.

## Global Constraints

- Preserve the existing SLM3 YAML and launcher unchanged.
- ONN core: one SLM, feedback enabled on index 0, no core inter-SLM distance.
- Optical 384-to-1024 output mapper remains two SLM layers with one 8000 um interval.
- Train for 25 epochs with batch size 15.
- Complete Test uses O1/O2/O3 with batch size 15.
- Physical GPU1 is exposed as logical cuda:0.
- Do not launch training during implementation verification.

---

### Task 1: Add the SLM1 configuration contract

**Files:**
- Modify: tests/test_optical_output_mapper.py
- Create: evals/intuitive_physics/configs/onn_feedback_optical_slm1_feedback_slm1_25ep_bs15.yaml

- [x] Add a test that loads the YAML and checks the ONN core, output mapper, batch size, epochs, and feedback gain.
- [x] Run the test and confirm it fails because the YAML is absent.
- [x] Create the YAML by copying the SLM3 configuration and changing only the approved structure and identifiers.
- [x] Run the focused test and confirm it passes.

### Task 2: Add and validate the train-to-Test launcher

**Files:**
- Create: scripts/run_optical_slm1_feedback_slm1_25ep_then_test_gpu1_bs15.sh

- [x] Follow the existing guarded SLM3 train-to-Test launcher.
- [x] Select physical GPU1 and retain logical --gpu 0.
- [x] Train 25 epochs at batch size 15 and skip the built-in final eval.
- [x] After training RUN_DONE, run complete O1/O2/O3 Test at batch size 15.
- [x] Require both answer files, the losses file, and exactly 12,960 rows per answer.
- [x] Validate with bash -n, YAML loading, configuration construction, and Python compilation without launching training.
