# Interpolate SLM1 Output-Mean GPU3 Pipeline Implementation Plan

> **For agentic workers:** Execute inline in the approved checkout; do not create a worktree, commit, or start training.

**Goal:** Build a source-matched SLM1 output-mean configuration and GPU3 train-then-full-Test launcher.

**Architecture:** Derive YAML from the source checkpoint, permitting only the three readout/metadata differences. Adapt the verified SLM3 file-backed Test launcher to SLM1 and physical GPU3.

**Tech Stack:** Python, PyYAML, Bash, PyTorch, CUDA.

## Global Constraints

- Preserve output_mode interpolate, one SLM layer, positive feedback at physical SLM1, and no feedback memory.
- Train 25 epochs with batch size 25 and learning rate 0.0001.
- Use output_mean with learnable_intensity_offset false.
- Run full Test with batch size 15 on physical GPU3.
- Never invoke Test through Python standard input.
- Do not start training while validating.

---

### Task 1: Generate and compare the YAML

**Files:**
- Create: evals/intuitive_physics/configs/onn_feedback_interpolate_slm1_output_mean_25ep.yaml

- [ ] Load the exact source checkpoint configuration.
- [ ] Change only readout_mode, learnable_intensity_offset, and explicit feedback_sign.
- [ ] Validate ONNConfig and recursively compare all values.

### Task 2: Generate and validate the launcher

**Files:**
- Create: scripts/run_interpolate_slm1_output_mean_25ep_then_test_gpu3_bs15.sh

- [ ] Adapt the verified file-backed SLM3 launcher.
- [ ] Bind physical GPU3 as logical cuda:0.
- [ ] Preserve training batch 25, 25 epochs, and learning rate 0.0001.
- [ ] Set full Test batch size 15 and require both completion markers.
- [ ] Run bash syntax and embedded runner compilation checks without launching training.
