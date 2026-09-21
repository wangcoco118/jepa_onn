# Interpolate SLM5 Output-Mean GPU2 Pipeline Implementation Plan

> **For agentic workers:** Execute inline in the approved checkout; do not create a worktree, commit, or start training.

**Goal:** Create a source-matched 25-epoch SLM5 output-mean experiment and GPU2 train-then-full-Test launcher.

**Architecture:** Generate YAML from the exact checkpoint configuration, allowing only four documented differences. Adapt the already verified SLM3 GPU2 file-backed Test launcher by changing only SLM-specific names and config.

**Tech Stack:** Python, PyYAML, Bash, PyTorch, CUDA.

## Global Constraints

- Preserve output_mode interpolate, five SLM layers, positive feedback at physical SLM4, and no feedback memory.
- Train 25 epochs with batch size 25 and learning rate 0.0001.
- Use output_mean with no learnable intensity offset.
- Run full Test with batch size 15 on physical GPU2.
- Do not invoke Test through Python standard input.
- Do not start training during validation.

---

### Task 1: Create and compare the YAML

**Files:**
- Create: evals/intuitive_physics/configs/onn_feedback_interpolate_slm5_output_mean_25ep.yaml

- [ ] Load the source checkpoint configuration.
- [ ] Change only epochs, readout_mode, learnable_intensity_offset, and explicit feedback_sign.
- [ ] Validate ONNConfig and recursively compare values.

### Task 2: Create and validate the launcher

**Files:**
- Create: scripts/run_interpolate_slm5_output_mean_25ep_then_test_gpu2_bs15.sh

- [ ] Adapt the verified SLM3 GPU2 launcher.
- [ ] Preserve GPU2 mapping, training batch 25, 25 epochs, and learning rate 0.0001.
- [ ] Set full Test batch size 15 with a file-backed runner.
- [ ] Require training and Test completion markers.
- [ ] Run syntax and static checks without launching training.
