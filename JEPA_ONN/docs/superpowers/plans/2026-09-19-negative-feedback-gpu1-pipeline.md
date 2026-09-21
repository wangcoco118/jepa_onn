# Negative Feedback GPU1 Pipeline Implementation Plan

> **For agentic workers:** Execute inline in the current approved checkout; do not create a worktree, commit, or start training.

**Goal:** Add an auditable 25-epoch negative-feedback training configuration and a GPU1 pipeline that automatically runs full O1/O2/O3 Test inference with batch size 15.

**Architecture:** Create one independent YAML derived from the existing source-matched 25-epoch configuration, restoring the original learnable-offset detector readout and changing only feedback polarity. Create one launcher based on the existing GPU0 pipeline, changing physical GPU binding, names, and Test batch size.

**Tech Stack:** Bash, YAML, Python, PyTorch/CUDA.

## Global Constraints

- Training parameters remain source-compatible: 25 epochs, batch size 25, learning rate 0.0001, linear output head, five SLM layers, single feedback at physical SLM4, no feedback memory.
- Detector readout remains `readout_mode: learnable_offset` with `learnable_intensity_offset: true`.
- Negative feedback is selected only by `feedback_sign: -1.0`.
- Physical GPU1 maps to program-visible logical `cuda:0` through `CUDA_VISIBLE_DEVICES=1`.
- Full Test runs only after successful training and uses batch size 15.
- Do not start training during implementation verification.

---

### Task 1: Create negative-feedback training configuration

**Files:**
- Create: `evals/intuitive_physics/configs/onn_feedback_linear_slm5_negative_25ep.yaml`

- [ ] Copy the existing source-matched 25-epoch configuration.
- [ ] Set `readout_mode: learnable_offset`, `learnable_intensity_offset: true`, and `feedback_sign: -1.0`.
- [ ] Verify all other YAML values are unchanged.

### Task 2: Create GPU1 train-then-Test launcher

**Files:**
- Create: `scripts/run_linear_slm5_negative_25ep_then_test_gpu1_bs15.sh`

- [ ] Adapt the existing launcher to use the new configuration.
- [ ] Set `CUDA_VISIBLE_DEVICES=1`, retain `--gpu 0`, train for 25 epochs with batch size 25, and Test with batch size 15.
- [ ] Keep unique output paths, per-user temporary/cache paths, checkpoint checks, and training/Test RUN_DONE checks.
- [ ] Run `bash -n` and static assertions without launching training.
