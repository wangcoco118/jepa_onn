# Linear SLM3 Negative-Feedback GPU0 Pipeline Implementation Plan

> **For agentic workers:** Execute inline in the approved checkout; do not create a worktree, commit, start training, or terminate existing processes.

**Goal:** Create a three-SLM negative-feedback configuration and GPU0 train-then-full-Test launcher based on the named trained SLM5 experiment.

**Architecture:** Copy the checkpoint configuration and permit only the three structural differences. Adapt the verified file-backed Test launcher for linear negative feedback and physical GPU0.

**Tech Stack:** Python, PyYAML, Bash, PyTorch, CUDA.

## Global Constraints

- Preserve linear output mode, feedback_sign -1.0, learnable_offset, no memory, 25 epochs, batch 25, and learning rate 0.0001.
- Use num_slm_layers 3, two 25000.0 inter-SLM intervals, and feedback_layer_index 1.
- Run full Test with batch size 15 on physical GPU0.
- Never invoke Test through Python standard input.
- Do not start training during validation.

---

### Task 1: Create and compare configuration

**Files:**
- Create: evals/intuitive_physics/configs/onn_feedback_linear_slm3_negative_25ep.yaml

- [ ] Load the named best checkpoint configuration.
- [ ] Change only num_slm_layers, slm_intervals_um, and feedback_layer_index.
- [ ] Validate ONNConfig and recursively compare all values.

### Task 2: Create and validate launcher

**Files:**
- Create: scripts/run_linear_slm3_negative_25ep_then_test_gpu0_bs15.sh

- [ ] Adapt the verified file-backed launcher.
- [ ] Bind physical GPU0 as logical cuda:0.
- [ ] Preserve 25 epochs, training batch 25, and learning rate 0.0001.
- [ ] Run full Test with batch 15 and both completion gates.
- [ ] Run Bash syntax and embedded runner compilation checks without launching training.
