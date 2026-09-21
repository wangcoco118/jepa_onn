# Interpolate SLM3 Output-Mean GPU2 Pipeline Implementation Plan

> **For agentic workers:** Execute inline in the approved checkout; do not create a worktree, commit, or start training.

**Goal:** Create a source-matched 25-epoch output-mean training configuration and a GPU2 train-then-full-Test launcher.

**Architecture:** Derive YAML from the source checkpoint configuration and allow only four explicit differences. Use a single shell launcher that creates a file-backed Python Test entrypoint inside the run directory, preventing multiprocessing spawn from resolving <stdin>.

**Tech Stack:** Python, PyYAML, Bash, PyTorch, CUDA.

## Global Constraints

- output_mode remains interpolate.
- num_slm_layers remains 3.
- feedback remains positive, single-layer at physical SLM2, with no memory.
- training uses 25 epochs, batch size 25, and learning rate 0.0001.
- detector readout is output_mean with no learnable intensity offset.
- full Test uses batch size 15 on physical GPU2.
- implementation verification must not start training.

---

### Task 1: Generate the experiment YAML

**Files:**
- Create: evals/intuitive_physics/configs/onn_feedback_interpolate_slm3_output_mean_25ep.yaml

- [ ] Load the source checkpoint config.
- [ ] Change only epochs, readout_mode, learnable_intensity_offset, and explicit feedback_sign.
- [ ] Recursively compare source and target configurations.

### Task 2: Generate and validate the pipeline

**Files:**
- Create: scripts/run_interpolate_slm3_output_mean_25ep_then_test_gpu2_bs15.sh

- [ ] Bind physical GPU2 through CUDA_VISIBLE_DEVICES=2 and use logical --gpu 0.
- [ ] Train 25 epochs with batch size 25 and learning rate 0.0001.
- [ ] Gate Test on the best checkpoint and training run_done marker.
- [ ] Generate a real Python Test runner file and compile it before use.
- [ ] Run full Test with batch size 15 and require run_done output_dir.
- [ ] Run bash -n and static assertions without launching training.
