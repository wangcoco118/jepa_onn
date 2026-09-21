# Interpolate SLM3 Output-Mean GPU2 Pipeline Design

## Goal

Reproduce the source interpolate SLM3 feedback experiment for 25 epochs while changing only detector centering from the legacy learnable intensity offset to per-token output mean, then run full O1/O2/O3 Test inference with batch size 15 on physical GPU2.

## Configuration

The source checkpoint configuration remains authoritative. Preserve interpolate output mode, three SLM layers, positive single-layer feedback at physical SLM2, no feedback memory, the 2000/200 split, batch size 25, and learning rate 0.0001. Change training epochs from 50 to 25, normalize readout mode to output_mean, disable learnable_intensity_offset, and explicitly record feedback_sign as 1.0.

## Pipeline

Use CUDA_VISIBLE_DEVICES=2 so program-visible cuda:0 maps to physical GPU2. Train with skip-final-eval, verify the training run_done marker and best checkpoint, create a full Test configuration with batch size 15, then invoke Test through a real Python file guarded by if __name__ == "__main__". Verify run_done output_dir before emitting PIPELINE_DONE.

## Verification

Compare the generated YAML recursively against the checkpoint configuration and permit only the four documented differences. Run bash -n, compile the embedded Test runner source, and assert all GPU, batch, completion-marker, and file-backed-runner settings without starting training.
