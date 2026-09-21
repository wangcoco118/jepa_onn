# Interpolate SLM5 Output-Mean GPU2 Pipeline Design

Derive an independent configuration from the interpolate SLM5 source checkpoint. Preserve interpolate output mode, five SLM layers, positive single feedback at physical SLM4, no feedback memory, the 2000/200 split, training batch size 25, and learning rate 0.0001. Set epochs to 25, replace the legacy learnable-offset readout with output_mean, disable learnable_intensity_offset, and explicitly record feedback_sign 1.0.

Create a physical-GPU2 serial launcher that verifies training run_done and the best checkpoint before running complete O1/O2/O3 Test inference with batch size 15. Test must run through a real Python file guarded by __main__, never through <stdin>. Require Test run_done output_dir before PIPELINE_DONE. Do not start training during implementation.
