# Interpolate SLM1 Output-Mean GPU3 Pipeline Design

Create an independent configuration from the interpolate SLM1 25-epoch checkpoint. Preserve every source value, including interpolate output mode, one SLM layer, positive feedback at physical SLM1, no feedback memory, batch size 25, and learning rate 0.0001. Change only detector readout to output_mean, disable learnable_intensity_offset, and explicitly record the existing positive feedback as feedback_sign 1.0.

Create a GPU3 serial launcher that trains for 25 epochs, verifies the training run_done marker and best checkpoint, then runs full O1/O2/O3 Test inference with batch size 15. Test must use a real file-backed Python entrypoint with an __main__ guard so multiprocessing spawn never resolves <stdin>. Require run_done output_dir before emitting PIPELINE_DONE. Do not start training during implementation.
