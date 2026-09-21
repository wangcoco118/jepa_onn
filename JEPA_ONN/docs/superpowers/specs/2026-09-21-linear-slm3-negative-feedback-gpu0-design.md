# Linear SLM3 Negative-Feedback GPU0 Pipeline Design

Derive the new configuration from the specified trained negative-feedback SLM5 checkpoint. Preserve linear output mode, feedback_sign -1.0, learnable_offset readout, feedback gain, no feedback memory, 25 epochs, training batch size 25, learning rate 0.0001, data split, propagation distances, and encoding. Change only the optical depth to three SLM layers, reduce slm_intervals_um to two unchanged 25000.0 intervals, and set feedback_layer_index to 1, meaning the second physical SLM.

Create a physical-GPU0 serial launcher. After training run_done and best-checkpoint checks, run complete O1/O2/O3 Test inference with batch size 15 through a real file-backed Python runner with an __main__ guard. Require Test run_done output_dir before PIPELINE_DONE. Do not start training or terminate existing processes during implementation.
