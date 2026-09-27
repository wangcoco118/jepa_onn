# Linear SLM3 Temporal-Difference Alpha 1 Design

Create an isolated train-to-Test launcher based on the completed alpha 0.5 SLM3 experiment. The source checkpoint is used only to recover the effective configuration; no model, optimizer, scheduler, epoch, or random state is resumed.

Only predictor.temporal_difference_alpha changes from 0.5 to 1.0. Temporal difference remains enabled, the ONN stays at three SLM layers with feedback index 1, training remains 25 epochs with batch size 25, and all other effective configuration values come from the source checkpoint.

The launcher uses physical GPU2 as logical cuda:0. After a successful fresh training run and a verified training RUN_DONE marker, it selects the validation-best checkpoint and runs complete O1/O2/O3 Test with batch size 10. Completion requires the Test RUN_DONE marker, both average and maximum answer files, the losses file, and exactly 12,960 rows in each answer file.
