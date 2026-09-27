# Linear SLM1 Temporal-Difference Alpha 1 Design

Create an isolated fresh-training pipeline derived from the completed Linear SLM1 alpha 0.5 experiment. The source best checkpoint supplies only the effective configuration; no learned weights or training state are resumed.

Preserve the one-SLM core, feedback index 0, empty inter-SLM distances, positive feedback default, Linear output, data split, and all other settings. Change only predictor.temporal_difference_alpha from 0.5 to 1.0 plus identifying tags.

Use physical GPU0. Train 25 epochs with batch size 25. After verified training completion, run complete O1/O2/O3 Test with batch size 15. Enable rolling Test resume every 20 batches and validate completion artifacts and answer row counts.
