# Linear SLM1 Negative-Feedback Design

Create an isolated fresh-training pipeline derived from the completed Linear SLM3 negative-feedback experiment. The source best checkpoint supplies only the effective configuration; no learned model, optimizer, scheduler, epoch, or random state is resumed.

Change the ONN core from three SLM layers to one, move the single feedback injection to index 0, and remove inter-SLM distances. Preserve negative feedback, feedback gain, readout, data split, learning rate, mask mode, Linear output, and all other settings.

Use physical GPU3 as logical cuda:0. Train 25 epochs with batch size 25. After verified training completion, run complete O1/O2/O3 Test with batch size 15 and require both 12,960-row answer files plus the losses artifact.
