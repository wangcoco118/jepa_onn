# Processed O1 Transformer Predictor Implementation Plan

> **For agentic workers:** Execute inline with test-driven development; preserve all existing ONN behavior and user changes.

**Goal:** Allow the processed O1 Test runner to evaluate the trained V-JEPA Transformer Predictor checkpoint.

**Architecture:** Add an explicit predictor-type CLI selector. For Transformer mode, validate the original V-JEPA checkpoint keys, construct the existing fixed IntPhys evaluation contract, and let the canonical evaluator load encoder, target encoder, and Transformer Predictor from that checkpoint. ONN remains the default path.

**Tech Stack:** Python, PyTorch, unittest.

## Global Constraints

- Modify only the processed O1 runner and its focused tests.
- Do not start the full 4320-movie Test run.
- Preserve ONN checkpoint validation and default behavior.
- Use GPU 0 and batch size 1 in the final command.

### Task 1: Add Transformer checkpoint support

**Files:**
- Modify: `JEPA_ONN/tests/test_processed_o1_test.py`
- Modify: `JEPA_ONN/evals/intphys_test/run_processed_o1_test.py`

- [ ] Add failing tests for the new CLI selector and Transformer checkpoint configuration.
- [ ] Run focused tests and confirm failure is caused by missing support.
- [ ] Add `--predictor-type`, Transformer checkpoint validation/configuration, and branched canonical model loading.
- [ ] Run focused tests and Python compilation.
- [ ] Load the real 4.8 GB checkpoint on CPU and verify all three trained state dictionaries are selected without starting dataset inference.
- [ ] Inspect the exact diff and provide the GPU 0, batch-size 1 command.
