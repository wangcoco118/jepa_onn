import tempfile
import unittest
from pathlib import Path

import torch

from evals.intphys_test import eval as test_eval


class TestRollingResumeCheckpoint(unittest.TestCase):
    def test_round_trip_preserves_latest_batch_and_accumulated_results(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "test_resume_latest.pt"
            losses = [torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]])]
            tasks = [torch.tensor([5]), torch.tensor([6])]

            test_eval._save_resume_checkpoint(
                checkpoint_path,
                next_batch=20,
                frame_step=2,
                context_lengths=[2, 4, 6, 8, 10],
                batch_size=15,
                predictor_checkpoint="/models/example.pt",
                all_losses=losses,
                all_tasks=tasks,
            )
            state = test_eval._load_resume_checkpoint(
                checkpoint_path,
                frame_step=2,
                context_lengths=[2, 4, 6, 8, 10],
                batch_size=15,
                predictor_checkpoint="/models/example.pt",
            )

            self.assertEqual(state["next_batch"], 20)
            self.assertEqual(len(state["all_losses"]), 2)
            self.assertTrue(torch.equal(state["all_losses"][1], losses[1]))
            self.assertTrue(torch.equal(state["all_tasks"][0], tasks[0]))

    def test_mismatched_checkpoint_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / "test_resume_latest.pt"
            test_eval._save_resume_checkpoint(
                checkpoint_path,
                next_batch=20,
                frame_step=2,
                context_lengths=[2, 4],
                batch_size=15,
                predictor_checkpoint="/models/example.pt",
                all_losses=[torch.tensor([[1.0]])],
                all_tasks=[torch.tensor([5])],
            )

            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                test_eval._load_resume_checkpoint(
                    checkpoint_path,
                    frame_step=2,
                    context_lengths=[2, 4],
                    batch_size=10,
                    predictor_checkpoint="/models/example.pt",
                )


if __name__ == "__main__":
    unittest.main()
