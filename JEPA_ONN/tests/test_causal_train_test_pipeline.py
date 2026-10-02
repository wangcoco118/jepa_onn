import copy
import importlib.util
from pathlib import Path
import unittest

import torch
import yaml


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/run_causal_recurrent_slm3_25ep_bs25_then_test_gpu1_bs10.py"


class CausalTrainTestPipelineTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SCRIPT.is_file(), "training/Test pipeline is missing")
        spec = importlib.util.spec_from_file_location("causal_pipeline", SCRIPT)
        self.pipeline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.pipeline)
        self.config = yaml.safe_load((REPO / (
            "evals/intuitive_physics/configs/onn_causal_recurrent_slm3.yaml"
        )).read_text())

    def checkpoint(self):
        return {
            "mode": "onn_causal_recurrent",
            "checkpoint_kind": "last",
            "checkpoint_selection": "last_epoch",
            "epoch": 25,
            "selected_epoch": 25,
            "architecture_version": 2,
            "output_dim": 1024,
            "loss_feature_dim": 1024,
            "output_layer_norm": False,
            "config": copy.deepcopy(self.config),
            "predictor": {"output_linear.weight": torch.zeros(1024, 384)},
        }

    def test_train_command_disables_validation_and_final_dev_test(self):
        command = self.pipeline.training_command(
            REPO, REPO / "train.yaml", "unique_run", "/python"
        )
        self.assertEqual(command[command.index("--gpu") + 1], "0")
        self.assertEqual(command[command.index("--batch-size") + 1], "25")
        self.assertEqual(command[command.index("--epochs") + 1], "25")
        self.assertEqual(command[command.index("--checkpoint-selection") + 1],
                         "last_epoch")
        self.assertIn("--skip-final-eval", command)
        self.assertNotIn("--resume", command)

    def test_last_checkpoint_is_taken_from_this_runs_completion_marker(self):
        root = REPO.parent / "output"
        expected = root / "unique_run_stamp" / "unique_run.last.pt"
        text = ("old output contains best=/old/best.pt\n"
                "run_done experiment_mode=onn_causal_recurrent "
                "selected_epoch=25 best=/old/best.pt last=" + str(expected))
        actual = self.pipeline.last_checkpoint_from_log(text, root, "unique_run")
        self.assertEqual(actual, expected)
        with self.assertRaisesRegex(ValueError, "run_done"):
            self.pipeline.last_checkpoint_from_log("training crashed", root,
                                                   "unique_run")
        with self.assertRaisesRegex(ValueError, "this run"):
            self.pipeline.last_checkpoint_from_log(
                text.replace("unique_run.last.pt", "wrong_run.last.pt"),
                root, "unique_run"
            )

    def test_only_last_epoch_25_linear_checkpoint_is_accepted(self):
        self.pipeline.validate_checkpoint(self.checkpoint())
        for field, value in (("epoch", 24), ("checkpoint_kind", "best"),
                             ("architecture_version", 1), ("output_dim", 384)):
            checkpoint = self.checkpoint()
            checkpoint[field] = value
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    self.pipeline.validate_checkpoint(checkpoint)

    def test_test_config_preserves_model_but_uses_batch_ten_and_mean(self):
        checkpoint = self.checkpoint()
        snapshot = copy.deepcopy(checkpoint["config"])
        last = REPO.parent / "output" / "run" / "run.last.pt"
        output = last.parent / "test_eval"
        config = self.pipeline.build_test_config(checkpoint, last, output)
        self.assertEqual(config["dataset"], "intphys-test")
        self.assertEqual(config["eval_name"], "intphys_test")
        self.assertEqual(config["mode"], "all")
        self.assertEqual(config["data"]["batch_size"], 10)
        self.assertEqual(config["evaluation"]["spatial_reduction"], "mean")
        self.assertEqual(config["evaluation"]["temporal_reduction"], "mean")
        self.assertFalse(config["test_resume"]["enabled"])
        self.assertEqual(Path(config["predictor_checkpoint"]), last)
        self.assertEqual(Path(config["output_dir"]), output)
        self.assertEqual(config["onn"], snapshot["onn"])
        self.assertEqual(config["predictor"], snapshot["predictor"])
        self.assertEqual(checkpoint["config"], snapshot)

    def test_answer_validation_checks_all_tasks_not_just_row_count(self):
        tasks = ["O1/a/1", "O2/b/1", "O3/c/1"]
        valid = ["O1/a/1 0.9", "O2/b/1 0.4", "O3/c/1 0.7"]
        self.pipeline.validate_answer_rows(valid, tasks)
        for rows in (valid[:-1],
                     [valid[0], valid[0], valid[2]],
                     [valid[0], "O2/b/1 nan", valid[2]],
                     [valid[0], "O2/b/1 1.2", valid[2]]):
            with self.subTest(rows=rows):
                with self.assertRaises(ValueError):
                    self.pipeline.validate_answer_rows(rows, tasks)


if __name__ == "__main__":
    unittest.main()
