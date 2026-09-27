import unittest
from pathlib import Path


class TrainingPipelineScriptTests(unittest.TestCase):
    def test_linear_slm3_temporal_diff_a1_pipeline_contract(self):
        script_path = Path(
            "scripts/"
            "run_linear_slm3_temporal_diff_a1_25ep_bs25_then_test_gpu2_bs10.sh"
        )
        script = script_path.read_text(encoding="utf-8")

        self.assertIn("linear_slm3_feedback_slm1_temporal_diff_a0.5_20260922_192812", script)
        self.assertIn('export CUDA_VISIBLE_DEVICES=2', script)
        self.assertIn('predictor_config["temporal_difference_alpha"] = 1.0', script)
        self.assertIn('source_alpha != 0.5', script)
        self.assertIn('actual_layers != 3', script)
        self.assertIn('actual_feedback != 1', script)
        self.assertIn('--epochs 25', script)
        self.assertIn('--batch-size 25', script)
        self.assertNotIn('--resume', script)
        self.assertIn('config["data"]["batch_size"] = 10', script)
        self.assertIn('config["mode"] = "all"', script)
        self.assertIn('run_done experiment_mode=onn_feedback', script)
        self.assertIn('run_done output_dir=', script)
        self.assertIn('average_surprise_answer.txt', script)
        self.assertIn('maximum_surprise_answer.txt', script)
        self.assertIn('losses_2fs_2_4_6_8_10ctxt.pth', script)
        self.assertIn('"$rows" -ne 12960', script)


    def test_linear_slm1_negative_feedback_pipeline_contract(self):
        script_path = Path(
            "scripts/"
            "run_linear_slm1_negative_25ep_bs25_then_test_gpu3_bs15.sh"
        )
        script = script_path.read_text(encoding="utf-8")

        self.assertIn("onn_linear_slm3_negative_feedback_25ep_gpu0", script)
        self.assertIn("export CUDA_VISIBLE_DEVICES=3", script)
        self.assertIn("actual_layers != 3", script)
        self.assertIn("actual_feedback != 1", script)
        self.assertIn("source_sign != -1.0", script)
        self.assertIn('onn_config["num_slm_layers"] = 1', script)
        self.assertIn('onn_config["feedback_layer_index"] = 0', script)
        self.assertIn('onn_config["slm_intervals_um"] = []', script)
        self.assertIn('onn_config["feedback_sign"] = -1.0', script)
        self.assertIn("--epochs 25", script)
        self.assertIn("--batch-size 25", script)
        self.assertNotIn("--resume", script)
        self.assertIn('config["data"]["batch_size"] = 15', script)
        self.assertIn('config["mode"] = "all"', script)
        self.assertIn("run_done experiment_mode=onn_feedback", script)
        self.assertIn("run_done output_dir=", script)
        self.assertIn("average_surprise_answer.txt", script)
        self.assertIn("maximum_surprise_answer.txt", script)
        self.assertIn("losses_2fs_2_4_6_8_10ctxt.pth", script)
        self.assertIn('"$rows" -ne 12960', script)


    def test_linear_slm1_temporal_diff_a1_pipeline_contract(self):
        script_path = Path(
            "scripts/"
            "run_linear_slm1_temporal_diff_a1_25ep_bs25_then_test_gpu0_bs15.sh"
        )
        script = script_path.read_text(encoding="utf-8")

        self.assertIn("linear_slm1_temporal_diff_a0.5_20260922_232750", script)
        self.assertIn("export CUDA_VISIBLE_DEVICES=0", script)
        self.assertNotIn("等待物理 GPU0 空闲", script)
        self.assertNotIn("nvidia-smi pmon", script)
        self.assertIn("actual_layers != 1", script)
        self.assertIn("actual_feedback != 0", script)
        self.assertIn("source_intervals", script)
        self.assertIn('predictor_config["temporal_difference_alpha"] = 1.0', script)
        self.assertIn("source_alpha != 0.5", script)
        self.assertIn("--epochs 25", script)
        self.assertIn("--batch-size 25", script)
        self.assertNotIn("--resume", script)
        self.assertIn('config["data"]["batch_size"] = 15', script)
        self.assertIn('config["mode"] = "all"', script)
        self.assertIn('"test_resume"', script)
        self.assertIn('"enabled": True', script)
        self.assertIn('"save_every_batches": 20', script)
        self.assertIn("run_done experiment_mode=onn_feedback", script)
        self.assertIn("run_done output_dir=", script)
        self.assertIn("average_surprise_answer.txt", script)
        self.assertIn("maximum_surprise_answer.txt", script)
        self.assertIn("losses_2fs_2_4_6_8_10ctxt.pth", script)
        self.assertIn('"$rows" -ne 12960', script)


if __name__ == "__main__":
    unittest.main()
