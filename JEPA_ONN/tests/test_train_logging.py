import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from evals.intuitive_physics import train_optical
from src.models.fsonn import FeedbackFSONN, ONNConfig
from src.models.optical_output import OpticalOutputMapper
from evals.intuitive_physics.train_optical import (
    _configure_logging,
    _format_jepa_batch_log,
    _format_progress,
)


class TrainLoggingTests(unittest.TestCase):
    def test_logger_writes_same_record_to_stdout_and_file(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "train.log"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                logger = _configure_logging(str(log_path))
                logger.info("step=1 loss=0.25 elapsed_s=1.5")
                for handler in logger.handlers:
                    handler.flush()

            self.assertIn("step=1 loss=0.25 elapsed_s=1.5", stdout.getvalue())
            self.assertIn("step=1 loss=0.25 elapsed_s=1.5", log_path.read_text(encoding="utf-8"))


    def test_progress_format_uses_completed_step_and_stage_total(self):
        self.assertEqual(
            _format_progress(10, 40),
            "[#####---------------] 25.0% (10/40)",
        )

    def test_batch_log_contains_only_dynamic_training_fields(self):
        message = _format_jepa_batch_log(
            epoch=0,
            stage="val",
            step=1,
            total_steps=4,
            mask_mode="unified_random",
            batch_size=20,
            n_ctxt=196,
            n_tgt=1372,
            covered_count=1568,
            missing_count=0,
            loss=0.713809,
            grad_norm=0.0,
            time_s=1.561,
        )

        self.assertEqual(
            message,
            "epoch=0 stage=val mask_mode=unified_random "
            "[#####---------------] 25.0% (1/4) batch=20 "
            "n_ctxt=196 n_tgt=1372 covered_count=1568 "
            "missing_count=0 loss=0.713809 grad_norm=0.000 time=1.561s",
        )
        for repeated_field in (
            "ctxt_shape",
            "onn_shape",
            "chunk=",
            "feedback_layer",
            "lr=",
            "batch_time_s",
            "epoch_time_s",
        ):
            self.assertNotIn(repeated_field, message)


    def test_optical_output_metadata_is_logged_separately_from_feedback(self):
        config = ONNConfig.from_mapping(
            {
                "input_dim": 2,
                "output_dim": 2,
                "num_slm_layers": 4,
                "chunk_tokens": 2,
                "grid_height": 2,
                "grid_width": 2,
                "feedback_layer_index": 2,
                "slm_intervals_um": [8.0, 8.0, 8.0],
                "input_to_first_slm_um": 8.0,
                "last_slm_to_detector_um": 8.0,
                "asm_padding_factor": 1.0,
            }
        )

        class PredictorStub(nn.Module):
            def __init__(self):
                super().__init__()
                self.onn_core = FeedbackFSONN(config)
                self.output_mode = "optical"
                self.optical_output_mapper = OpticalOutputMapper()

        metadata = train_optical._feedback_runtime_metadata(PredictorStub())

        self.assertEqual(metadata["output_mode"], "optical")
        self.assertTrue(metadata["target_only"])
        self.assertEqual(metadata["optical_num_slm_layers"], 2)
        self.assertEqual(metadata["optical_grid_size"], 32)
        self.assertEqual(metadata["optical_wavelength_nm"], 532.0)
        self.assertEqual(metadata["optical_pixel_size_um"], 8.0)
        self.assertEqual(
            metadata["optical_distances_um"],
            [8000.0, 8000.0, 8000.0],
        )
        self.assertEqual(
            metadata["optical_output_centering"],
            "per_token_mean",
        )
        message = train_optical._format_feedback_metadata(metadata)
        self.assertIn("output_mode=optical", message)
        self.assertIn("target_only=True", message)
        self.assertIn("optical_distances_um=[8000.0, 8000.0, 8000.0]", message)

    def test_feedback_metadata_and_checkpoint_use_resolved_independent_model(self):
        self.assertTrue(hasattr(train_optical, "_feedback_runtime_metadata"))
        config = ONNConfig.from_mapping(
            {
                "input_dim": 2,
                "output_dim": 2,
                "num_slm_layers": 5,
                "chunk_tokens": 2,
                "grid_height": 2,
                "grid_width": 2,
                "feedback_layer_mode": "multi",
                "feedback_layer_indices": [2, 3, 4],
                "feedback_gain_mode": "independent",
                "feedback_gain_init": [0.5, 1.5, 3.0],
                "feedback_phase_max_rad": 0.75,
                "feedback_sign": -1.0,
                "readout_mode": "output_mean",
                "learnable_intensity_offset": False,
                "feedback_memory_enabled": True,
                "feedback_memory_alpha": 0.8,
                "slm_intervals_um": [8.0, 8.0, 8.0, 8.0],
                "input_to_first_slm_um": 8.0,
                "last_slm_to_detector_um": 8.0,
                "asm_padding_factor": 1.0,
            }
        )

        class PredictorStub(nn.Module):
            def __init__(self):
                super().__init__()
                self.onn_core = FeedbackFSONN(config)
                self.feedback_mode = "fixed_middle_phase"

        predictor = PredictorStub()
        metadata = train_optical._feedback_runtime_metadata(predictor)

        self.assertEqual(metadata["feedback_layer_mode"], "multi")
        self.assertNotIn("feedback_layer_index", metadata)
        self.assertEqual(metadata["feedback_layer_indices"], [2, 3, 4])
        self.assertEqual(metadata["physical_feedback_layers"], [3, 4, 5])
        self.assertEqual(metadata["feedback_gain_mode"], "independent")
        self.assertEqual(metadata["feedback_sign"], -1.0)
        self.assertEqual(metadata["readout_mode"], "output_mean")
        self.assertEqual(metadata["feedback_gain_parameter_count"], 3)
        self.assertTrue(metadata["feedback_memory_enabled"])
        self.assertEqual(metadata["feedback_memory_alpha"], 0.8)
        self.assertEqual(
            metadata["feedback_memory_update"],
            "H0=Z0; Ht=0.8*Hprev+0.2*Zt",
        )
        self.assertTrue(
            torch.allclose(
                torch.tensor(metadata["effective_feedback_gains"]),
                torch.tensor([0.5, 1.5, 3.0]),
                atol=1e-6,
            )
        )
        message = train_optical._format_feedback_metadata(metadata)
        self.assertIn("feedback_layer_mode=multi", message)
        self.assertIn("feedback_sign=-1", message)
        self.assertIn("readout_mode=output_mean", message)
        self.assertIn("SLM3_K=0.500000", message)
        self.assertIn("SLM4_K=1.500000", message)
        self.assertIn("SLM5_K=3.000000", message)
        self.assertIn("feedback_memory_enabled=true", message)
        self.assertIn("feedback_memory_alpha=0.8", message)
        self.assertIn(
            "feedback_memory_update=H0=Z0; Ht=0.8*Hprev+0.2*Zt",
            message,
        )

        optimizer = torch.optim.AdamW(predictor.parameters(), lr=1e-3)
        checkpoint = train_optical._end_to_end_checkpoint(
            predictor,
            optimizer,
            None,
            1,
            1,
            0.5,
            {"train_video_ids": ["a"], "val_video_ids": ["b"]},
            "/tmp/split.json",
            {
                "pretrain": {"folder": "/checkpoint", "checkpoint": "official.pt"},
                "training": {"mode": "onn_feedback"},
            },
            "best",
            experiment_mode="onn_feedback",
        )
        for key, value in metadata.items():
            self.assertEqual(checkpoint[key], value)



    def test_training_summary_recursively_records_effective_config_with_chinese_comments(self):
        with tempfile.TemporaryDirectory() as directory:
            summary_path = Path(directory) / "training_summary.txt"
            config = {
                "training": {"epochs": 25},
                "data": {"batch_size": 15},
                "onn": {"feedback_enabled": True},
                "custom": {"new_parameter": [1, 2]},
            }

            train_optical._write_training_summary(
                summary_path,
                config,
                stage="start",
                cli_args={"gpu": 2, "learning_rate": 1e-4},
                runtime_metadata={"output_path": "/tmp/model.pt"},
            )

            summary = summary_path.read_text(encoding="utf-8")
            self.assertIn("[\u8fd0\u884c\u4fe1\u606f]", summary)
            self.assertIn("status = running", summary)
            self.assertIn("training.epochs = 25  # \u603b\u8bad\u7ec3\u8f6e\u6570", summary)
            self.assertIn("data.batch_size = 15  # \u6bcf\u4e2a\u8bad\u7ec3\u6279\u6b21\u7684\u6837\u672c\u6570", summary)
            self.assertIn(
                "custom.new_parameter = [1, 2]  # \u53ef\u914d\u7f6e\u53c2\u6570\uff1acustom.new_parameter",
                summary,
            )
            assignment_lines = [
                line
                for line in summary.splitlines()
                if " = " in line and not line.startswith("status = ")
            ]
            self.assertTrue(assignment_lines)
            self.assertTrue(all("  # " in line for line in assignment_lines))

    def test_training_summary_appends_runtime_and_final_results(self):
        with tempfile.TemporaryDirectory() as directory:
            summary_path = Path(directory) / "training_summary.txt"
            config = {"training": {"epochs": 3}}
            train_optical._write_training_summary(summary_path, config, stage="start")
            train_optical._write_training_summary(
                summary_path,
                config,
                stage="runtime",
                runtime_metadata={"trainable_parameter_count": 123},
            )
            train_optical._write_training_summary(
                summary_path,
                config,
                stage="complete",
                results={
                    "best_epoch": 2,
                    "best_val_loss": 0.125,
                    "best_checkpoint": "/tmp/best.pt",
                },
            )

            summary = summary_path.read_text(encoding="utf-8")
            self.assertIn("status = completed", summary)
            self.assertNotIn("status = running", summary)
            self.assertEqual(summary.count("[\u5b8c\u6574\u751f\u6548\u914d\u7f6e]"), 1)
            self.assertIn("[\u8fd0\u884c\u65f6\u6d3e\u751f\u53c2\u6570]", summary)
            self.assertIn(
                "runtime.trainable_parameter_count = 123  # \u53ef\u8bad\u7ec3\u53c2\u6570\u603b\u6570",
                summary,
            )
            self.assertIn("[\u6700\u7ec8\u7ed3\u679c]", summary)
            self.assertIn("result.best_epoch = 2  # \u9a8c\u8bc1\u96c6\u8868\u73b0\u6700\u4f73\u7684\u8f6e\u6b21", summary)
            self.assertIn("result.best_val_loss = 0.125  # \u6700\u4f73\u9a8c\u8bc1\u635f\u5931", summary)


if __name__ == "__main__":
    unittest.main()
