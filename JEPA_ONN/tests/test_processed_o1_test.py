import tempfile
import unittest
import zipfile
from unittest import mock

import torch
from pathlib import Path

from evals.intphys_test import run_processed_o1_test as runner
from evals.intphys_test.processed_test_support import (
    audit_manifest_records,
    build_task_mapping,
    create_submission_zip,
    plausibility_from_surprise,
)


class ProcessedO1TestSupportTests(unittest.TestCase):
    def test_manifest_requires_exact_four_movies_per_scene(self):
        records = [
            {"sample_id": "O1/0001/1", "path": "videos/a.pt"},
            {"sample_id": "O1/0001/2", "path": "videos/b.pt"},
            {"sample_id": "O1/0001/3", "path": "videos/c.pt"},
        ]
        audit = audit_manifest_records(records, data_root=Path("/unused"),
                                       expected_scene_count=1)
        self.assertFalse(audit["complete_o1"])
        self.assertEqual(audit["missing_count"], 1)
        self.assertEqual(audit["scene_count"], 1)

    def test_task_mapping_is_id_based_and_preserves_task_order(self):
        records = [
            {"sample_id": "O1/0001/2", "path": "videos/b.pt"},
            {"sample_id": "O1/0001/1", "path": "videos/a.pt"},
        ]
        mapping, audit = build_task_mapping(
            ["O1/0001/1", "O1/0001/2"], records
        )
        self.assertEqual(mapping, [("O1/0001/1", "videos/a.pt"),
                                    ("O1/0001/2", "videos/b.pt")])
        self.assertTrue(audit["path_mapping_valid"])

    def test_task_mapping_rejects_duplicate_task_paths(self):
        records = [{"sample_id": "O1/0001/1", "path": "videos/a.pt"}]
        mapping, audit = build_task_mapping(
            ["O1/0001/1", "O1/0001/1"], records
        )
        self.assertEqual(mapping, [])
        self.assertFalse(audit["path_mapping_valid"])
        self.assertEqual(audit["duplicate_count"], 1)

    def test_plausibility_is_bounded_and_decreases_with_surprise(self):
        high = plausibility_from_surprise(0.2)
        low = plausibility_from_surprise(1.2)
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)
        self.assertGreater(high, low)

    def test_submission_zip_contains_only_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            answer = root / "answer.txt"
            answer.write_text("O1/0001/1 0.5\n", encoding="utf-8")
            archive = root / "submission_o1.zip"
            create_submission_zip(answer, archive)
            with zipfile.ZipFile(archive) as zf:
                self.assertEqual(zf.namelist(), ["answer.txt"])

    def test_parser_accepts_transformer_predictor_type(self):
        parser = runner._build_parser()
        option_strings = {
            option
            for action in parser._actions
            for option in action.option_strings
        }
        self.assertIn("--predictor-type", option_strings)
        args = parser.parse_args([
            "--checkpoint", "vitl16.pth.tar",
            "--data-root", "processed",
            "--output-dir", "out",
            "--predictor-type", "vit_transformer",
        ])
        self.assertEqual(args.predictor_type, "vit_transformer")

    def test_transformer_checkpoint_uses_canonical_intphys_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "vitl16.pth.tar"
            torch.save({
                "epoch": 300,
                "encoder": {},
                "target_encoder": {},
                "predictor": {},
            }, checkpoint_path)
            try:
                checkpoint, config, pretrained_path = runner._checkpoint_config(
                    checkpoint_path,
                    predictor_type="vit_transformer",
                )
            except Exception as exc:
                self.fail(f"Transformer checkpoint should be accepted: {exc}")
            self.assertEqual(checkpoint["epoch"], 300)
            self.assertEqual(pretrained_path, checkpoint_path)
            self.assertEqual(config["predictor_type"], "vit_transformer")
            self.assertEqual(config["pretrain"]["model_name"], "vit_large")
            self.assertEqual(config["predictor"]["predictor_dim"], 384)
            self.assertEqual(config["data"]["context_lengths"], [2, 4, 6, 8, 10])

    def test_transformer_model_loads_all_weights_from_pretrained_checkpoint(self):
        checkpoint_path = Path("/models/vitl16.pth.tar")
        config = {
            "predictor_type": "vit_transformer",
            "pretrain": {
                "model_name": "vit_large",
                "patch_size": 16,
                "tubelet_size": 2,
                "frames_per_clip": 16,
                "pred_depth": 12,
                "enc_checkpoint_key": "encoder",
                "pred_checkpoint_key": "predictor",
            },
            "data": {"resolution": 224, "frames_per_clip": 16},
            "predictor": {"predictor_dim": 384},
        }
        modules = (torch.nn.Identity(), torch.nn.Identity(), torch.nn.Identity())
        with mock.patch.object(
            runner.canonical_eval,
            "init_model",
            return_value=modules,
        ) as init_model:
            runner._load_models(checkpoint_path, config, torch.device("cpu"))
        kwargs = init_model.call_args.kwargs
        self.assertEqual(kwargs["predictor_type"], "vit_transformer")
        self.assertEqual(kwargs["pretrained"], str(checkpoint_path))
        self.assertIsNone(kwargs["predictor_checkpoint"])


if __name__ == "__main__":
    unittest.main()
