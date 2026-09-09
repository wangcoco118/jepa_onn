import unittest

import torch
import torch.distributed as dist

from evals.intuitive_physics import eval as dev_eval
from evals.intuitive_physics import utils as dev_utils
from evals.intphys_test import utils as test_utils


class EvalDistributedFallbackTests(unittest.TestCase):
    def test_batched_intphys_quadruplets_keep_all_groups_and_metrics_are_safe(self):
        prepare_batch = getattr(dev_eval, "_prepare_intphys_batch", None)
        self.assertIsNotNone(prepare_batch)

        clips = torch.zeros(2, 4, 1, 3, 1, 1)
        labels = torch.tensor(
            [[0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]]
        )

        flat_clips, flat_labels, pair_matches, group_labels = prepare_batch(
            clips, labels
        )

        self.assertEqual(tuple(flat_clips.shape), (8, 1, 3, 1, 1))
        self.assertTrue(torch.equal(flat_labels, labels.reshape(-1)))
        self.assertTrue(torch.equal(group_labels, labels))
        self.assertEqual(pair_matches, [[0, 1], [2, 3], [4, 5], [6, 7]])

        losses = torch.arange(24, dtype=torch.float32).reshape(8, 3)
        try:
            metrics = dev_eval.compute_metrics(losses, flat_labels)
        except Exception as exc:
            self.fail(f"small valid IntPhys metrics raised {exc!r}")
        self.assertIn("Classifier threhshold", metrics)

    def test_last_context_copy_flag_defaults_off_and_accepts_boolean(self):
        self.assertFalse(dev_eval._last_context_copy_enabled({}))
        self.assertFalse(
            dev_eval._last_context_copy_enabled(
                {"evaluation": {"last_context_copy_baseline": False}}
            )
        )
        self.assertTrue(
            dev_eval._last_context_copy_enabled(
                {"evaluation": {"last_context_copy_baseline": True}}
            )
        )
        with self.assertRaisesRegex(TypeError, "must be a boolean"):
            dev_eval._last_context_copy_enabled(
                {"evaluation": {"last_context_copy_baseline": "true"}}
            )

    def test_last_context_copy_repeats_complete_last_spatial_slice(self):
        context = torch.arange(18, dtype=torch.float32).reshape(1, 6, 3)

        prediction = dev_eval._last_context_copy_prediction(
            context,
            num_target_tokens=4,
            spatial_tokens=2,
            normalize=False,
        )

        expected = context[:, -2:, :].repeat(1, 2, 1)
        self.assertTrue(torch.equal(prediction, expected))
        self.assertEqual(tuple(prediction.shape), (1, 4, 3))

    def test_batch_all_gather_returns_local_tensor_without_process_group(self):
        self.assertFalse(dist.is_initialized())
        tensor = torch.tensor([[1.0, 2.0]])

        for module in (dev_utils, test_utils):
            with self.subTest(module=module.__name__):
                try:
                    gathered = module.batch_all_gather(tensor)
                except Exception as exc:
                    self.fail(f"single-process gather raised {exc!r}")
                self.assertTrue(torch.equal(gathered, tensor))
                self.assertEqual(gathered.data_ptr(), tensor.data_ptr())


if __name__ == "__main__":
    unittest.main()
