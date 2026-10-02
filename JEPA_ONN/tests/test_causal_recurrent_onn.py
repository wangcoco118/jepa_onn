import unittest

import torch
import torch.nn as nn

from evals.causal_recurrent import (
    compute_causal_recurrent_loss,
    compute_causal_recurrent_scores,
    encode_independent_chunks,
    reduce_scores,
)
from src.models.causal_recurrent_onn import CausalRecurrentONNPredictor
from src.models.fsonn import ONNConfig


class ScaleONN(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.weight = nn.Parameter(
            torch.tensor(
                [[float(scale), 0.2], [0.1, float(scale) + 0.3]]
            )
        )

    def forward(self, x):
        return torch.matmul(x, self.weight.t())


class RecordingEncoder(nn.Module):
    def __init__(self, tokens, dim):
        super().__init__()
        self.tokens = int(tokens)
        self.dim = int(dim)
        self.batch_sizes = []

    def forward(self, clips, masks):
        self.batch_sizes.append(int(clips.shape[0]))
        base = clips.mean(dim=(1, 2, 3, 4)).reshape(-1, 1, 1)
        offsets = torch.arange(
            self.dim, device=clips.device, dtype=clips.dtype
        ).reshape(1, 1, self.dim)
        features = (base + offsets).expand(-1, self.tokens, -1)
        return [features]


def small_onn_config(dim=2, chunk_tokens=2):
    return ONNConfig.from_mapping(
        {
            "input_dim": dim,
            "output_dim": dim,
            "num_slm_layers": 1,
            "chunk_tokens": chunk_tokens,
            "grid_height": chunk_tokens,
            "grid_width": dim,
            "feedback_mode": "fixed_middle_phase",
            "feedback_enabled": False,
            "feedback_memory_enabled": False,
            "feedback_layer_index": 0,
            "readout_mode": "learnable_offset",
            "pixel_pitch_um": 8.0,
            "wavelength_nm": 532.0,
            "slm_intervals_um": [],
            "input_to_first_slm_um": 8.0,
            "last_slm_to_detector_um": 8.0,
            "asm_padding_factor": 1.0,
            "learnable_intensity_offset": True,
            "use_differential_detector": False,
        }
    )


def make_predictor(memory_lambda=0.5, num_context_chunks=3):
    model = CausalRecurrentONNPredictor(
        embed_dim=2,
        predictor_embed_dim=2,
        num_context_chunks=num_context_chunks,
        chunk_tokens=2,
        memory_lambda=memory_lambda,
        onn_config=small_onn_config(),
        input_onn=ScaleONN(1.0),
        memory_onn=ScaleONN(1.0),
        prediction_onn=ScaleONN(1.0),
    )
    state = {
        "module.backbone.predictor_embed.weight": torch.eye(2),
        "module.backbone.predictor_embed.bias": torch.zeros(2),
    }
    model.load_predictor_embed_state_dict(state)
    return model


class CausalRecurrentPredictorTests(unittest.TestCase):
    def test_output_shape_and_state_shape(self):
        model = make_predictor()
        context = torch.randn(2, 3, 2, 2)

        predictions, states = model(context, return_states=True)

        self.assertEqual(tuple(predictions.shape), (2, 3, 2, 2))
        self.assertEqual(tuple(states.shape), (2, 3, 2, 2))

    def test_projector_is_loaded_by_suffix_and_frozen(self):
        model = make_predictor()
        torch.testing.assert_close(
            model.predictor_embed.weight, torch.eye(2)
        )
        torch.testing.assert_close(
            model.predictor_embed.bias, torch.zeros(2)
        )
        self.assertFalse(
            any(p.requires_grad for p in model.predictor_embed.parameters())
        )

    def test_projector_requires_exactly_one_weight_and_bias(self):
        model = make_predictor()
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            model.load_predictor_embed_state_dict(
                {"predictor_embed.weight": torch.eye(2)}
            )

    def test_memory_lambda_is_persistent_buffer_not_parameter(self):
        model = make_predictor(memory_lambda=0.2)
        self.assertIn("memory_lambda", dict(model.named_buffers()))
        self.assertNotIn("memory_lambda", dict(model.named_parameters()))
        self.assertAlmostEqual(float(model.memory_lambda), 0.2)
        self.assertIn("memory_lambda", model.state_dict())

    def test_three_onns_are_independent_objects(self):
        model = make_predictor()
        self.assertIsNot(model.input_onn, model.memory_onn)
        self.assertIsNot(model.input_onn, model.prediction_onn)
        self.assertIsNot(model.memory_onn, model.prediction_onn)

    def test_last_step_loss_backpropagates_through_all_three_onns(self):
        model = make_predictor(memory_lambda=0.5)
        torch.manual_seed(13)
        context = torch.randn(1, 3, 2, 2)

        predictions = model(context)
        predictions[:, -1, :, 0].sum().backward()

        for module in (
            model.input_onn,
            model.memory_onn,
            model.prediction_onn,
        ):
            self.assertIsNotNone(module.weight.grad)
            self.assertGreater(
                float(module.weight.grad.abs().sum()), 0.0
            )

    def test_lambda_zero_removes_earlier_history_from_later_steps(self):
        model = make_predictor(memory_lambda=0.0)
        first = torch.zeros(1, 3, 2, 2)
        second = first.clone()
        second[:, 0, :, 0] = 100.0

        first_predictions = model(first)
        second_predictions = model(second)

        self.assertFalse(torch.equal(
            first_predictions[:, 0], second_predictions[:, 0]
        ))
        torch.testing.assert_close(
            first_predictions[:, 1:],
            second_predictions[:, 1:],
        )

    def test_target_features_stay_in_encoder_space(self):
        model = make_predictor()
        targets = torch.randn(1, 4, 2, 2)
        expected = targets
        actual = model.prepare_target_features(targets)
        torch.testing.assert_close(actual, expected)

    def test_real_onns_disable_internal_feedback_and_freeze_gain(self):
        model = CausalRecurrentONNPredictor(
            embed_dim=2,
            predictor_embed_dim=2,
            num_context_chunks=3,
            chunk_tokens=2,
            memory_lambda=0.5,
            onn_config=small_onn_config(),
        )
        for module in (
            model.input_onn,
            model.memory_onn,
            model.prediction_onn,
        ):
            self.assertFalse(module.config.feedback_enabled)
            self.assertFalse(module.config.feedback_memory_enabled)
            self.assertFalse(module.feedback_gain_raw.requires_grad)


class CausalRecurrentLossTests(unittest.TestCase):
    def test_loss_shapes_and_motion_weights_are_detached(self):
        predictions = torch.randn(
            2, 3, 4, 5, requires_grad=True
        )
        targets = torch.randn(
            2, 4, 4, 5, requires_grad=True
        )

        result = compute_causal_recurrent_loss(
            predictions,
            targets,
            loss_exp=1.0,
            motion_beta=1.0,
            motion_epsilon=1.0e-6,
        )

        self.assertEqual(tuple(result["token_error"].shape), (2, 3, 4))
        self.assertEqual(tuple(result["motion"].shape), (2, 3, 4))
        self.assertEqual(tuple(result["motion_weights"].shape), (2, 3, 4))
        self.assertFalse(result["motion"].requires_grad)
        self.assertFalse(result["motion_weights"].requires_grad)
        result["loss"].backward()
        self.assertIsNotNone(predictions.grad)

    def test_epsilon_keeps_static_sequence_finite(self):
        predictions = torch.zeros(1, 2, 3, 4)
        targets = torch.zeros(1, 3, 3, 4)

        result = compute_causal_recurrent_loss(
            predictions,
            targets,
            motion_epsilon=1.0e-6,
        )

        self.assertTrue(torch.isfinite(result["loss"]))
        self.assertTrue(torch.isfinite(result["motion_weights"]).all())

    def test_target_change_does_not_change_predictions(self):
        model = make_predictor()
        context = torch.randn(1, 3, 2, 2)
        predictions = model(context).detach().clone()
        targets_a = model.prepare_target_features(torch.randn(1, 4, 2, 2))
        targets_b = targets_a.clone()
        targets_b[:, -1] += 10.0

        loss_a = compute_causal_recurrent_loss(predictions, targets_a)["loss"]
        loss_b = compute_causal_recurrent_loss(predictions, targets_b)["loss"]

        torch.testing.assert_close(predictions, model(context))
        self.assertNotEqual(float(loss_a), float(loss_b))


class CausalRecurrentScoringTests(unittest.TestCase):
    def test_copy_baseline_is_previous_target(self):
        predictions = torch.zeros(1, 3, 2, 2)
        targets = torch.arange(
            1 * 4 * 2 * 2, dtype=torch.float32
        ).reshape(1, 4, 2, 2)

        scores = compute_causal_recurrent_scores(predictions, targets)

        expected = torch.abs(
            targets[:, :-1] - targets[:, 1:]
        ).mean(dim=-1)
        torch.testing.assert_close(
            scores["copy_token_error"], expected
        )

    def test_score_reducers_support_mean_max_and_top_n(self):
        values = torch.tensor([[1.0, 4.0, 2.0, 3.0]])
        torch.testing.assert_close(
            reduce_scores(values, "mean", dim=1, top_n=2),
            torch.tensor([2.5]),
        )
        torch.testing.assert_close(
            reduce_scores(values, "max", dim=1, top_n=2),
            torch.tensor([4.0]),
        )
        torch.testing.assert_close(
            reduce_scores(values, "top_n", dim=1, top_n=2),
            torch.tensor([3.5]),
        )


class IndependentChunkEncodingTests(unittest.TestCase):
    def test_chunks_are_independent_batch_items(self):
        context_encoder = RecordingEncoder(tokens=4, dim=3)
        target_encoder = RecordingEncoder(tokens=4, dim=3)
        clips = torch.randn(2, 3, 16, 4, 4)

        context, targets = encode_independent_chunks(
            clips,
            context_encoder,
            target_encoder,
            chunk_frames=2,
            chunk_tokens=4,
        )

        self.assertEqual(tuple(context.shape), (2, 7, 4, 3))
        self.assertEqual(tuple(targets.shape), (2, 8, 4, 3))
        self.assertEqual(context_encoder.batch_sizes, [14])
        self.assertEqual(target_encoder.batch_sizes, [16])
        self.assertFalse(context.requires_grad)
        self.assertFalse(targets.requires_grad)


class CausalRecurrentEvaluationIntegrationTests(unittest.TestCase):
    def test_pretrained_loader_extracts_only_shared_projector(self):
        from unittest.mock import patch
        from evals.intuitive_physics import eval as dev_eval

        encoder = nn.Linear(2, 2)
        target_encoder = nn.Linear(2, 2)
        predictor = make_predictor()
        expected_weight = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
        expected_bias = torch.tensor([0.5, -0.5])
        checkpoint = {
            "encoder": encoder.state_dict(),
            "target_encoder": target_encoder.state_dict(),
            "predictor": {
                "module.predictor_embed.weight": expected_weight,
                "module.predictor_embed.bias": expected_bias,
                "unrelated.weight": torch.ones(1),
            },
            "epoch": 0,
        }

        with patch.object(dev_eval.torch, "load", return_value=checkpoint):
            dev_eval.load_pretrained(
                encoder,
                target_encoder,
                predictor,
                pretrained="/weights/vjepa.pt",
                load_predictor=False,
                load_predictor_embed=True,
            )

        torch.testing.assert_close(
            predictor.predictor_embed.weight, expected_weight
        )
        torch.testing.assert_close(
            predictor.predictor_embed.bias, expected_bias
        )
        self.assertFalse(
            any(
                parameter.requires_grad
                for parameter in predictor.predictor_embed.parameters()
            )
        )

    def test_causal_window_evaluation_keeps_seven_step_scores(self):
        from evals.causal_recurrent import (
            evaluate_causal_recurrent_windows,
        )

        predictor = make_predictor(num_context_chunks=7)
        context_encoder = RecordingEncoder(tokens=2, dim=2)
        target_encoder = RecordingEncoder(tokens=2, dim=2)
        windows = torch.randn(2, 3, 16, 2, 2)

        result = evaluate_causal_recurrent_windows(
            windows,
            context_encoder,
            target_encoder,
            predictor,
            chunk_frames=2,
            chunk_tokens=2,
            spatial_reduction="mean",
            spatial_top_n=1,
        )

        self.assertEqual(
            tuple(result["primary_step_scores"].shape), (2, 7)
        )
        self.assertEqual(
            tuple(result["copy_step_scores"].shape), (2, 7)
        )
        self.assertFalse(result["primary_step_scores"].requires_grad)

    def test_checkpoint_mode_accepts_causal_recurrent_predictor(self):
        from evals.intuitive_physics import eval as dev_eval

        self.assertTrue(
            dev_eval._is_full_predictor_checkpoint_mode(
                "onn_causal_recurrent"
            )
        )

    def test_test_metrics_exposes_selected_top_n_reduction(self):
        from evals.intphys_test import eval as test_eval

        losses = torch.tensor([[0.1, 0.4, 0.2, 0.3]])
        metrics = test_eval.compute_metrics(
            losses,
            temporal_reduction="top_n",
            temporal_top_n=2,
        )

        torch.testing.assert_close(
            metrics["selected_surprise"], torch.tensor([0.65])
        )


class CausalRecurrentTrainingIntegrationTests(unittest.TestCase):
    def test_mode_is_registered_and_uses_no_mask_collator(self):
        from evals.intuitive_physics import train_optical

        config = {
            "training": {"experiment_mode": "onn_causal_recurrent"},
            "mask_mode": "unified_random",
            "data": {"frames_per_clip": 16, "resolution": 224},
            "pretrain": {"patch_size": 16, "tubelet_size": 2},
        }

        self.assertEqual(
            train_optical._resolve_experiment_mode(config),
            "onn_causal_recurrent",
        )
        self.assertIsNone(train_optical._make_jepa_mask_collator(config))

    def test_trainability_keeps_projector_and_disabled_feedback_frozen(self):
        from evals.intuitive_physics import train_optical

        model = CausalRecurrentONNPredictor(
            embed_dim=2,
            predictor_embed_dim=2,
            num_context_chunks=3,
            chunk_tokens=2,
            memory_lambda=0.5,
            onn_config=small_onn_config(),
        )

        train_optical._set_predictor_trainability(model)

        self.assertFalse(
            any(p.requires_grad for p in model.predictor_embed.parameters())
        )
        for module in (
            model.input_onn,
            model.memory_onn,
            model.prediction_onn,
        ):
            self.assertFalse(module.feedback_gain_raw.requires_grad)
            active = [
                p for name, p in module.named_parameters()
                if name != "feedback_gain_raw"
            ]
            self.assertTrue(active)
            self.assertTrue(all(p.requires_grad for p in active))

    def test_epoch_updates_recurrent_onns_once_per_batch(self):
        import logging
        from evals.intuitive_physics import train_optical

        model = make_predictor(num_context_chunks=7)
        context_encoder = RecordingEncoder(tokens=2, dim=2)
        target_encoder = RecordingEncoder(tokens=2, dim=2)
        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=1.0e-2,
        )
        loader = [(torch.randn(1, 3, 16, 2, 2),)]
        before = model.input_onn.weight.detach().clone()

        metrics = train_optical._run_causal_recurrent_epoch(
            loader=loader,
            args_eval={
                "predictor": {
                    "chunk_frames": 2,
                    "chunk_tokens": 2,
                },
                "loss": {
                    "loss_exp": 1.0,
                    "motion_beta": 1.0,
                    "motion_epsilon": 1.0e-6,
                },
            },
            encoder=context_encoder,
            target_encoder=target_encoder,
            predictor=model,
            optimizer=optimizer,
            device=torch.device("cpu"),
            logger=logging.getLogger("causal-recurrent-test"),
            epoch=1,
            training=True,
        )

        self.assertEqual(metrics["batches"], 1)
        self.assertIn("global_loss", metrics)
        self.assertIn("motion_loss", metrics)
        self.assertFalse(torch.equal(before, model.input_onn.weight))

    def test_checkpoint_records_causal_architecture(self):
        from evals.intuitive_physics import train_optical

        model = make_predictor()
        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=1.0e-2,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda _: 1.0
        )
        args_eval = {
            "predictor_type": "onn_causal_recurrent",
            "mask_mode": "unified_random",
            "training": {
                "experiment_mode": "onn_causal_recurrent",
                "checkpoint_selection": "last_epoch",
            },
            "predictor": {
                "num_chunks": 8,
                "num_context_chunks": 7,
                "chunk_frames": 2,
                "chunk_tokens": 196,
                "predictor_dim": 384,
                "memory_lambda": 0.5,
            },
            "loss": {
                "motion_beta": 1.0,
                "motion_epsilon": 1.0e-6,
            },
            "pretrain": {"folder": "/weights", "checkpoint": "vjepa.pt"},
            "onn": {},
        }

        checkpoint = train_optical._end_to_end_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch=1,
            global_step=2,
            best_val_loss=None,
            split={"train_video_ids": [], "val_video_ids": []},
            split_manifest="/data/linux/wkx/split.json",
            args_eval=args_eval,
            kind="last",
            experiment_mode="onn_causal_recurrent",
        )

        self.assertEqual(checkpoint["mode"], "onn_causal_recurrent")
        self.assertEqual(checkpoint["architecture_version"], 2)
        self.assertEqual(checkpoint["output_dim"], model.embed_dim)
        self.assertEqual(checkpoint["loss_feature_dim"], model.embed_dim)
        self.assertEqual(checkpoint["output_mode"], "linear")
        self.assertFalse(checkpoint["output_layer_norm"])
        self.assertEqual(checkpoint["test_score_mapping"], "reciprocal")
        self.assertEqual(checkpoint["num_context_chunks"], 7)
        self.assertEqual(checkpoint["chunk_frames"], 2)
        self.assertEqual(checkpoint["memory_lambda"], 0.5)
        self.assertEqual(
            checkpoint["primary_test_score"],
            "unweighted_prediction_error",
        )



class CausalRecurrentReviewFixTests(unittest.TestCase):
    def test_test_scores_above_one_preserve_order(self):
        from evals.intphys_test import eval as test_eval
        losses = torch.tensor([[1.1, 1.2], [1.4, 1.5]])
        metrics = test_eval.compute_metrics(
            losses, temporal_reduction="mean", score_scale=2.0
        )
        torch.testing.assert_close(
            metrics["selected_surprise"], torch.tensor([0.425, 0.275])
        )
        self.assertGreater(
            float(metrics["maximum_surprise"][0]),
            float(metrics["maximum_surprise"][1]),
        )

    def test_top_k_alias_and_named_answer_score(self):
        from evals.intphys_test import eval as test_eval
        losses = torch.tensor([[0.1, 0.4, 0.2, 0.3]])
        for mode in ("top_n", "top_k", "top-k"):
            metrics = test_eval.compute_metrics(
                losses, temporal_reduction=mode,
                temporal_top_n=2, score_scale=2.0,
            )
            torch.testing.assert_close(
                metrics["top_2_surprise"], torch.tensor([0.825])
            )
            torch.testing.assert_close(
                metrics["top_2_surprise"], metrics["selected_surprise"]
            )

    def test_dev_selected_top_k_differs_from_mean(self):
        from evals.intuitive_physics import eval as dev_eval
        losses = torch.tensor([
            [0.1, 0.9, 0.2], [0.4, 0.5, 0.6],
            [0.6, 0.7, 0.8], [0.7, 0.8, 0.9],
        ])
        labels = torch.tensor([1, 0, 1, 0])
        metrics = dev_eval.compute_metrics(
            losses, labels, temporal_reduction="top_k", temporal_top_n=1
        )
        self.assertEqual(metrics["Relative Accuracy (avg)"], 100.0)
        self.assertEqual(metrics["Relative Accuracy (selected)"], 50.0)
        official = dev_eval.compute_official_metrics(
            losses.unsqueeze(0), labels.unsqueeze(0),
            temporal_reduction="top_k", temporal_top_n=1,
        )
        self.assertEqual(official["official_mean_relative_accuracy"], 1.0)
        self.assertEqual(official["official_selected_relative_accuracy"], 0.0)

    def test_bfloat16_errors_are_accumulated_in_float32(self):
        p = torch.randn(1, 3, 2, 2, dtype=torch.bfloat16)
        y = torch.randn(1, 4, 2, 2, dtype=torch.bfloat16)
        losses = compute_causal_recurrent_loss(p, y)
        scores = compute_causal_recurrent_scores(p, y)
        self.assertEqual(losses["token_error"].dtype, torch.float32)
        self.assertEqual(scores["token_error"].dtype, torch.float32)

    def test_training_and_validation_honor_bfloat16_flag(self):
        import logging
        from evals.intuitive_physics import train_optical
        model = make_predictor(num_context_chunks=7)
        dtypes = []
        hook = model.input_onn.register_forward_pre_hook(
            lambda module, inputs: dtypes.append(inputs[0].dtype)
        )
        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad], lr=0.01
        )
        config = {
            "data": {"use_bfloat16": True},
            "predictor": {"chunk_frames": 2, "chunk_tokens": 2},
        }
        try:
            for training in (True, False):
                train_optical._run_causal_recurrent_epoch(
                    loader=[(torch.randn(1, 3, 16, 2, 2),)],
                    args_eval=config,
                    encoder=RecordingEncoder(tokens=2, dim=2),
                    target_encoder=RecordingEncoder(tokens=2, dim=2),
                    predictor=model, optimizer=optimizer,
                    device=torch.device("cpu"),
                    logger=logging.getLogger("causal-review-test"),
                    epoch=1, training=training,
                )
        finally:
            hook.remove()
        self.assertEqual(len(dtypes), 14)
        self.assertTrue(all(dtype == torch.bfloat16 for dtype in dtypes))

class CausalRecurrentLinearOutputTests(unittest.TestCase):
    def make_model(self, steps=3):
        model = CausalRecurrentONNPredictor(
            embed_dim=4, predictor_embed_dim=2,
            num_context_chunks=steps, chunk_tokens=2,
            memory_lambda=0.4, onn_config=small_onn_config(),
            input_onn=ScaleONN(1.0), memory_onn=ScaleONN(1.0),
            prediction_onn=ScaleONN(1.0),
        )
        model.load_predictor_embed_state_dict({
            "predictor_embed.weight": torch.tensor(
                [[1., 0., 0., 0.], [0., 1., 0., 0.]]
            ),
            "predictor_embed.bias": torch.zeros(2),
        })
        return model

    def test_linear_restores_encoder_width_without_final_layer_norm(self):
        model = self.make_model()
        with torch.no_grad():
            model.output_linear.bias.fill_(10.)
        context = torch.randn(2, 3, 2, 4)
        predictions, states = model(context, return_states=True)
        self.assertEqual(tuple(predictions.shape), (2, 3, 2, 4))
        self.assertEqual(tuple(states.shape), (2, 3, 2, 2))
        expected = model.output_linear(torch.nn.functional.layer_norm(
            model.prediction_onn(states[:, -1]), (2,)
        ))
        torch.testing.assert_close(predictions[:, -1], expected)
        self.assertTrue((predictions > 5.).all())
        self.assertTrue(all(p.requires_grad for p in model.output_linear.parameters()))

    def test_targets_keep_full_encoder_space_and_are_detached(self):
        model = self.make_model()
        targets = torch.randn(1, 4, 2, 4, requires_grad=True) + 10.
        prepared = model.prepare_target_features(targets)
        torch.testing.assert_close(prepared, targets)
        self.assertEqual(tuple(prepared.shape), (1, 4, 2, 4))
        self.assertFalse(prepared.requires_grad)

    def test_final_step_backpropagates_to_head_and_early_history(self):
        torch.manual_seed(71)
        model = self.make_model()
        context = torch.randn(1, 3, 2, 4, requires_grad=True)
        predictions = model(context)
        predictions[:, -1].square().mean().backward()
        self.assertGreater(float(context.grad[:, 0].abs().sum()), 0.)
        self.assertIsNone(model.predictor_embed.weight.grad)
        for module in (model.input_onn, model.memory_onn,
                       model.prediction_onn, model.output_linear):
            self.assertIsNotNone(module.weight.grad)
            self.assertGreater(float(module.weight.grad.abs().sum()), 0.)

    def test_evaluation_scores_use_uncompressed_targets(self):
        from evals.causal_recurrent import evaluate_causal_recurrent_windows
        model = self.make_model(steps=7)
        context_encoder = RecordingEncoder(tokens=2, dim=4)
        target_encoder = RecordingEncoder(tokens=2, dim=4)
        windows = torch.randn(1, 3, 16, 2, 2)
        context, targets = encode_independent_chunks(
            windows, context_encoder, target_encoder, chunk_tokens=2,
        )
        expected = compute_causal_recurrent_scores(model(context), targets)
        actual = evaluate_causal_recurrent_windows(
            windows, context_encoder, target_encoder, model, chunk_tokens=2,
        )
        for key in ("token_error", "motion", "copy_token_error"):
            torch.testing.assert_close(actual[key], expected[key])

    def test_causal_answer_mapping_preserves_errors_above_two(self):
        from evals.intphys_test import eval as test_eval
        losses = torch.tensor([[2., 4., 6.], [5., 7., 9.]])
        for mode, expected_loss in (
            ("mean", torch.tensor([4., 7.])),
            ("max", torch.tensor([6., 9.])),
            ("top_k", torch.tensor([5., 8.])),
        ):
            metrics = test_eval.compute_metrics(
                losses, temporal_reduction=mode, temporal_top_n=2,
                score_mapping="reciprocal",
            )
            torch.testing.assert_close(
                metrics["selected_surprise"], 1. / (1. + expected_loss),
            )
            torch.testing.assert_close(
                metrics["average_surprise"], 1. / (1. + losses.mean(1)),
            )
            torch.testing.assert_close(
                metrics["maximum_surprise"], 1. / (1. + losses.max(1).values),
            )
            self.assertGreater(float(metrics["selected_surprise"][0]),
                               float(metrics["selected_surprise"][1]))
            self.assertTrue((metrics["selected_surprise"] > 0.).all())

    def test_training_epoch_updates_linear_head_in_encoder_space(self):
        import logging
        from unittest.mock import patch
        from evals.intuitive_physics import train_optical
        model = self.make_model(steps=7)
        before = model.output_linear.weight.detach().clone()
        optimizer = torch.optim.SGD(
            [p for p in model.parameters() if p.requires_grad], lr=0.01,
        )
        shapes = []
        def record_loss(predictions, targets, **kwargs):
            shapes.append((tuple(predictions.shape), tuple(targets.shape)))
            return compute_causal_recurrent_loss(predictions, targets, **kwargs)
        with patch.object(train_optical, "compute_causal_recurrent_loss",
                          side_effect=record_loss):
            for training in (True, False):
                train_optical._run_causal_recurrent_epoch(
                    loader=[(torch.randn(1, 3, 16, 2, 2),)],
                    args_eval={"predictor": {"chunk_frames": 2, "chunk_tokens": 2}},
                    encoder=RecordingEncoder(tokens=2, dim=4),
                    target_encoder=RecordingEncoder(tokens=2, dim=4),
                    predictor=model, optimizer=optimizer,
                    device=torch.device("cpu"),
                    logger=logging.getLogger("causal-linear-test"),
                    epoch=1, training=training,
                )
        self.assertEqual(shapes, [((1, 7, 2, 4), (1, 8, 2, 4))] * 2)
        self.assertFalse(torch.equal(before, model.output_linear.weight))
        self.assertIsNone(model.predictor_embed.weight.grad)

    def test_real_three_slm_onns_receive_gradient_through_linear_head(self):
        from dataclasses import replace
        config = replace(small_onn_config(dim=4), num_slm_layers=3,
                         slm_intervals_um=(8., 8.))
        torch.manual_seed(72)
        model = CausalRecurrentONNPredictor(
            embed_dim=8, predictor_embed_dim=4, num_context_chunks=7,
            chunk_tokens=2, memory_lambda=0.4, onn_config=config,
        )
        predictions = model(torch.randn(1, 7, 2, 8))
        self.assertEqual(tuple(predictions.shape), (1, 7, 2, 8))
        targets = torch.randn(1, 8, 2, 8)
        result = compute_causal_recurrent_loss(predictions, targets)
        result["loss"].backward()
        for module in (model.input_onn, model.memory_onn,
                       model.prediction_onn, model.output_linear):
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(float(g.abs().sum()) for g in grads), 0.)

    def test_training_config_has_current_weight_point_six(self):
        from pathlib import Path
        import yaml
        config_path = Path(__file__).resolve().parents[1] / (
            "evals/intuitive_physics/configs/onn_causal_recurrent_slm3.yaml"
        )
        config = yaml.safe_load(config_path.read_text())
        self.assertAlmostEqual(config["predictor"]["memory_lambda"], 0.4)
        self.assertEqual(config["evaluation"]["spatial_reduction"], "mean")
        self.assertEqual(config["evaluation"]["temporal_reduction"], "mean")


if __name__ == "__main__":
    unittest.main()
