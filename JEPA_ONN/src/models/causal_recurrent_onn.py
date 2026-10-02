"""Causal recurrent ONN Predictor for independent video chunks."""

from dataclasses import replace
import math
from typing import Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.fsonn import FeedbackFSONN, ONNConfig


class CausalRecurrentONNPredictor(nn.Module):
    """Predict every next chunk from current evidence and recurrent ONN state."""

    def __init__(
        self,
        embed_dim: int,
        predictor_embed_dim: int,
        num_context_chunks: int,
        chunk_tokens: int,
        memory_lambda: float,
        onn_config: ONNConfig,
        input_onn: Optional[nn.Module] = None,
        memory_onn: Optional[nn.Module] = None,
        prediction_onn: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.predictor_embed_dim = int(predictor_embed_dim)
        self.num_context_chunks = int(num_context_chunks)
        self.chunk_tokens = int(chunk_tokens)
        if self.embed_dim <= 0 or self.predictor_embed_dim <= 0:
            raise ValueError("feature dimensions must be positive")
        if self.num_context_chunks <= 0 or self.chunk_tokens <= 0:
            raise ValueError("chunk dimensions must be positive")
        if (
            onn_config.input_dim != self.predictor_embed_dim
            or onn_config.output_dim != self.predictor_embed_dim
            or onn_config.chunk_tokens != self.chunk_tokens
        ):
            raise ValueError(
                "ONN dimensions must match predictor_embed_dim and chunk_tokens"
            )
        memory_lambda = float(memory_lambda)
        if not math.isfinite(memory_lambda) or not 0.0 <= memory_lambda <= 1.0:
            raise ValueError("memory_lambda must be finite and satisfy 0 <= value <= 1")

        self.predictor_embed = nn.Linear(
            self.embed_dim, self.predictor_embed_dim, bias=True
        )
        for parameter in self.predictor_embed.parameters():
            parameter.requires_grad_(False)

        # Match the existing Linear output mode: restore the encoder width.
        self.output_linear = nn.Linear(
            self.predictor_embed_dim, self.embed_dim, bias=True
        )

        base_config = replace(
            onn_config,
            feedback_enabled=False,
            feedback_memory_enabled=False,
        )
        self.input_onn = input_onn or FeedbackFSONN(base_config)
        self.memory_onn = memory_onn or FeedbackFSONN(base_config)
        self.prediction_onn = prediction_onn or FeedbackFSONN(base_config)
        if len(
            {
                id(self.input_onn),
                id(self.memory_onn),
                id(self.prediction_onn),
            }
        ) != 3:
            raise ValueError("input, memory, and prediction ONNs must be distinct")

        for module in (
            self.input_onn,
            self.memory_onn,
            self.prediction_onn,
        ):
            gain = getattr(module, "feedback_gain_raw", None)
            config = getattr(module, "config", None)
            if gain is not None and config is not None and not config.feedback_enabled:
                gain.requires_grad_(False)

        self.register_buffer(
            "memory_lambda",
            torch.tensor(memory_lambda, dtype=torch.float32),
            persistent=True,
        )

    def load_predictor_embed_state_dict(
        self,
        pretrained_predictor_state: Mapping[str, torch.Tensor],
    ) -> None:
        weight_keys = [
            key
            for key in pretrained_predictor_state
            if key.endswith("predictor_embed.weight")
        ]
        bias_keys = [
            key
            for key in pretrained_predictor_state
            if key.endswith("predictor_embed.bias")
        ]
        if len(weight_keys) != 1 or len(bias_keys) != 1:
            raise RuntimeError(
                "expected exactly one predictor_embed weight and bias, got "
                f"weights={weight_keys}, biases={bias_keys}"
            )
        weight = pretrained_predictor_state[weight_keys[0]]
        bias = pretrained_predictor_state[bias_keys[0]]
        if weight.shape != self.predictor_embed.weight.shape:
            raise RuntimeError(
                "predictor_embed weight shape mismatch: "
                f"expected {tuple(self.predictor_embed.weight.shape)}, "
                f"got {tuple(weight.shape)}"
            )
        if bias.shape != self.predictor_embed.bias.shape:
            raise RuntimeError(
                "predictor_embed bias shape mismatch: "
                f"expected {tuple(self.predictor_embed.bias.shape)}, "
                f"got {tuple(bias.shape)}"
            )
        with torch.no_grad():
            self.predictor_embed.weight.copy_(weight)
            self.predictor_embed.bias.copy_(bias)
        for parameter in self.predictor_embed.parameters():
            parameter.requires_grad_(False)

    def _project_features(
        self,
        features: torch.Tensor,
        expected_chunks: int,
        label: str,
    ) -> torch.Tensor:
        expected = (
            expected_chunks,
            self.chunk_tokens,
            self.embed_dim,
        )
        if features.ndim != 4 or tuple(features.shape[1:]) != expected:
            raise ValueError(
                f"{label} must have shape [B,{expected_chunks},"
                f"{self.chunk_tokens},{self.embed_dim}]"
            )
        projected = self.predictor_embed(features)
        return F.layer_norm(projected, (self.predictor_embed_dim,))

    def project_context_features(self, features: torch.Tensor) -> torch.Tensor:
        return self._project_features(
            features,
            expected_chunks=self.num_context_chunks,
            label="context features",
        )

    def prepare_target_features(self, features: torch.Tensor) -> torch.Tensor:
        """Keep frozen encoder targets in the original feature space."""
        expected = (
            self.num_context_chunks + 1, self.chunk_tokens, self.embed_dim
        )
        if features.ndim != 4 or tuple(features.shape[1:]) != expected:
            raise ValueError(
                "target features must have shape "
                f"[B,{expected[0]},{expected[1]},{expected[2]}]"
            )
        return features.detach()

    def forward_projected(
        self,
        projected_context: torch.Tensor,
        return_states: bool = False,
    ):
        expected = (
            self.num_context_chunks,
            self.chunk_tokens,
            self.predictor_embed_dim,
        )
        if (
            projected_context.ndim != 4
            or tuple(projected_context.shape[1:]) != expected
        ):
            raise ValueError(
                "projected context must have shape "
                f"[B,{self.num_context_chunks},{self.chunk_tokens},"
                f"{self.predictor_embed_dim}]"
            )

        predictions = []
        states = []
        hidden = self.input_onn(projected_context[:, 0])
        states.append(hidden)

        for step in range(self.num_context_chunks):
            prediction = self.prediction_onn(hidden)
            prediction = F.layer_norm(
                prediction, (self.predictor_embed_dim,)
            )
            # No LayerNorm after Linear: learn target amplitude and offset.
            prediction = self.output_linear(prediction)
            predictions.append(prediction)

            if step + 1 < self.num_context_chunks:
                current = self.input_onn(projected_context[:, step + 1])
                memory = self.memory_onn(hidden)
                memory_lambda = self.memory_lambda.to(
                    device=hidden.device,
                    dtype=hidden.dtype,
                )
                hidden = (
                    (1.0 - memory_lambda) * current
                    + memory_lambda * memory
                )
                states.append(hidden)

        prediction_tensor = torch.stack(predictions, dim=1)
        if return_states:
            return prediction_tensor, torch.stack(states, dim=1)
        return prediction_tensor

    def forward(
        self,
        context_features: torch.Tensor,
        return_states: bool = False,
    ):
        projected = self.project_context_features(context_features)
        return self.forward_projected(
            projected,
            return_states=return_states,
        )
