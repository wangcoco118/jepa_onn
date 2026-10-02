"""Shared encoding, loss, and scoring utilities for causal recurrent ONN."""

import torch
import torch.nn.functional as F


def _validate_prediction_targets(predictions, targets):
    if predictions.ndim != 4 or targets.ndim != 4:
        raise ValueError("predictions and targets must both be rank-4 tensors")
    if predictions.shape[0] != targets.shape[0]:
        raise ValueError("prediction and target batch sizes must match")
    if targets.shape[1] != predictions.shape[1] + 1:
        raise ValueError("targets must contain one more chunk than predictions")
    if predictions.shape[2:] != targets.shape[2:]:
        raise ValueError("prediction and target token/feature shapes must match")


@torch.no_grad()
def encode_independent_chunks(
    clips,
    encoder,
    target_encoder,
    chunk_frames=2,
    chunk_tokens=196,
):
    """Encode each chunk as an independent batch item with no cross-chunk attention."""
    if clips.ndim != 5:
        raise ValueError("clips must have shape [B,C,T,H,W]")
    batch_size, channels, frames, height, width = clips.shape
    chunk_frames = int(chunk_frames)
    chunk_tokens = int(chunk_tokens)
    if chunk_frames <= 0 or frames % chunk_frames != 0:
        raise ValueError("frames must be divisible by a positive chunk_frames")
    num_chunks = frames // chunk_frames
    if num_chunks != 8:
        raise ValueError("causal recurrent ONN requires exactly 8 chunks")

    chunks = clips.unfold(2, chunk_frames, chunk_frames)
    chunks = chunks.permute(0, 2, 1, 5, 3, 4).contiguous()
    context_clips = chunks[:, :-1].reshape(
        batch_size * (num_chunks - 1),
        channels,
        chunk_frames,
        height,
        width,
    )
    target_clips = chunks.reshape(
        batch_size * num_chunks,
        channels,
        chunk_frames,
        height,
        width,
    )

    context_mask = torch.arange(
        chunk_tokens, device=clips.device, dtype=torch.long
    ).unsqueeze(0).repeat(context_clips.shape[0], 1)
    target_mask = torch.arange(
        chunk_tokens, device=clips.device, dtype=torch.long
    ).unsqueeze(0).repeat(target_clips.shape[0], 1)

    context_output = encoder(context_clips, [context_mask])
    target_output = target_encoder(target_clips, [target_mask])
    context = context_output[0] if isinstance(context_output, (list, tuple)) else context_output
    target = target_output[0] if isinstance(target_output, (list, tuple)) else target_output

    if context.ndim != 3 or context.shape[1] != chunk_tokens:
        raise ValueError("context encoder returned an unexpected token shape")
    if target.ndim != 3 or target.shape[1] != chunk_tokens:
        raise ValueError("target encoder returned an unexpected token shape")
    if context.shape[-1] != target.shape[-1]:
        raise ValueError("context and target encoder dimensions must match")

    target = F.layer_norm(target, (target.shape[-1],))
    feature_dim = context.shape[-1]
    context = context.reshape(
        batch_size, num_chunks - 1, chunk_tokens, feature_dim
    )
    target = target.reshape(
        batch_size, num_chunks, chunk_tokens, feature_dim
    )
    return context, target


@torch.no_grad()
def evaluate_causal_recurrent_windows(
    windows,
    encoder,
    target_encoder,
    predictor,
    chunk_frames=2,
    chunk_tokens=196,
    motion_epsilon=1.0e-6,
    spatial_reduction="mean",
    spatial_top_n=20,
):
    context_features, target_features = encode_independent_chunks(
        windows,
        encoder,
        target_encoder,
        chunk_frames=chunk_frames,
        chunk_tokens=chunk_tokens,
    )
    predictions = predictor(context_features)
    predictor_model = (
        predictor.module if hasattr(predictor, "module") else predictor
    )
    prepared_targets = predictor_model.prepare_target_features(
        target_features
    )
    return compute_causal_recurrent_scores(
        predictions,
        prepared_targets,
        motion_epsilon=motion_epsilon,
        spatial_reduction=spatial_reduction,
        spatial_top_n=spatial_top_n,
    )


def compute_causal_recurrent_loss(
    predictions,
    targets,
    loss_exp=1.0,
    motion_beta=1.0,
    motion_epsilon=1.0e-6,
):
    """Combine the seven next-chunk errors with detached motion weighting."""
    _validate_prediction_targets(predictions, targets)
    predictions = predictions.float()
    targets = targets.float()
    loss_exp = float(loss_exp)
    motion_beta = float(motion_beta)
    motion_epsilon = float(motion_epsilon)
    if loss_exp <= 0.0:
        raise ValueError("loss_exp must be positive")
    if motion_beta < 0.0:
        raise ValueError("motion_beta must be non-negative")
    if motion_epsilon <= 0.0:
        raise ValueError("motion_epsilon must be positive")

    next_targets = targets[:, 1:]
    token_error = (
        torch.abs(predictions - next_targets).pow(loss_exp).mean(dim=-1)
        / loss_exp
    )
    global_loss = token_error.mean()

    motion = torch.abs(
        targets[:, 1:] - targets[:, :-1]
    ).mean(dim=-1).detach()
    motion_weights = (
        motion_epsilon
        + motion
        / (motion.mean(dim=-1, keepdim=True) + motion_epsilon)
    ).detach()
    motion_loss = (
        (motion_weights * token_error).sum(dim=-1)
        / motion_weights.sum(dim=-1)
    ).mean()
    total_loss = global_loss + motion_beta * motion_loss
    return {
        "loss": total_loss,
        "global_loss": global_loss,
        "motion_loss": motion_loss,
        "token_error": token_error,
        "motion": motion,
        "motion_weights": motion_weights,
    }


def reduce_scores(values, mode, dim, top_n):
    mode = str(mode).strip().lower().replace("-", "_")
    if mode == "mean":
        return values.mean(dim=dim)
    if mode == "max":
        return values.max(dim=dim).values
    if mode in {"top_n", "top_k"}:
        k = min(int(top_n), values.shape[dim])
        if k <= 0:
            raise ValueError("top_n must be positive")
        return values.topk(k, dim=dim).values.mean(dim=dim)
    raise ValueError(f"unsupported score reduction: {mode}")


def compute_causal_recurrent_scores(
    predictions,
    targets,
    motion_epsilon=1.0e-6,
    spatial_reduction="mean",
    spatial_top_n=20,
):
    """Return unweighted primary scores plus diagnostic motion/copy scores."""
    _validate_prediction_targets(predictions, targets)
    predictions = predictions.float()
    targets = targets.float()
    motion_epsilon = float(motion_epsilon)
    if motion_epsilon <= 0.0:
        raise ValueError("motion_epsilon must be positive")

    next_targets = targets[:, 1:]
    token_error = torch.abs(predictions - next_targets).mean(dim=-1)
    motion = torch.abs(
        targets[:, 1:] - targets[:, :-1]
    ).mean(dim=-1).detach()
    motion_weights = (
        motion_epsilon
        + motion
        / (motion.mean(dim=-1, keepdim=True) + motion_epsilon)
    ).detach()
    copy_token_error = torch.abs(
        targets[:, :-1] - next_targets
    ).mean(dim=-1)

    primary_step_scores = reduce_scores(
        token_error,
        spatial_reduction,
        dim=-1,
        top_n=spatial_top_n,
    )
    copy_step_scores = reduce_scores(
        copy_token_error,
        spatial_reduction,
        dim=-1,
        top_n=spatial_top_n,
    )
    motion_only_step_scores = reduce_scores(
        motion,
        spatial_reduction,
        dim=-1,
        top_n=spatial_top_n,
    )
    motion_weighted_step_scores = (
        (motion_weights * token_error).sum(dim=-1)
        / motion_weights.sum(dim=-1)
    )
    return {
        "token_error": token_error,
        "primary_step_scores": primary_step_scores,
        "motion": motion,
        "motion_weights": motion_weights,
        "motion_weighted_step_scores": motion_weighted_step_scores,
        "motion_only_step_scores": motion_only_step_scores,
        "copy_token_error": copy_token_error,
        "copy_step_scores": copy_step_scores,
    }
