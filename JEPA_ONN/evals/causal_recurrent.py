"""Shared encoding, loss, and scoring utilities for causal recurrent ONN."""

import torch
import torch.nn.functional as F


def resolve_causal_mode(predictor_config=None):
    cfg = predictor_config or {}
    source = cfg.get("feature_source", "legacy_dual")
    objective = cfg.get("objective", "next_chunk")
    if source not in {"legacy_dual", "shared_target"}:
        raise ValueError("feature_source must be legacy_dual or shared_target")
    if objective not in {"next_chunk", "reconstruct_current"}:
        raise ValueError("objective must be next_chunk or reconstruct_current")
    return source, objective


def _positive_step(value, label):
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(label + " must be a scalar or single-element list")
        value = value[0]
    step = int(value)
    if step <= 0 or step != float(value):
        raise ValueError(label + " must be a positive integer")
    return step


def validate_causal_config(args_eval, ordinary_test=False):
    cfg = args_eval.get("predictor") or {}
    source, objective = resolve_causal_mode(cfg)
    if cfg.get("freeze_predictor_embed", True) is not True:
        raise ValueError("causal input Linear must remain frozen")
    if source == "shared_target":
        data = args_eval.get("data") or {}
        if "sampling_rate" not in data or "frame_steps" not in data:
            raise ValueError("shared_target requires explicit sampling_rate and frame_steps")
        train_step = _positive_step(data["sampling_rate"], "sampling_rate")
        test_step = _positive_step(data["frame_steps"], "frame_steps")
        if train_step != test_step:
            raise ValueError("sampling_rate and frame_steps must match for shared_target")
    if objective == "reconstruct_current":
        loss = args_eval.get("loss") or {}
        if float(loss.get("motion_beta", 1.0)) != 0.0:
            raise ValueError("reconstruct_current requires motion_beta=0")
        if float(loss.get("loss_exp", 1.0)) != 1.0:
            raise ValueError("reconstruct_current requires loss_exp=1 (plain L1)")
        if ordinary_test:
            raise ValueError("reconstruction cannot enter ordinary next-chunk IntPhys Test; "
                             "use validate_causal_reconstruction")
    return source, objective


def causal_runtime_metadata(args_eval):
    source, objective = validate_causal_config(args_eval)
    data = args_eval.get("data") or {}
    pretrain = args_eval.get("pretrain") or {}
    sampling = data.get("sampling_rate")
    if sampling is None:
        sampling = pretrain.get("sampling_rate", 4)
    if isinstance(sampling, (list, tuple)):
        sampling = sampling[0]
    context_key = ("target_encoder" if source == "shared_target"
                   else pretrain.get("enc_checkpoint_key", "encoder"))
    return {
        "feature_source": source,
        "objective": objective,
        "experiment_version": "V2" if source == "shared_target" else "V1",
        "input_linear_frozen": True,
        "encoder_weight_source": {
            "context": context_key, "target": "target_encoder",
            "pretrained": str(pretrain.get("folder", "")).rstrip("/") + "/"
                          + str(pretrain.get("checkpoint", "")),
        },
        "normalization_rule": (
            "shared_target_ln1024_then_projected_ln384"
            if source == "shared_target"
            else "legacy_context_raw_target_ln1024_projected_ln384"
        ),
        "training_sampling_rate": int(sampling),
        "evaluation_frame_steps": data.get("frame_steps", 4),
        "supervised_chunks": "1..7 (chunk 8 excluded)" if objective == "reconstruct_current"
                             else "2..8",
    }


def validate_causal_checkpoint(checkpoint, args_eval):
    source, objective = validate_causal_config(args_eval)
    saved = resolve_causal_mode({
        "feature_source": checkpoint.get("feature_source", "legacy_dual"),
        "objective": checkpoint.get("objective", "next_chunk"),
    })
    if saved != (source, objective):
        raise ValueError("causal checkpoint mode mismatch: "
                         f"saved={saved}, requested={(source, objective)}; "
                         "migration is not resume")
    saved_config = checkpoint.get("config") or {}
    if saved_config and resolve_causal_mode(saved_config.get("predictor")) != saved:
        raise ValueError("checkpoint mode metadata disagrees with saved config")
    if checkpoint.get("input_linear_frozen", True) is not True:
        raise ValueError("checkpoint input Linear is not frozen")
    expected = causal_runtime_metadata(args_eval)
    for key in ("normalization_rule", "encoder_weight_source", "training_sampling_rate"):
        if key in checkpoint and checkpoint[key] != expected[key]:
            raise ValueError("causal checkpoint metadata mismatch: " + key)


@torch.no_grad()
def compute_reconstruction_metrics(predictions, targets):
    _validate_prediction_targets(predictions, targets)
    p, z = predictions.detach().float(), targets[:, :-1].detach().float()
    step = (p - z).abs().mean(dim=(0, 2, 3))
    return {
        "reconstruction_l1": step.mean().item(),
        "per_step_l1": step.tolist(),
        "first_step_l1": step[0].item(),
        "prediction_std": p.std(unbiased=False).item(),
        "target_std": z.std(unbiased=False).item(),
        "prediction_variation_std": p.std(dim=(0, 1, 2), unbiased=False).mean().item(),
        "target_variation_std": z.std(dim=(0, 1, 2), unbiased=False).mean().item(),
    }


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
    feature_source="legacy_dual",
):
    """Encode each chunk as an independent batch item with no cross-chunk attention."""
    resolve_causal_mode({"feature_source": feature_source})
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

    if feature_source == "shared_target":
        for module in (encoder, target_encoder):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        mask = torch.arange(chunk_tokens, device=clips.device).unsqueeze(0).repeat(
            target_clips.shape[0], 1)
        encoded = target_encoder(target_clips, [mask])
        encoded = encoded[0] if isinstance(encoded, (list, tuple)) else encoded
        if encoded.ndim != 3 or encoded.shape[1] != chunk_tokens:
            raise ValueError("target encoder returned an unexpected token shape")
        target = F.layer_norm(encoded, (encoded.shape[-1],)).reshape(
            batch_size, num_chunks, chunk_tokens, encoded.shape[-1])
        return target[:, :-1], target

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
    predictor_model = predictor.module if hasattr(predictor, "module") else predictor
    if getattr(predictor_model, "objective", "next_chunk") != "next_chunk":
        raise ValueError("reconstruction cannot enter next-chunk scoring")
    context_features, target_features = encode_independent_chunks(
        windows,
        encoder,
        target_encoder,
        chunk_frames=chunk_frames,
        chunk_tokens=chunk_tokens,
        feature_source=getattr(predictor_model, "feature_source", "legacy_dual"),
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
    objective="next_chunk",
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

    resolve_causal_mode({"objective": objective})
    if objective == "reconstruct_current":
        if motion_beta != 0.0 or loss_exp != 1.0:
            raise ValueError("reconstruct_current requires motion_beta=0 and loss_exp=1")
        token_error = torch.abs(predictions - targets[:, :-1]).mean(dim=-1)
        loss = token_error.mean()
        return {"loss": loss, "global_loss": loss,
                "motion_loss": loss.detach().new_zeros(()), "token_error": token_error,
                "motion": torch.zeros_like(token_error),
                "motion_weights": torch.ones_like(token_error)}
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
