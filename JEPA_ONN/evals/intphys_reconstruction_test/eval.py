"""Independent current-chunk reconstruction Test for IntPhys O1/O2/O3.

Shares data/model/metric helpers, never calls or relaxes the prediction Test entry.
Reconstruction errors are anomaly scores, not future-prediction errors.
"""
import argparse
import copy
import hashlib
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import torch

from evals.causal_recurrent import (
    causal_runtime_metadata,
    compute_causal_recurrent_loss,
    encode_independent_chunks,
    validate_causal_checkpoint,
    validate_causal_config,
)
from src.utils.amp import autocast_context

KIND = "reconstruction_test"
ALLOWED_ROOTS = (Path("/data/linux/wkx"), Path("/data/wkx"))


def validate_test_checkpoint(checkpoint):
    if checkpoint.get("mode") != "onn_causal_recurrent":
        raise ValueError("reconstruction Test requires an onn_causal_recurrent checkpoint")
    if not isinstance(checkpoint.get("config"), dict) or "predictor" not in checkpoint:
        raise ValueError("reconstruction Test requires a full checkpoint with config and predictor")
    config = copy.deepcopy(checkpoint["config"])
    _, objective = validate_causal_config(config)
    validate_causal_checkpoint(checkpoint, config)
    if objective != "reconstruct_current":
        raise ValueError("checkpoint was not trained for reconstruct_current")
    if config.get("predictor_type") != "onn_causal_recurrent":
        raise ValueError("checkpoint predictor_type must be onn_causal_recurrent")
    if int(config.get("data", {}).get("frames_per_clip", 16)) != 16:
        raise ValueError("reconstruction Test requires 16-frame windows")
    predictor = config.get("predictor") or {}
    if int(predictor.get("chunk_frames", 2)) != 2 or int(predictor.get("num_context_chunks", 7)) != 7:
        raise ValueError("reconstruction Test requires seven two-frame inputs")
    return config


@torch.no_grad()
def reconstruction_step_scores(predictions, targets):
    # This is the exact training objective; no next-chunk shift or motion weighting.
    losses = compute_causal_recurrent_loss(
        predictions, targets, objective="reconstruct_current", loss_exp=1.0, motion_beta=0.0
    )
    return losses["token_error"].mean(dim=-1)


@torch.no_grad()
def score_video_batch(clips, encoder, target_encoder, predictor, *,
                      window_batch_size=15, stride=2, use_bfloat16=False, device=None):
    model = predictor.module if hasattr(predictor, "module") else predictor
    if getattr(model, "objective", "next_chunk") != "reconstruct_current":
        raise ValueError("only reconstruct_current models can enter reconstruction Test")
    if window_batch_size <= 0 or stride <= 0:
        raise ValueError("window_batch_size and stride must be positive")
    if clips.ndim != 5 or clips.shape[2] < 16:
        raise ValueError("clips must be [B,C,T,H,W] with at least 16 sampled frames")
    if device is None:
        device = next(model.parameters(), torch.empty(0, device=clips.device)).device
    device = torch.device(device)
    for module in (encoder, target_encoder, predictor):
        module.eval()
    # Keep a strided view; materialize only one window microbatch, not all B*W windows.
    windows = clips.unfold(2, 16, stride).permute(0, 2, 1, 5, 3, 4)
    videos, count = windows.shape[:2]
    scores = []
    for start in range(0, videos * count, window_batch_size):
        end = min(start + window_batch_size, videos * count)
        block = torch.stack([windows[i // count, i % count] for i in range(start, end)])
        block = block.to(device, non_blocking=True)
        with autocast_context(device, use_bfloat16):
            context, targets = encode_independent_chunks(
                block, encoder, target_encoder, chunk_frames=2,
                chunk_tokens=model.chunk_tokens,
                feature_source=getattr(model, "feature_source", "legacy_dual"),
            )
            predictions = predictor(context)
            targets = model.prepare_target_features(targets) if hasattr(model, "prepare_target_features") else targets
            score = reconstruction_step_scores(predictions, targets)
        if not torch.isfinite(score).all():
            raise ValueError("non-finite reconstruction error; stopping without official answers")
        scores.append(score.detach().cpu())
    return torch.cat(scores).reshape(videos, count, 7)


def answer_metrics(step_scores, temporal_reduction="mean", temporal_top_n=2):
    if step_scores.ndim != 3 or step_scores.shape[-1] != 7:
        raise ValueError("scores must be [videos,windows,7]")
    if not torch.isfinite(step_scores).all() or (step_scores < 0).any():
        raise ValueError("reconstruction scores must be finite non-negative errors")
    # Reuse the prediction entry's pure aggregation/answer mapping, not its inference.
    from evals.intphys_test.eval import compute_metrics
    return compute_metrics(
        step_scores.float().flatten(1), temporal_reduction=temporal_reduction,
        temporal_top_n=temporal_top_n, score_mapping="reciprocal",
    )


def answer_rows(tasks, indices, values):
    indices = torch.as_tensor(indices).long().flatten().tolist()
    values = torch.as_tensor(values).float().flatten().tolist()
    if len(indices) != len(tasks) or sorted(indices) != list(range(len(tasks))):
        raise ValueError("official answers require every unique task ID exactly once")
    if len(values) != len(indices):
        raise ValueError("answer values do not match task IDs")
    ordered = dict(zip(indices, values))
    return [f"{task} {ordered[i]:.9g}\n" for i, task in enumerate(tasks)]


def prepare_output(output_dir, metadata, resume=False):
    folder = Path(output_dir).resolve()
    if not any(folder.is_relative_to(root.resolve()) and folder != root.resolve()
               for root in ALLOWED_ROOTS):
        raise ValueError("output directory must be under /data/linux/wkx or /data/wkx")
    manifest = folder / "test_metadata.json"
    if folder.exists() and any(folder.iterdir()):
        if not resume or not manifest.is_file():
            raise ValueError("use an empty independent output directory, not a prediction/training directory")
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if saved != metadata or saved.get("evaluation_kind") != KIND:
            raise ValueError("reconstruction output metadata mismatch; prediction caches cannot be reused")
    else:
        folder.mkdir(parents=True, exist_ok=True)
        with manifest.open("x", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2)
    return folder


def validate_progress(state, metadata, total_batches, total_tasks):
    if state.get("metadata") != metadata or metadata.get("evaluation_kind") != KIND:
        raise ValueError("reconstruction resume metadata mismatch")
    next_batch = state.get("next_batch", -1)
    scores, indices = state.get("scores"), state.get("indices")
    if not isinstance(next_batch, int) or not 0 <= next_batch <= total_batches:
        raise ValueError("invalid reconstruction resume batch")
    if not isinstance(scores, list) or not isinstance(indices, list) or len(scores) != next_batch or len(indices) != next_batch:
        raise ValueError("reconstruction resume is missing completed batches")
    count = 0
    windows = None
    for value, ids in zip(scores, indices):
        if not torch.is_tensor(value) or value.ndim != 3 or value.shape[-1] != 7:
            raise ValueError("invalid reconstruction resume score shape")
        if not torch.isfinite(value).all() or (value < 0).any():
            raise ValueError("invalid reconstruction resume score values")
        if windows is not None and value.shape[1] != windows:
            raise ValueError("inconsistent reconstruction window counts")
        windows = value.shape[1]
        ids = torch.as_tensor(ids).long().flatten()
        if len(ids) != len(value) or not torch.equal(ids, torch.arange(count, count + len(ids))):
            raise ValueError("reconstruction task IDs are duplicated, missing, or out of order")
        count += len(ids)
    if count > total_tasks:
        raise ValueError("reconstruction resume contains too many tasks")


def save_progress(path, metadata, next_batch, scores, indices):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"metadata": metadata, "next_batch": next_batch,
                "scores": scores, "indices": indices}, temporary)
    os.replace(temporary, path)


def select_device(gpu):
    from evals.intuitive_physics.train_optical import _device_for_gpu
    return _device_for_gpu(gpu, world_size=1)


def prepare_models(config, device):
    from evals.intuitive_physics.train_optical import _prepare_end_to_end_models
    encoder, target, predictor, _, _ = _prepare_end_to_end_models(
        config, device, experiment_mode="onn_causal_recurrent"
    )
    return encoder, target, predictor


def make_test_loader(config, batch_size, num_workers, data_root):
    from evals.intphys_test.data_manager import init_data
    from src.utils.transforms import make_transforms
    resolution = int(config["data"].get("resolution", 224))
    transform = make_transforms(
        random_horizontal_flip=False, random_resize_aspect_ratio=[1., 1.],
        random_resize_scale=[1., 1.], reprob=0., auto_augment=False,
        motion_shift=False, crop_size=resolution,
    )
    frame_step = int(config["data"]["frame_steps"])
    return init_data(
        batch_size=batch_size, transform=transform, data="IntPhys-test", collator=None,
        pin_mem=True, num_workers=num_workers, world_size=1, rank=0,
        root_path=str(Path(data_root).resolve()) + "/",
        clip_len=99 // frame_step, frame_sample_rate=frame_step,
        deterministic=True, log_dir=None,
    )[0]


@torch.no_grad()
def run_test(checkpoint_path, *, gpu=0, batch_size=15, window_batch_size=15,
             output_dir=None, resume=False, num_workers=8, max_batches=None,
             save_every_batches=20, temporal_reduction="mean", temporal_top_n=2,
             data_root=None):
    if min(batch_size, window_batch_size, num_workers, save_every_batches, temporal_top_n) <= 0:
        raise ValueError("batch sizes, num_workers, save interval and top_n must be positive")
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive")
    if temporal_reduction not in {"mean", "max", "top_n", "top_k"}:
        raise ValueError("unsupported temporal reduction")
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = validate_test_checkpoint(checkpoint)  # reject prediction models before GPU/output work
    data = config["data"]
    if isinstance(data["frame_steps"], (list, tuple)):
        data["frame_steps"] = int(data["frame_steps"][0])
    stride = int(data.get("stride_sliding_window", 2))
    if stride <= 0:
        raise ValueError("stride_sliding_window must be positive")
    if data_root is None:
        from evals.intphys_test.utils import get_dataset_paths
        data_root = get_dataset_paths(["IntPhys-test"])[0]
    data_root = Path(data_root).resolve()
    if not data_root.is_dir():
        raise ValueError(f"IntPhys Test data directory does not exist: {data_root}")
    loader = make_test_loader(config, batch_size, num_workers, data_root)
    tasks = list(loader.dataset.tasks)
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("Test task list must be non-empty and unique")
    stat = checkpoint_path.stat() if checkpoint_path.is_file() else None
    metadata = {
        **causal_runtime_metadata(config), "evaluation_kind": KIND, "cache_version": 1,
        "checkpoint": str(checkpoint_path), "checkpoint_size": stat.st_size if stat else None,
        "checkpoint_mtime_ns": stat.st_mtime_ns if stat else None,
        "checkpoint_epoch": int(checkpoint.get("epoch", 0)), "config": config,
        "data_root": str(data_root), "task_count": len(tasks),
        "task_list_sha256": hashlib.sha256("\n".join(tasks).encode()).hexdigest(),
        "batch_size": batch_size, "window_batch_size": window_batch_size,
        "stride_sliding_window": stride, "spatial_reduction": "mean",
        "temporal_reduction": temporal_reduction, "temporal_top_n": temporal_top_n,
        "score_mapping": "reciprocal",
    }
    if output_dir is None:
        output_dir = checkpoint_path.parent / (
            "test_reconstruction_bs" + str(batch_size) + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
    folder = prepare_output(output_dir, metadata, resume=resume)
    logger = logging.getLogger("intphys_reconstruction_test")
    logger.setLevel(logging.INFO)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    file_handler = logging.FileHandler(folder / "inference.log", mode="a")
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler())
    logger.propagate = False
    started = time.perf_counter()
    logger.info("run_start evaluation_kind=%s checkpoint=%s videos=%d batches=%d batch_size=%d window_batch_size=%d",
                KIND, checkpoint_path, len(tasks), len(loader), batch_size, window_batch_size)
    scores, indices, start_batch = [], [], 0
    resume_path = folder / "reconstruction_resume.pt"
    if resume and resume_path.is_file():
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        validate_progress(state, metadata, len(loader), len(tasks))
        scores, indices, start_batch = state["scores"], state["indices"], state["next_batch"]
        logger.info("resume_loaded next_batch=%d processed=%d", start_batch, sum(len(x) for x in indices))
    elif resume and (folder / "reconstruction_losses.pth").exists():
        raise ValueError("result exists but reconstruction resume state is missing")
    device = select_device(gpu)
    logger.info("device=%s CUDA_VISIBLE_DEVICES=%s", device, os.environ.get("CUDA_VISIBLE_DEVICES", "<all>"))
    encoder, target, predictor = prepare_models(config, device)
    predictor.load_state_dict(checkpoint["predictor"], strict=True)
    for module in (encoder, target, predictor):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    next_batch = start_batch
    for index, batch in enumerate(loader):
        if index < start_batch:
            continue
        if max_batches is not None and index >= max_batches:
            break
        batch_started = time.perf_counter()
        clip, ids = batch
        ids = torch.as_tensor(ids).long().flatten().cpu()
        values = score_video_batch(
            clip, encoder, target, predictor, window_batch_size=window_batch_size,
            stride=stride, use_bfloat16=bool(data.get("use_bfloat16", False)), device=device,
        )
        scores.append(values)
        indices.append(ids)
        next_batch = index + 1
        processed = sum(len(x) for x in indices)
        logger.info("batch=%d/%d processed=%d/%d windows=%d mean_l1=%.6f time_s=%.3f",
                    next_batch, len(loader), processed, len(tasks), values.shape[1],
                    values.mean().item(), time.perf_counter() - batch_started)
        if next_batch % save_every_batches == 0:
            save_progress(resume_path, metadata, next_batch, scores, indices)
    state = {"metadata": metadata, "next_batch": next_batch, "scores": scores, "indices": indices}
    validate_progress(state, metadata, len(loader), len(tasks))
    if not scores:
        raise RuntimeError("no reconstruction Test batches were evaluated")
    save_progress(resume_path, metadata, next_batch, scores, indices)
    all_scores = torch.cat(scores)
    all_indices = torch.cat(indices)
    complete = next_batch == len(loader) and len(all_indices) == len(tasks)
    result = {
        "evaluation_kind": KIND, "objective": "reconstruct_current",
        "checkpoint_epoch": metadata["checkpoint_epoch"], "complete": complete,
        "processed_videos": len(all_indices), "total_videos": len(tasks), "batches": next_batch,
        "windows_per_video": all_scores.shape[1], "scored_chunks_per_window": 7,
        "reconstruction_l1": all_scores.mean().item(),
        "per_step_l1": all_scores.mean(dim=(0, 1)).tolist(),
        "elapsed_s": time.perf_counter() - started,
        "output_dir": str(folder), "score_mapping": "reciprocal",
        "spatial_reduction": "mean", "temporal_reduction": temporal_reduction,
        "official_metrics_available": False,
    }
    torch.save({"metadata": metadata, "tasks": tasks, "task_indices": all_indices,
                "step_scores": all_scores, "complete": complete}, folder / "reconstruction_losses.pth")
    if complete:
        answer_folder = folder / "intphys-test-reconstruction"
        answer_folder.mkdir(exist_ok=True)
        metrics = answer_metrics(all_scores, temporal_reduction, temporal_top_n)
        for name, values in metrics.items():
            (answer_folder / (name + "_answer.txt")).write_text(
                "".join(answer_rows(tasks, all_indices, values)), encoding="utf-8"
            )
        result["answer_dir"] = str(answer_folder)
    else:
        logger.info("partial_run: official answer files are NOT generated")
    (folder / "reconstruction_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("run_done complete=%s processed=%d/%d output_dir=%s", complete, len(all_indices), len(tasks), folder)
    print("RECONSTRUCTION_TEST_DONE" if complete else "RECONSTRUCTION_TEST_PARTIAL", folder)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu", type=int, default=0, help="logical GPU after CUDA_VISIBLE_DEVICES")
    parser.add_argument("--batch-size", type=int, default=15, help="videos per data batch")
    parser.add_argument("--window-batch-size", type=int, default=15, help="windows per GPU microbatch")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-batches", type=int, help="debug only; incomplete runs do not export answers")
    parser.add_argument("--save-every-batches", type=int, default=20)
    parser.add_argument("--temporal-reduction", choices=["mean", "max", "top_n", "top_k"], default="mean")
    parser.add_argument("--temporal-top-n", type=int, default=2)
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()
    run_test(args.checkpoint, **{key: value for key, value in vars(args).items() if key != "checkpoint"})


if __name__ == "__main__":
    main()
