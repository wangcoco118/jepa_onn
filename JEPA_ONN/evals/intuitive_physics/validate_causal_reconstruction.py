"""Validate current-chunk reconstruction on the existing held-out training split.

This is not the official IntPhys prediction Test and never writes answer/ZIP files.
"""
import argparse
import copy
import json
import logging
from datetime import datetime
from pathlib import Path

import torch

from evals.causal_recurrent import (validate_causal_config, validate_causal_checkpoint,
                                   causal_runtime_metadata)
from evals.intuitive_physics import train_optical as train


def run_validation(checkpoint_path, gpu=0, batch_size=15, max_batches=None, output=None):
    if batch_size <= 0 or (max_batches is not None and max_batches <= 0):
        raise ValueError("batch_size and max_batches must be positive")
    checkpoint_path = Path(checkpoint_path).resolve()
    cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if cp.get("mode") != "onn_causal_recurrent" or not isinstance(cp.get("config"), dict):
        raise ValueError("validation requires a full causal recurrent checkpoint")
    cfg = copy.deepcopy(cp["config"])
    _, objective = validate_causal_config(cfg)
    validate_causal_checkpoint(cp, cfg)
    if objective != "reconstruct_current":
        raise ValueError("checkpoint was not trained for reconstruct_current")
    cfg.setdefault("data", {})["batch_size"] = batch_size
    device = train._device_for_gpu(gpu, world_size=1)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("causal-reconstruction-validation")
    logger.info("reconstruction validation only; chunk 8 excluded; %s",
                json.dumps(causal_runtime_metadata(cfg)))
    split_cfg = cfg.get("data_split") or {}
    manifest = cp.get("split_manifest") or cfg["training"]["split_manifest"]
    split = train.require_existing_video_split(
        train.get_dataset_paths(["IntPhys-train"])[0], manifest,
        num_train_videos=int(split_cfg.get("num_train_videos", 2000)),
        num_val_videos=int(split_cfg.get("num_val_videos", 200)),
        split_seed=int(split_cfg.get("split_seed", 42)))
    encoder, target_encoder, predictor, _, _ = train._prepare_end_to_end_models(
        cfg, device, experiment_mode="onn_causal_recurrent")
    predictor.load_state_dict(cp["predictor"], strict=True)
    loader = train._make_jepa_loader(cfg, split["val_video_ids"],
                                    deterministic=True, collator=None)
    metrics = train._run_causal_recurrent_epoch(
        loader, cfg, encoder, target_encoder, predictor, None, device,
        logger, int(cp.get("epoch", 0)), training=False, max_steps=max_batches)
    metrics.update(causal_runtime_metadata(cfg))
    metrics.update(checkpoint=str(checkpoint_path), evaluation_kind="reconstruction_validation",
                   checkpoint_epoch=int(cp.get("epoch", 0)),
                   max_batches=max_batches, evaluated_split="val")
    if output is None:
        output = checkpoint_path.parent / (
            "reconstruction_validation_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json")
    output = Path(output).resolve()
    allowed = [Path("/data/linux/wkx").resolve(), Path("/data/wkx").resolve()]
    if not any(output.is_relative_to(root) for root in allowed):
        raise ValueError("validation output must be under /data/linux/wkx or /data/wkx")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print("RECONSTRUCTION_VALIDATION_DONE", output)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--gpu", type=int, default=0, help="logical visible GPU")
    parser.add_argument("--batch-size", type=int, default=15)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    run_validation(args.checkpoint, args.gpu, args.batch_size, args.max_batches, args.output)


if __name__ == "__main__":
    main()
