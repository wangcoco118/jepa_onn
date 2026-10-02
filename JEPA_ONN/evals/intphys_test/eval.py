# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import copy

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import logging
import pprint
import time

import numpy as np
from einops import rearrange
import pandas as pd

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed import all_gather
from sklearn.metrics import precision_recall_curve,roc_curve,auc
from scipy.stats import mannwhitneyu,wilcoxon,ttest_rel,ttest_ind
import torch.distributed as dist


from src.utils.tensors import repeat_interleave_batch
from src.utils.amp import autocast_context
from src.masks.utils import apply_masks
import src.models.vision_transformer as vit
import src.models.predictor as vit_pred
from src.models.utils.multimask import MultiMaskWrapper, PredictorMultiMaskWrapper
from src.models.fsonn import ONNConfig, OpticalQKVConfig
from src.models.causal_recurrent_onn import CausalRecurrentONNPredictor
from evals.causal_recurrent import (
    evaluate_causal_recurrent_windows,
    reduce_scores,
)
from evals.intphys_test.data_manager import init_data
from src.masks.random_tube import MaskCollator as TubeMaskCollator
from src.masks.multiblock3d import MaskCollator as MB3DMaskCollator
from src.masks.causal import MaskCollator as CausalMaskCollator
from src.utils.distributed import (
    init_distributed,
    AllReduce
)
from src.utils.logging import (
    AverageMeter,
    CSVLogger
)

from src.utils.transforms import make_transforms

from evals.intphys_test.utils import get_time_masks,get_dataset_paths,batch_all_gather,PROPERTIES_BY_DATASET,pad_tensors
import evals.intphys_test.videomae as videomae

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

pp = pprint.PrettyPrinter(indent=4)


def _last_context_copy_enabled(args_eval):
    value = (args_eval.get("evaluation") or {}).get(
        "last_context_copy_baseline", False
    )
    if not isinstance(value, bool):
        raise TypeError("evaluation.last_context_copy_baseline must be a boolean")
    return value


def _last_context_copy_prediction(
    context,
    num_target_tokens,
    spatial_tokens,
    normalize=True,
):
    if context.ndim != 3:
        raise ValueError("context must have shape [B,N,D]")
    if spatial_tokens <= 0:
        raise ValueError("spatial_tokens must be positive")
    if context.shape[1] < spatial_tokens or context.shape[1] % spatial_tokens != 0:
        raise ValueError("context must contain complete spatial token slices")
    if num_target_tokens <= 0 or num_target_tokens % spatial_tokens != 0:
        raise ValueError("target must contain complete spatial token slices")

    last_context = context[:, -spatial_tokens:, :]
    if normalize:
        last_context = F.layer_norm(last_context, (last_context.size(-1),))
    future_steps = num_target_tokens // spatial_tokens
    return last_context.repeat(1, future_steps, 1)


def _configure_inference_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "inference.log")
    for handler in list(logger.handlers):
        if getattr(handler, "_intphys_test_inference_handler", False):
            logger.removeHandler(handler)
            handler.close()
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    file_handler._intphys_test_inference_handler = True
    logger.addHandler(file_handler)
    return log_path


_RESUME_CHECKPOINT_VERSION = 1


def _normalize_resume_context_lengths(context_lengths):
    if isinstance(context_lengths, (list, tuple)):
        return tuple(int(value) for value in context_lengths)
    return (int(context_lengths),)


def _resume_metadata(frame_step, context_lengths, batch_size, predictor_checkpoint):
    return {
        "resume_version": _RESUME_CHECKPOINT_VERSION,
        "frame_step": int(frame_step),
        "context_lengths": _normalize_resume_context_lengths(context_lengths),
        "batch_size": int(batch_size),
        "predictor_checkpoint": (
            os.path.abspath(os.fspath(predictor_checkpoint))
            if predictor_checkpoint is not None
            else None
        ),
    }


def _save_resume_checkpoint(
    path,
    *,
    next_batch,
    frame_step,
    context_lengths,
    batch_size,
    predictor_checkpoint,
    all_losses,
    all_tasks,
):
    path = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = _resume_metadata(
        frame_step,
        context_lengths,
        batch_size,
        predictor_checkpoint,
    )
    state.update(
        {
            "next_batch": int(next_batch),
            "all_losses": [
                value.detach().cpu() if torch.is_tensor(value) else value
                for value in all_losses
            ],
            "all_tasks": [
                value.detach().cpu() if torch.is_tensor(value) else value
                for value in all_tasks
            ],
        }
    )
    temporary_path = f"{path}.tmp"
    torch.save(state, temporary_path)
    os.replace(temporary_path, path)


def _load_resume_checkpoint(
    path,
    *,
    frame_step,
    context_lengths,
    batch_size,
    predictor_checkpoint,
):
    if not path or not os.path.exists(path):
        return None
    path = os.path.abspath(os.fspath(path))
    state = torch.load(path, map_location="cpu", weights_only=False)
    expected = _resume_metadata(
        frame_step,
        context_lengths,
        batch_size,
        predictor_checkpoint,
    )
    for key, expected_value in expected.items():
        actual_value = state.get(key)
        if key == "context_lengths" and actual_value is not None:
            actual_value = _normalize_resume_context_lengths(actual_value)
        if actual_value != expected_value:
            raise ValueError(
                f"resume checkpoint metadata mismatch for {key}: "
                f"expected {expected_value!r}, got {actual_value!r}"
            )
    next_batch = int(state.get("next_batch", -1))
    if next_batch < 0:
        raise ValueError("resume checkpoint next_batch must be non-negative")
    return state


def _format_batch_progress(
    *,
    batch_index,
    total_batches,
    movie_paths,
    data_read_time_s,
    feature_time_s,
    onn_time_s,
    surprise,
    plausibility,
    processed,
    failures,
):
    return (
        f"batch={batch_index}/{total_batches} "
        f"movie_path_range={movie_paths[0]}..{movie_paths[-1]} "
        f"data_read_time_s={data_read_time_s:.3f} "
        f"feature_time_s={feature_time_s:.3f} "
        f"onn_time_s={onn_time_s:.3f} "
        f"surprise={surprise:.6f} plausibility={plausibility:.6f} "
        f"processed={processed} failures={failures}"
    )


def _synchronize_device(device):
    if getattr(device, "type", None) == "cuda":
        torch.cuda.synchronize(device)


def main(args_eval, resume_preempt=False):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- PRETRAIN
    args_pretrain = args_eval.get('pretrain')
    enc_checkpoint_key = args_pretrain.get('enc_checkpoint_key', 'encoder')
    pred_checkpoint_key = args_pretrain.get('pred_checkpoint_key', 'predictor')
    model_name = args_pretrain.get('model_name', None)
    patch_size = args_pretrain.get('patch_size', None)
    pretrain_folder = args_pretrain.get('folder', None)
    ckp_fname = args_pretrain.get('checkpoint', None)
    tag = args_pretrain.get('write_tag', None)
    use_sdpa = args_pretrain.get('use_sdpa', True)
    use_SiLU = args_pretrain.get('use_silu', False)
    wide_SiLU = args_pretrain.get('wide_silu', True)
    uniform_power = args_pretrain.get('uniform_power', False)
    is_causal = args_pretrain.get('is_causal', False)
    pred_is_causal = args_pretrain.get('pred_is_causal', False)
    pred_depth = args_pretrain.get('pred_depth', 12)
    optical_qkv = args_eval.get('optical_qkv', {})
    predictor_checkpoint = args_eval.get("predictor_checkpoint")
    pretrained_path = os.path.join(pretrain_folder, ckp_fname)
    # [for Video model]:
    tubelet_size = args_pretrain.get('tubelet_size', 2)
    pretrain_frames_per_clip = args_pretrain.get('frames_per_clip', 1)

    # -- MASK

    # -- DATA
    args_data = args_eval.get('data')
    resolution = args_data.get('resolution', 224)
    batch_size = args_data.get('batch_size', 1)
    stride_sliding_window = args_data.get('stride_sliding_window',2)
    use_bfloat16 = args_data.get('use_bfloat16')
    eval_frames_per_clip = args_data.get('frames_per_clip', 16)

    all_context_lengths = args_data.get('context_lengths', 4)
    eval_frame_steps = args_data.get('frame_steps', 4)

    normalize_enc =  args_data.get('normalize_context', False)


    # -- EXPERIMENT
    eval_tag = args_eval.get('tag', None)
    mode = args_eval.get('mode', 'all')
    assert mode in ['all','losses','metrics']
    dataset = args_eval.get('dataset', 'intphys')
    assert dataset in ['intphys-test']
    is_mae = args_eval.get('is_mae', False)
    mae_decoder_blocks = args_eval.get('mae_decoder_blocks', -1)
    normalize_targets =args_eval.get('normalize_targets',True)
    last_context_copy_baseline = _last_context_copy_enabled(args_eval)
    predictor_type = args_eval.get("predictor_type", "onn_feedback")
    evaluation_config = args_eval.get("evaluation") or {}
    if predictor_type == "onn_causal_recurrent":
        all_context_lengths = ["causal_recurrent"]
    resume_config = args_eval.get("test_resume") or {}
    if not isinstance(resume_config, dict):
        raise TypeError("test_resume must be a mapping")
    resume_enabled = resume_config.get("enabled", False)
    if not isinstance(resume_enabled, bool):
        raise TypeError("test_resume.enabled must be a boolean")
    resume_checkpoint_path = resume_config.get("checkpoint_path")
    resume_every_batches = int(resume_config.get("save_every_batches", 20))
    if resume_every_batches <= 0:
        raise ValueError("test_resume.save_every_batches must be positive")
    if resume_enabled and not resume_checkpoint_path:
        raise ValueError(
            "test_resume.checkpoint_path is required when resume is enabled"
        )
    if predictor_type == "onn_causal_recurrent" and resume_enabled:
        raise ValueError(
            "test_resume is not supported for causal diagnostic caches"
        )
    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')

    # -- log/checkpointing paths
    output_root = args_eval.get("output_dir") or os.path.join(
        pretrain_folder, "intphys_test/"
    )
    folder = output_root
    if eval_tag is not None:
        folder = os.path.join(folder, f"{dataset}-{eval_tag}")
    os.makedirs(folder, exist_ok=True)
    if rank == 0:
        log_path = _configure_inference_logging(output_root)
        logger.info(
            "run_start dataset=%s checkpoint=%s batch_size=%d "
            "prediction_mode=%s output_dir=%s log=%s",
            dataset,
            predictor_checkpoint,
            batch_size,
            "last_context_copy" if last_context_copy_baseline else "predictor",
            folder,
            log_path,
        )
    # Initialize model

    # -- pretrained encoder (frozen)
    encoder,target_encoder, predictor = init_model(
        crop_size=resolution,
        device=device,
        pretrained=pretrained_path,
        model_name=model_name,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        frames_per_clip=eval_frames_per_clip,
        is_causal=is_causal,
        pred_is_causal=pred_is_causal,
        pred_depth=pred_depth,
        uniform_power=uniform_power,
        enc_checkpoint_key=enc_checkpoint_key,
        pred_checkpoint_key=pred_checkpoint_key,
        use_SiLU=use_SiLU,
        wide_SiLU=wide_SiLU,
        use_sdpa=use_sdpa,
        is_mae=is_mae,
        optical_qkv=optical_qkv,
        predictor_checkpoint=predictor_checkpoint,
        predictor_type=predictor_type,
        output_mode=args_eval.get("predictor", {}).get("output_mode", "mlp"),
        temporal_difference_enabled=args_eval.get("predictor", {}).get(
            "temporal_difference_enabled", False
        ),
        temporal_difference_alpha=args_eval.get("predictor", {}).get(
            "temporal_difference_alpha", 0.5
        ),
        optical_output_config=args_eval.get("predictor", {}).get(
            "optical_output"
        ),
        direct_384_loss=args_eval.get("predictor", {}).get(
            "direct_384_loss", False
        ),
        onn_feedback_config=args_eval.get(
            "onn", args_eval.get("onn_feedback", optical_qkv)
        ),
        causal_recurrent_config=args_eval.get("predictor", {}),
    )
    
    if not is_mae:
        target_encoder.eval()
        predictor.eval()
        for p in target_encoder.parameters():
            p.requires_grad = False
        
        for p in predictor.parameters():
            p.requires_grad = False

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # Initialize data loaders
    #TODO: Might be issues here, we want a resize more than a center crop
    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=[1/1, 1/1],
        random_resize_scale=[1.0, 1.0],
        reprob=0.,
        auto_augment=False,
        motion_shift=False,
        crop_size=resolution)

    if not isinstance(eval_frame_steps, list):
        eval_frame_steps = [eval_frame_steps]

    init_logger = True
    for frame_step in eval_frame_steps:
        if mode in ['losses','all']:
            logger.info(f"Extracting loss ...")
            extracted = extract_losses(
                device=device,
                encoder=encoder,
                target_encoder=target_encoder,
                predictor=predictor,
                transform=transform,
                use_bfloat16=use_bfloat16,
                frame_step=frame_step,
                context_lengths=all_context_lengths,
                batch_size=batch_size,
                frames_per_clip=eval_frames_per_clip,
                stride=stride_sliding_window,
                world_size=world_size,
                rank=rank,
                normalize_enc=normalize_enc,
                dataset=dataset,
                is_mae=is_mae,
                mae_decoder_blocks=mae_decoder_blocks,
                patch_size=patch_size,
                resolution=resolution,
                normalize_targets=normalize_targets,
                last_context_copy_baseline=last_context_copy_baseline,
                predictor_type=predictor_type,
                motion_epsilon=args_eval.get("loss", {}).get(
                    "motion_epsilon", 1.0e-6
                ),
                spatial_reduction=evaluation_config.get(
                    "spatial_reduction", "mean"
                ),
                spatial_top_n=evaluation_config.get("spatial_top_n", 20),
                return_causal_diagnostics=(
                    predictor_type == "onn_causal_recurrent"
                ),
                log_progress=(rank == 0),
                resume_checkpoint_path=(
                    resume_checkpoint_path if resume_enabled else None
                ),
                resume_every_batches=resume_every_batches,
                resume_predictor_checkpoint=predictor_checkpoint)
            if predictor_type == "onn_causal_recurrent":
                (
                    all_losses,
                    all_labels,
                    tasks,
                    causal_diagnostics,
                ) = extracted
            else:
                all_losses, all_labels, tasks = extracted
                causal_diagnostics = {}

            all_losses = batch_all_gather(all_losses).cpu()
            causal_diagnostics = {
                name: batch_all_gather(values).cpu()
                for name, values in causal_diagnostics.items()
            }
            all_labels = batch_all_gather(all_labels).cpu().numpy().astype(int)
            all_tasks = list(np.array(tasks)[all_labels])
            
            if rank == 0 :
                logger.info('saving')
                logger.info(os.path.join(folder, f'losses_{frame_step}fs_{"_".join([str(ctxt) for ctxt in all_context_lengths])}ctxt.pth'))
                torch.save({"frame_step":frame_step,
                            "context_lengths":all_context_lengths,
                            "losses":all_losses,
                            "tasks":all_tasks,
                            "causal_diagnostics":causal_diagnostics,
                            },
                            os.path.join(folder, f'losses_{frame_step}fs_{"_".join([str(ctxt) for ctxt in all_context_lengths])}ctxt.pth'))
        
        if mode in ['metrics','all']:
            logger.info(f"Computing metrics ...")
            if mode == "metrics":
                data = torch.load(os.path.join(folder, f'losses_{frame_step}fs_{"_".join([str(ctxt) for ctxt in all_context_lengths])}ctxt.pth'))
                all_losses = data["losses"]
                all_tasks = data["tasks"]
                causal_diagnostics = data.get(
                    "causal_diagnostics", {}
                )

            filtered = all_losses.min(1)[0]
            metrics = compute_metrics(
                filtered,
                temporal_reduction=(
                    evaluation_config.get("temporal_reduction", "mean")
                    if predictor_type == "onn_causal_recurrent"
                    else None
                ),
                temporal_top_n=evaluation_config.get(
                    "temporal_top_n", 2
                ),
                score_mapping=(
                    "reciprocal"
                    if predictor_type == "onn_causal_recurrent" else "linear"
                ),
            )

            if rank == 0:
                for metric,values in metrics.items():
                    data = {'tasks':all_tasks, 'surprises': values}
                    df = pd.DataFrame(data)
                    # Save to CSV without headers
                    log_file = os.path.join(folder, f'{metric}_answer.txt')
                    df.to_csv(log_file, index=False, header=False,sep=" ")

    if rank == 0:
        logger.info("run_done output_dir=%s", folder)


@torch.no_grad()
def extract_losses(
    device,
    encoder,
    target_encoder,
    predictor,
    transform,
    use_bfloat16=False,
    frame_step=1,
    context_lengths=[2],
    batch_size=1,
    frames_per_clip=16,
    stride=2,
    world_size=1,
    rank=0,
    normalize_enc=False,
    dataset="intphys",
    is_mae=False,
    mae_decoder_blocks=-1,
    patch_size=16,
    resolution=224,
    normalize_targets=True,
    last_context_copy_baseline=False,
    predictor_type="vit_transformer",
    motion_epsilon=1.0e-6,
    spatial_reduction="mean",
    spatial_top_n=20,
    return_causal_diagnostics=False,
    log_progress=True,
    resume_checkpoint_path=None,
    resume_every_batches=20,
    resume_predictor_checkpoint=None,
):
    print(context_lengths)

    sampling_rate,num_frames = frame_step ,99//frame_step
    
    print(f"Sampling rate 1/{sampling_rate} frames")

    if dataset == "intphys-test":
        data_name = f"IntPhys-test" 

    (data,unsupervised_sampler) = init_data(
        batch_size = batch_size,
        transform=transform,
        data=data_name,
        collator=None,
        pin_mem=True,
        num_workers=8,
        world_size=world_size,
        rank=rank,
        root_path=get_dataset_paths([data_name])[0],
        clip_len=num_frames,
        frame_sample_rate=sampling_rate,
        deterministic=True,
        log_dir=None)


    loader = iter(data)
    total_batches = len(loader)

    all_tasks = []
    all_losses = []
    causal_diagnostics = {
        "motion_weighted_score": [],
        "motion_only_score": [],
        "copy_baseline_score": [],
    }
    start_batch = 0
    if resume_checkpoint_path:
        resume_every_batches = int(resume_every_batches)
        if resume_every_batches <= 0:
            raise ValueError("resume_every_batches must be positive")
        resume_state = _load_resume_checkpoint(
            resume_checkpoint_path,
            frame_step=frame_step,
            context_lengths=context_lengths,
            batch_size=batch_size,
            predictor_checkpoint=resume_predictor_checkpoint,
        )
        if resume_state is not None:
            start_batch = int(resume_state["next_batch"])
            if start_batch > total_batches:
                raise ValueError(
                    "resume checkpoint next_batch exceeds available batches"
                )
            all_losses = [
                value.to(device) if torch.is_tensor(value) else value
                for value in resume_state.get("all_losses", [])
            ]
            all_tasks = list(resume_state.get("all_tasks", []))
            if log_progress:
                logger.info(
                    "resume_loaded frame_step=%d next_batch=%d/%d checkpoint=%s",
                    frame_step,
                    start_batch,
                    total_batches,
                    resume_checkpoint_path,
                )
    processed = 0
    failures = 0
    if log_progress:
        logger.info(
            "data_ready tasks=%d batches=%d batch_size=%d",
            len(data.dataset),
            total_batches,
            batch_size,
        )

    for i in range(total_batches):
        data_read_started = time.perf_counter()
        udata = next(loader)
        if i < start_batch:
            continue

        tasks = udata[1]

        clip = udata[0]
        clip = clip.to(device)
        _synchronize_device(device)
        data_read_time_s = time.perf_counter() - data_read_started

        #Batch size
        num_videos = clip.shape[0]

        pieces = clip.unfold(2, frames_per_clip,stride).permute(0,2,-1,1,3,4).contiguous()

        pieces = pieces.flatten(0,1)#.view(-1,3,16,224,224)
        pieces = rearrange(pieces,"b t c h w ->  b c t h w")

        pieces = pieces.contiguous()

        B, C, T, H, W = pieces.shape

        
        all_losses_ctxt = []
        current_causal_diagnostics = None
        feature_time_s = 0.0
        onn_time_s = 0.0
        if predictor_type == "onn_causal_recurrent":
            if is_mae or last_context_copy_baseline:
                raise ValueError(
                    "causal recurrent evaluation is incompatible with "
                    "MAE and last-context-copy modes"
                )
            recurrent_started = time.perf_counter()
            with autocast_context(device, use_bfloat16):
                recurrent_scores = evaluate_causal_recurrent_windows(
                    pieces,
                    encoder,
                    target_encoder,
                    predictor,
                    chunk_frames=2,
                    chunk_tokens=(resolution // patch_size) ** 2,
                    motion_epsilon=motion_epsilon,
                    spatial_reduction=spatial_reduction,
                    spatial_top_n=spatial_top_n,
                )
            _synchronize_device(device)
            onn_time_s += time.perf_counter() - recurrent_started
            all_losses_ctxt.append(
                recurrent_scores["primary_step_scores"].reshape(
                    num_videos, -1
                )
            )
            current_causal_diagnostics = {
                "motion_weighted_score": recurrent_scores[
                    "motion_weighted_step_scores"
                ].reshape(num_videos, -1),
                "motion_only_score": recurrent_scores[
                    "motion_only_step_scores"
                ].reshape(num_videos, -1),
                "copy_baseline_score": recurrent_scores[
                    "copy_step_scores"
                ].reshape(num_videos, -1),
            }
        loop_context_lengths = (
            [] if predictor_type == "onn_causal_recurrent"
            else context_lengths
        )
        for CTXT_LEN in loop_context_lengths:

            m,m_,full_m = get_time_masks(CTXT_LEN,spatial_size=(patch_size,patch_size),temporal_dim=frames_per_clip,as_bool=is_mae)
            full_m = full_m.unsqueeze(0).to(device)
            m = m.unsqueeze(0).to(device)
            m_ = m_.unsqueeze(0).to(device)
            
            if is_mae:
                masks_enc = m.repeat(B, 1)
                masks_pred = m_.repeat(B, 1)
                full_mask = full_m.repeat(B, 1)
            else:
                masks_enc = [m.repeat(B, 1)]
                masks_pred = [m_.repeat(B, 1)]
                full_mask = [full_m.repeat(B, 1)]

            with autocast_context(device, use_bfloat16):
                if is_mae: 
                    if mae_decoder_blocks == -1:
                        mean = torch.as_tensor((0.485, 0.456, 0.406)).to(device)[None, :, None, None, None]
                        std = torch.as_tensor((0.229, 0.224, 0.225)).to(device)[None, :, None, None, None]
                        unnorm_videos = pieces * std + mean  # in [0, 1]

                        videos_squeeze = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=2, p1=patch_size, p2=patch_size)
                        var = videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6
                        mean = videos_squeeze.mean(dim=-2, keepdim=True)
                        videos_norm = (videos_squeeze - mean) / (var)
                        videos_patch = rearrange(videos_norm, 'b n p c -> b n (p c)')
                        B, _, C = videos_patch.shape
                        targets = videos_patch[masks_pred].reshape(B, -1, C)
                    else:
                        targets = encoder(pieces,~full_m.repeat(B,1),decoder_blocks=mae_decoder_blocks)
                        B, _, C = targets.shape
                        targets = targets[masks_pred].reshape(B, -1, C)

                    preds = encoder(pieces,masks_pred,decoder_blocks=mae_decoder_blocks)
                  
                    preds = preds.view(num_videos,-1,*preds.shape[1:])
                    preds = torch.zeros_like(preds,device=preds.device)
                    targets = targets.view(num_videos,-1,*targets.shape[1:])

                else:
                    _synchronize_device(device)
                    feature_started = time.perf_counter()
                    h = target_encoder(pieces,full_mask)[0]
                    if normalize_targets:
                        h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim  [B, N, D]
                    # -- create targets (masked regions of h)
                    targets = apply_masks(h, masks_pred, concat=False)


                    context = encoder(pieces, masks_enc)
                    if normalize_enc:
                        z_ = []
                        for zi in context:
                            z_ += [F.layer_norm(zi,(zi.size(-1),))]
                        context = z_
                    _synchronize_device(device)
                    feature_time_s += time.perf_counter() - feature_started

                    onn_started = time.perf_counter()
                    if last_context_copy_baseline:
                        if resolution % patch_size != 0:
                            raise ValueError(
                                "resolution must be divisible by patch_size"
                            )
                        spatial_tokens = (resolution // patch_size) ** 2
                        preds = [
                            _last_context_copy_prediction(
                                context[0],
                                num_target_tokens=masks_pred[0].shape[1],
                                spatial_tokens=spatial_tokens,
                                normalize=normalize_targets,
                            )
                        ]
                    else:
                        targets = vit_pred.project_targets_for_loss(
                            predictor, targets
                        )
                        preds = predictor(
                            context, targets, masks_enc, masks_pred
                        )
                    _synchronize_device(device)
                    onn_time_s += time.perf_counter() - onn_started

                    preds = preds[0].view(num_videos,-1,*preds[0].shape[1:])
                    targets = targets[0].view(num_videos,-1,*targets[0].shape[1:])
            all_losses_ctxt.append(F.l1_loss(preds,targets,reduction="none").mean((2,3)).detach())
        losses = torch.stack(all_losses_ctxt)
        losses = losses.permute(1,0,2)

        # Always append by matches for easy filtering later
        # i.e. all_losses[all_labels == 0] and 1 are matched pairwise
        all_losses.append(losses)
        all_tasks.append(tasks)
        if current_causal_diagnostics is not None:
            for name, values in current_causal_diagnostics.items():
                causal_diagnostics[name].append(values)
        if (
            resume_checkpoint_path
            and (i + 1) % resume_every_batches == 0
        ):
            _save_resume_checkpoint(
                resume_checkpoint_path,
                next_batch=i + 1,
                frame_step=frame_step,
                context_lengths=context_lengths,
                batch_size=batch_size,
                predictor_checkpoint=resume_predictor_checkpoint,
                all_losses=all_losses,
                all_tasks=all_tasks,
            )
        if log_progress:
            task_indices = tasks.detach().cpu().flatten().numpy().astype(int)
            movie_paths = [data.dataset.tasks[index] for index in task_indices]
            surprise = losses.min(1)[0].mean().item()
            if predictor_type == "onn_causal_recurrent":
                plausibility = 1.0 / (1.0 + surprise)
            else:
                plausibility = float(np.clip(1.0 - surprise, 0.0, 1.0))
            processed += num_videos
            logger.info(
                _format_batch_progress(
                    batch_index=i + 1,
                    total_batches=total_batches,
                    movie_paths=movie_paths,
                    data_read_time_s=data_read_time_s,
                    feature_time_s=feature_time_s,
                    onn_time_s=onn_time_s,
                    surprise=surprise,
                    plausibility=plausibility,
                    processed=processed,
                    failures=failures,
                )
            )
    # This padding is only used for InfLevel but ensures easy processing
    # The padding can be removed by filtering end zeros since the loss is never zero
    # This can lead to slighlty innacurate metrics computed from this script
    lengths = []
    for l in all_losses:
        lengths.append(l.size(-1))
    max_length = torch.tensor([max(lengths)]).to(device)
    #We need to sync the max lengths otherwise we can't gather the losses afterwards
    if (
        dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    ):
        dist.all_reduce(max_length, op=dist.ReduceOp.MAX)
    
    all_losses = torch.concat(pad_tensors(all_losses,max_length.item()))
    all_tasks = torch.concat(all_tasks).flatten()
    logger.info(all_tasks.shape)

    result = (
        all_losses,
        all_tasks.to(device),
        data.dataset.tasks,
    )
    if return_causal_diagnostics:
        finalized_diagnostics = {
            name: torch.concat(pad_tensors(values, max_length.item()))
            for name, values in causal_diagnostics.items()
        }
        return (*result, finalized_diagnostics)
    return result


def compute_metrics(
    losses,
    temporal_reduction=None,
    temporal_top_n=2,
    score_scale=1.0,
    score_mapping="linear",
):
    # Preserve legacy mapping; unbounded causal errors use reciprocal mapping.
    if score_mapping not in {"linear", "reciprocal"}:
        raise ValueError("unsupported score_mapping")
    score_scale = float(score_scale)
    if not score_scale > 0.0:
        raise ValueError("score_scale must be positive")
    metrics = {}

    def map_error(error):
        if score_mapping == "reciprocal":
            return 1.0 / (1.0 + error)
        return (1.0 - error / score_scale).clamp(0.0, 1.0)

    # Aggregate raw prediction errors first, then map to plausibility.
    average_losses = map_error(losses.mean(1))
    max_losses = map_error(losses.max(1)[0])

    metrics["maximum_surprise"] = max_losses
    metrics["average_surprise"] = average_losses
    if temporal_reduction is not None:
        selected_loss = reduce_scores(
            losses,
            temporal_reduction,
            dim=1,
            top_n=temporal_top_n,
        )
        metrics["selected_surprise"] = map_error(selected_loss)
        if str(temporal_reduction).strip().lower().replace("-", "_") in {
            "top_n", "top_k"
        }:
            k = min(int(temporal_top_n), losses.shape[1])
            metrics[f"top_{k}_surprise"] = metrics["selected_surprise"]

    return metrics





def load_pretrained(
    encoder,
    target_encoder,
    predictor,
    pretrained,
    enc_checkpoint_key='encoder',
    target_enc_checkpoint_key='target_encoder',
    pred_checkpoint_key='predictor',
    is_mae=False,
    load_predictor=True,
    load_predictor_embed=False,
):
    logger.info(f'Loading pretrained model from {pretrained}')
    checkpoint = torch.load(pretrained, map_location='cpu')

    try:
        enc_pretrained_dict = checkpoint[enc_checkpoint_key]
    except Exception:
        enc_pretrained_dict = checkpoint['encoder']
    enc_pretrained_dict = {
        key.replace('module.', ''): value
        for key, value in enc_pretrained_dict.items()
    }
    msg = encoder.load_state_dict(enc_pretrained_dict, strict=False)
    logger.info(f'loaded pretrained model with msg: {msg}')
    print(encoder)

    if not is_mae:
        try:
            target_enc_pretrained_dict = checkpoint[target_enc_checkpoint_key]
        except Exception:
            target_enc_pretrained_dict = checkpoint["target_encoder"]
        target_enc_pretrained_dict = {
            key.replace('module.', ''): value
            for key, value in target_enc_pretrained_dict.items()
        }
        msg = target_encoder.load_state_dict(target_enc_pretrained_dict, strict=False)
        logger.info(f'loaded pretrained model with msg: {msg}')
        print(target_encoder)

        if load_predictor:
            try:
                pred_pretrained_dict = checkpoint[pred_checkpoint_key]
            except Exception:
                pred_pretrained_dict = checkpoint['predictor']
            pred_pretrained_dict = {
                key.replace('module.', ''): value
                for key, value in pred_pretrained_dict.items()
            }
            msg = predictor.load_state_dict(pred_pretrained_dict, strict=False)
            logger.info(f'loaded pretrained model with msg: {msg}')
            logger.info(
                f'loaded pretrained predictor from epoch: {checkpoint["epoch"]}\n'
                f' path: {pretrained}'
            )
            print(predictor)
        elif load_predictor_embed:
            try:
                pred_pretrained_dict = checkpoint[pred_checkpoint_key]
            except Exception:
                pred_pretrained_dict = checkpoint["predictor"]
            predictor_model = (
                predictor.backbone
                if hasattr(predictor, "backbone")
                else predictor
            )
            predictor_model.load_predictor_embed_state_dict(
                pred_pretrained_dict
            )
            logger.info(
                "loaded frozen predictor_embed from pretrained Predictor"
            )
        else:
            logger.info("skipping legacy Predictor checkpoint for ONN feedback mode")

    del checkpoint
    return encoder, target_encoder, predictor


def _load_trained_predictor(predictor, checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if checkpoint.get("mode") not in {
        "end_to_end_jepa", "electronic_control", "onn_feedback",
        "onn_causal_recurrent"
    }:
        raise ValueError(
            "trained Predictor checkpoint must have mode=end_to_end_jepa, "
            "electronic_control, onn_feedback, or onn_causal_recurrent"
        )
    state_dict = checkpoint.get("predictor")
    if state_dict is None:
        raise ValueError("trained Predictor checkpoint has no full Predictor state")
    normalized = {
        key.replace("module.", ""): value for key, value in state_dict.items()
    }
    expected = predictor.state_dict()
    missing = sorted(set(expected).difference(normalized))
    unexpected = sorted(set(normalized).difference(expected))
    mismatched = sorted(
        key for key in set(expected).intersection(normalized)
        if expected[key].shape != normalized[key].shape
    )
    if missing or unexpected or mismatched:
        raise RuntimeError(
            "trained predictor checkpoint mismatch: "
            f"missing={missing}, unexpected={unexpected}, shape_mismatch={mismatched}"
        )
    predictor.load_state_dict(normalized, strict=True)


def init_model(
    device,
    pretrained,
    model_name,
    patch_size=16,
    crop_size=224,
    frames_per_clip=16,
    tubelet_size=2,
    use_sdpa=False,
    use_SiLU=False,
    wide_SiLU=False,
    is_causal=False,
    pred_is_causal=False,
    uniform_power=False,
    enc_checkpoint_key='encoder',
    pred_checkpoint_key='predictor',
    use_mask_tokens=True,
    pred_embed_dim=384,
    pred_depth=12,
    num_mask_tokens=2,
    is_mae=False,
    optical_qkv=None,
    predictor_checkpoint=None,
    predictor_type="vit_transformer",
    output_mode="mlp",
    temporal_difference_enabled=False,
    temporal_difference_alpha=0.5,
    optical_output_config=None,
    direct_384_loss=False,
    onn_feedback_config=None,
    causal_recurrent_config=None,
):
    optical_qkv = optical_qkv or {}
    if is_mae:
        encoder = videomae.__dict__[model_name]()
        target_encoder = None
        predictor = None
    else:
        encoder = vit.__dict__[model_name](
            img_size=crop_size,
            patch_size=patch_size,
            num_frames=frames_per_clip,
            tubelet_size=tubelet_size,
            uniform_power=uniform_power,
            use_sdpa=use_sdpa,
            use_SiLU=use_SiLU,
            wide_SiLU=wide_SiLU,
            is_causal=is_causal,
        )
        target_encoder = copy.deepcopy(encoder)
        encoder = MultiMaskWrapper(encoder)
        target_encoder = MultiMaskWrapper(target_encoder)

        if predictor_type == "onn_feedback":
            onn_config = dict(onn_feedback_config or optical_qkv)
            predictor = vit_pred.ONNFeedbackPredictor(
                img_size=crop_size,
                patch_size=patch_size,
                num_frames=frames_per_clip,
                tubelet_size=tubelet_size,
                embed_dim=encoder.backbone.embed_dim,
                predictor_embed_dim=pred_embed_dim,
                output_mode=output_mode,
                temporal_difference_enabled=temporal_difference_enabled,
                temporal_difference_alpha=temporal_difference_alpha,
                optical_output_config=optical_output_config,
                direct_384_loss=direct_384_loss,
                num_tokens=1568,
                num_chunks=8,
                chunk_tokens=196,
                output_mlp_hidden_dim=int(
                    onn_config.get("output_mlp_hidden_dim", pred_embed_dim)
                ),
                feedback_mode=onn_config.get("feedback_mode", "fixed_middle"),
                feedback_layer_index=onn_config.get("feedback_layer_index"),
                uniform_power=uniform_power,
                optical_config=onn_config,
            )
        elif predictor_type == "onn_causal_recurrent":
            predictor_cfg = dict(causal_recurrent_config or {})
            onn_config = ONNConfig.from_mapping(
                dict(onn_feedback_config or optical_qkv)
            )
            predictor = CausalRecurrentONNPredictor(
                embed_dim=encoder.backbone.embed_dim,
                predictor_embed_dim=pred_embed_dim,
                num_context_chunks=int(
                    predictor_cfg.get("num_context_chunks", 7)
                ),
                chunk_tokens=int(
                    predictor_cfg.get("chunk_tokens", 196)
                ),
                memory_lambda=float(
                    predictor_cfg.get("memory_lambda", 0.5)
                ),
                onn_config=onn_config,
            )
        elif predictor_type == "vit_transformer":
            use_rope = 'rope' in model_name
            rope_is_1D = 'rope1D' in model_name
            predictor = vit_pred.vit_predictor(
                img_size=crop_size,
                use_mask_tokens=use_mask_tokens,
                is_causal=pred_is_causal,
                patch_size=patch_size,
                num_frames=frames_per_clip,
                tubelet_size=tubelet_size,
                embed_dim=encoder.backbone.embed_dim,
                predictor_embed_dim=pred_embed_dim,
                depth=pred_depth,
                num_heads=encoder.backbone.num_heads,
                uniform_power=uniform_power,
                num_mask_tokens=num_mask_tokens,
                zero_init_mask_tokens=True,
                use_sdpa=use_sdpa,
                use_SiLU=use_SiLU,
                use_rope=use_rope,
                rope_is_1D=rope_is_1D,
                wide_SiLU=wide_SiLU,
            )
        else:
            raise ValueError(f"unsupported predictor_type: {predictor_type}")
        if predictor_type != "onn_causal_recurrent":
            predictor = PredictorMultiMaskWrapper(predictor)
        predictor.to(device)
        target_encoder.to(device)

    encoder.to(device)
    encoder, target_encoder, predictor = load_pretrained(
        encoder=encoder,
        predictor=predictor,
        target_encoder=target_encoder,
        pretrained=pretrained,
        enc_checkpoint_key=enc_checkpoint_key,
        pred_checkpoint_key=pred_checkpoint_key,
        is_mae=is_mae,
        load_predictor=(
            predictor_type not in {"onn_feedback", "onn_causal_recurrent"}
        ),
        load_predictor_embed=(
            predictor_type == "onn_causal_recurrent"
        ),
    )
    if (
        predictor_type not in {"onn_feedback", "onn_causal_recurrent"}
        and optical_qkv.get("qkv_backend") == "fsonn_tdm"
    ):
        optical_config = OpticalQKVConfig.from_mapping(optical_qkv)
        vit_pred.install_optical_qkv(
            predictor,
            optical_config=optical_config,
            replace_layers=optical_qkv.get("replace_layers", "all"),
        )
    if predictor_checkpoint is not None:
        _load_trained_predictor(predictor, predictor_checkpoint)
    return encoder, target_encoder, predictor
