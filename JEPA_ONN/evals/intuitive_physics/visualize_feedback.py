"""Visualize one eval-only ONN Feedback Predictor forward pass."""

import argparse
import copy
import csv
import json
import math
import os
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.intuitive_physics.eval import init_model
from evals.intuitive_physics.train_optical import (
    _make_jepa_loader,
    _make_jepa_mask_collator,
    _prepare_jepa_batch,
)


FULL_MODES = {"onn_feedback", "end_to_end_jepa"}


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _to_numpy(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.dtype == torch.bfloat16:
            value = value.float()
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _sample_tensor(value):
    array = _to_numpy(value)
    if array is None:
        return None
    return array[0] if array.ndim >= 3 and array.shape[0] == 1 else array


def _sample_phase_stack(value):
    array = _to_numpy(value)
    if array is None:
        return None
    if array.ndim == 4:
        if array.shape[1] != 1:
            raise ValueError(
                "visualization currently expects a single sample for phase traces"
            )
        return array[:, 0]
    if array.ndim == 3:
        return array
    if array.ndim == 2:
        return array[None]
    raise ValueError(f"unexpected phase trace shape: {array.shape}")


def _wrap_phase(value):
    return np.arctan2(np.sin(value), np.cos(value))


def _value_stats(value):
    finite = np.asarray(value)[np.isfinite(value)]
    if finite.size == 0:
        return {"min": None, "max": None, "mean": None, "std": None}
    return {
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
    }


def _robust_limits(value, center_zero=False, lower=0.5, upper=99.5):
    finite = np.asarray(value)[np.isfinite(value)]
    if finite.size == 0:
        return -1.0, 1.0
    low = float(np.percentile(finite, lower))
    high = float(np.percentile(finite, upper))
    if center_zero and float(finite.min()) < 0.0 < float(finite.max()):
        bound = max(abs(low), abs(high), 1.0e-6)
        return -bound, bound
    if math.isclose(low, high):
        margin = max(abs(low) * 0.05, 1.0e-6)
        return low - margin, high + margin
    return low, high


def _symmetric_limit(value, percentile=99.5):
    finite = np.asarray(value)[np.isfinite(value)]
    if finite.size == 0:
        return 1.0
    limit = float(np.percentile(np.abs(finite), percentile))
    return max(limit, 1.0e-6)


def _phase_statistics(base_phases):
    return [
        {
            "physical_slm_index": index + 1,
            **_value_stats(phase),
        }
        for index, phase in enumerate(base_phases)
    ]


def _active_indices(onn_config):
    if not onn_config.get("feedback_enabled", True):
        return []
    mode = onn_config.get("feedback_layer_mode", "single")
    if mode == "single":
        index = onn_config.get("feedback_layer_index")
        if index is None:
            raise ValueError("checkpoint ONN config has no feedback_layer_index")
        return [int(index)]
    indices = onn_config.get("feedback_layer_indices")
    if not indices:
        raise ValueError("multi-feedback checkpoint has no feedback_layer_indices")
    return [int(index) for index in indices]


def _checkpoint_config(checkpoint):
    mode = checkpoint.get("mode")
    if mode not in FULL_MODES:
        raise ValueError(
            "visualization requires a full checkpoint with "
            f"mode in {sorted(FULL_MODES)}, got {mode!r}"
        )
    if "predictor" not in checkpoint or "config" not in checkpoint:
        raise ValueError("checkpoint must contain both predictor and config")
    config = copy.deepcopy(checkpoint["config"])
    if not isinstance(config, dict):
        raise ValueError("checkpoint config must be a mapping")
    if "onn" not in config:
        saved_onn = checkpoint.get("onn") or checkpoint.get("onn_feedback")
        if not saved_onn:
            raise ValueError("checkpoint has no ONN configuration")
        config["onn"] = copy.deepcopy(saved_onn)
    config["predictor_type"] = "onn_feedback"
    predictor_config = dict(config.get("predictor") or {})
    if "output_mode" not in predictor_config or predictor_config["output_mode"] is None:
        keys = checkpoint["predictor"].keys()
        if any("output_linear.weight" in key for key in keys):
            predictor_config["output_mode"] = "linear"
        elif any("output_mlp.0.weight" in key for key in keys):
            predictor_config["output_mode"] = "mlp"
        else:
            predictor_config["output_mode"] = "mlp"
    config["predictor"] = predictor_config
    return config


def _select_video_id(checkpoint, sample_index, video_id):
    split = checkpoint.get("data_split") or {}
    train_ids = [str(item) for item in split.get("train_video_ids", [])]
    if video_id is not None:
        if train_ids and str(video_id) not in train_ids:
            raise ValueError(f"video_id {video_id!r} is not in checkpoint train split")
        return str(video_id), sample_index
    if not train_ids:
        raise ValueError("checkpoint data_split has no train_video_ids")
    if not 0 <= int(sample_index) < len(train_ids):
        raise IndexError(
            f"sample_index must be in [0, {len(train_ids) - 1}]"
        )
    return train_ids[int(sample_index)], int(sample_index)


def _default_output_root(checkpoint_path):
    return checkpoint_path.parent


def _plot_chunk_grid(output_dir, values, filename, title, colorbar_label,
                     cmap, vmin, vmax):
    figure, axes = plt.subplots(2, 4, figsize=(14, 6.8), squeeze=False)
    image = None
    for index in range(8):
        image = axes[index // 4, index % 4].imshow(
            values[index], vmin=vmin, vmax=vmax, cmap=cmap
        )
        axes[index // 4, index % 4].set_title(f"t={index}")
        axes[index // 4, index % 4].set_xticks([])
        axes[index // 4, index % 4].set_yticks([])
    figure.suptitle(title)
    figure.colorbar(
        image, ax=list(axes.ravel()), label=colorbar_label, shrink=0.88
    )
    figure.savefig(output_dir / filename, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_input_phase(output_dir, input_phases):
    _plot_chunk_grid(
        output_dir,
        input_phases,
        "figure_E_input_phase.png",
        "ONN input phase before propagation",
        "ONN input phase (rad)",
        "twilight",
        -np.pi,
        np.pi,
    )


def _plot_input_amplitude(output_dir, input_amplitudes):
    low, high = _robust_limits(input_amplitudes)
    _plot_chunk_grid(
        output_dir,
        input_amplitudes,
        "figure_E_input_amplitude.png",
        "ONN input amplitude before propagation",
        "ONN input amplitude",
        "viridis",
        low,
        high,
    )


def _plot_input_phase_difference(output_dir, input_phases):
    phase_differences = _wrap_phase(input_phases - input_phases[0:1])
    _plot_chunk_grid(
        output_dir,
        phase_differences,
        "figure_E_input_phase_difference.png",
        "ONN input phase difference from t=0",
        "wrapped input phase difference (rad)",
        "twilight",
        -np.pi,
        np.pi,
    )


def _plot_base(output_dir, base_phases):
    displayed = _wrap_phase(base_phases)
    num_layers = displayed.shape[0]
    num_columns = min(3, num_layers)
    num_rows = int(math.ceil(num_layers / num_columns))
    figure, axes = plt.subplots(
        num_rows,
        num_columns,
        squeeze=False,
        figsize=(4.2 * num_columns, 4.2 * num_rows),
    )
    axes = axes.ravel()
    used_axes = axes[:num_layers]
    images = []
    for index, axis in enumerate(used_axes):
        image = axis.imshow(
            displayed[index], vmin=-np.pi, vmax=np.pi, cmap="twilight"
        )
        images.append(image)
        axis.set_title(f"SLM{index + 1} base phase")
        axis.set_xlabel("output feature")
        axis.set_ylabel("chunk token")
    for axis in axes[num_layers:]:
        axis.axis("off")
    figure.colorbar(
        images[0], ax=list(used_axes), label="base phase (rad)", shrink=0.85
    )
    figure.savefig(
        output_dir / "figure_A_base_phase.png", dpi=180, bbox_inches="tight"
    )
    plt.close(figure)


def _plot_base_contrast(output_dir, base_phases):
    deviations = base_phases - np.pi
    num_layers = deviations.shape[0]
    num_columns = min(3, num_layers)
    num_rows = int(math.ceil(num_layers / num_columns))
    limit = _symmetric_limit(deviations)
    figure, axes = plt.subplots(
        num_rows,
        num_columns,
        squeeze=False,
        figsize=(4.2 * num_columns, 4.2 * num_rows),
    )
    axes = axes.ravel()
    used_axes = axes[:num_layers]
    image = None
    for index, axis in enumerate(used_axes):
        image = axis.imshow(
            deviations[index], vmin=-limit, vmax=limit, cmap="RdBu_r"
        )
        axis.set_title(f"SLM{index + 1}: base phase - π")
        axis.set_xlabel("output feature")
        axis.set_ylabel("chunk token")
    for axis in axes[num_layers:]:
        axis.axis("off")
    figure.colorbar(
        image, ax=list(used_axes), label="base phase - π (rad)", shrink=0.85
    )
    figure.savefig(
        output_dir / "figure_A_base_phase_contrast.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def _plot_detector(output_dir, outputs, detector_phases):
    figure, axes = plt.subplots(2, 8, figsize=(24, 6.6), squeeze=False)
    output_low, output_high = _robust_limits(outputs, center_zero=True)
    output_image = None
    phase_image = None
    for index in range(8):
        output_image = axes[0, index].imshow(
            outputs[index], vmin=output_low, vmax=output_high, cmap="viridis"
        )
        phase_image = axes[1, index].imshow(
            detector_phases[index], vmin=-np.pi, vmax=np.pi, cmap="twilight"
        )
        axes[0, index].set_title(f"t={index}")
        for row in range(2):
            axes[row, index].set_xticks([])
            axes[row, index].set_yticks([])
    axes[0, 0].set_ylabel("ONN output y_t")
    axes[1, 0].set_ylabel("detector phase")
    figure.colorbar(
        output_image, ax=list(axes[0]), label="ONN output y_t", shrink=0.85
    )
    figure.colorbar(
        phase_image, ax=list(axes[1]), label="detector phase (rad)", shrink=0.85
    )
    figure.savefig(
        output_dir / "figure_B_detector_output_and_phase.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def _plot_detector_contrast(output_dir, outputs, detector_phases):
    output_deltas = outputs - outputs[0:1]
    phase_deltas = _wrap_phase(detector_phases - detector_phases[0:1])
    output_limit = _symmetric_limit(output_deltas)
    figure, axes = plt.subplots(2, 8, figsize=(24, 6.6), squeeze=False)
    output_image = None
    phase_image = None
    for index in range(8):
        output_image = axes[0, index].imshow(
            output_deltas[index],
            vmin=-output_limit,
            vmax=output_limit,
            cmap="RdBu_r",
        )
        phase_image = axes[1, index].imshow(
            phase_deltas[index], vmin=-np.pi, vmax=np.pi, cmap="twilight"
        )
        axes[0, index].set_title(f"t={index}")
        for row in range(2):
            axes[row, index].set_xticks([])
            axes[row, index].set_yticks([])
    axes[0, 0].set_ylabel("y_t - y_0")
    axes[1, 0].set_ylabel("phase_t - phase_0")
    figure.colorbar(
        output_image, ax=list(axes[0]), label="ONN output difference", shrink=0.85
    )
    figure.colorbar(
        phase_image,
        ax=list(axes[1]),
        label="wrapped detector phase difference (rad)",
        shrink=0.85,
    )
    figure.savefig(
        output_dir / "figure_B_detector_output_and_phase_contrast.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def _plot_feedback(output_dir, states, state_valid, feedback_phases,
                   effective_phases, active_indices, phase_max,
                   filename="figure_C_feedback_and_effective_phase.png"):
    num_layers = len(active_indices)
    columns = 1 + 2 * num_layers
    figure = plt.figure(
        figsize=(3.0 * columns + 2.6, 1.9 * 8), constrained_layout=False
    )
    grid = figure.add_gridspec(
        8,
        columns + 3,
        width_ratios=[1.0] * columns + [0.10, 0.10, 0.10],
        wspace=0.28,
        hspace=0.42,
    )
    axes = np.empty((8, columns), dtype=object)
    for row in range(8):
        for column in range(columns):
            axes[row, column] = figure.add_subplot(grid[row, column])
    colorbar_axes = [
        figure.add_subplot(grid[:, columns + index]) for index in range(3)
    ]
    valid_states = states[state_valid]
    state_limit = (
        max(abs(_robust_limits(valid_states, center_zero=True)[1]), 1.0e-6)
        if valid_states.size else 1.0
    )
    state_image = None
    delta_image = None
    effective_image = None
    for chunk_index in range(8):
        state_axis = axes[chunk_index, 0]
        state_axis.set_ylabel(f"t={chunk_index}", rotation=0, labelpad=18)
        if chunk_index == 0:
            state_axis.set_title("Z_t")
        if not state_valid[chunk_index]:
            state_axis.text(
                0.5, 0.5, "no previous output", ha="center", va="center"
            )
        else:
            state_image = state_axis.imshow(
                states[chunk_index],
                vmin=-state_limit,
                vmax=state_limit,
                cmap="RdBu_r",
            )
        for layer_position, layer_index in enumerate(active_indices):
            delta_column = 1 + 2 * layer_position
            effective_column = delta_column + 1
            delta_axis = axes[chunk_index, delta_column]
            effective_axis = axes[chunk_index, effective_column]
            if chunk_index == 0:
                delta_axis.set_title(f"Δφ SLM{layer_index + 1}")
                effective_axis.set_title(f"φeff SLM{layer_index + 1}")
            if chunk_index == 0 or not state_valid[chunk_index]:
                delta_axis.text(
                    0.5, 0.5, "no feedback", ha="center", va="center"
                )
            else:
                delta_image = delta_axis.imshow(
                    feedback_phases[chunk_index, layer_position],
                    vmin=-phase_max,
                    vmax=phase_max,
                    cmap="RdBu_r",
                )
            effective_image = effective_axis.imshow(
                _wrap_phase(effective_phases[chunk_index, layer_index]),
                vmin=-np.pi,
                vmax=np.pi,
                cmap="twilight",
            )
        for column in range(columns):
            axes[chunk_index, column].set_xticks([])
            axes[chunk_index, column].set_yticks([])
    if state_image is not None:
        figure.colorbar(
            state_image, cax=colorbar_axes[0], label="feedback state Z_t"
        )
    else:
        colorbar_axes[0].axis("off")
    if delta_image is not None:
        figure.colorbar(
            delta_image, cax=colorbar_axes[1], label="feedback phase Δφ (rad)"
        )
    else:
        colorbar_axes[1].axis("off")
    if effective_image is not None:
        figure.colorbar(
            effective_image, cax=colorbar_axes[2], label="effective phase (rad)"
        )
    else:
        colorbar_axes[2].axis("off")
    figure.savefig(
        output_dir / filename,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def _plot_effective_phase_contrast(output_dir, base_phases,
                                   effective_phases, active_indices, phase_max,
                                   filename="figure_C_effective_phase_contrast.png"):
    num_layers = len(active_indices)
    if not num_layers:
        return
    deviations = _wrap_phase(
        effective_phases[:, active_indices]
        - base_phases[active_indices][None, ...]
    )
    figure = plt.figure(
        figsize=(3.4 * num_layers + 2.0, 1.9 * 8), constrained_layout=False
    )
    grid = figure.add_gridspec(
        8,
        num_layers + 1,
        width_ratios=[1.0] * num_layers + [0.12],
        wspace=0.28,
        hspace=0.42,
    )
    axes = np.empty((8, num_layers), dtype=object)
    for row in range(8):
        for column in range(num_layers):
            axes[row, column] = figure.add_subplot(grid[row, column])
    colorbar_axis = figure.add_subplot(grid[:, num_layers])
    image = None
    for chunk_index in range(8):
        for layer_position, layer_index in enumerate(active_indices):
            axis = axes[chunk_index, layer_position]
            if chunk_index == 0:
                axis.set_title(f"φeff - φbase SLM{layer_index + 1}")
            axis.set_ylabel(f"t={chunk_index}", rotation=0, labelpad=18)
            if chunk_index == 0:
                axis.text(
                    0.5, 0.5, "no feedback", ha="center", va="center"
                )
            else:
                image = axis.imshow(
                    deviations[chunk_index, layer_position],
                    vmin=-phase_max,
                    vmax=phase_max,
                    cmap="RdBu_r",
                )
            axis.set_xticks([])
            axis.set_yticks([])
    if image is not None:
        figure.colorbar(
            image,
            cax=colorbar_axis,
            label="effective phase - base phase (rad)",
        )
    else:
        colorbar_axis.axis("off")
    figure.savefig(
        output_dir / filename,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(figure)


def _feedback_metrics(states, state_valid, feedback_phases,
                      effective_phases, active_indices, feedback_used,
                      phase_max):
    delta_abs = np.abs(feedback_phases)
    delta_mean_abs = delta_abs.mean(axis=(2, 3))
    delta_rms = np.sqrt(np.mean(feedback_phases ** 2, axis=(2, 3)))
    delta_max_abs = delta_abs.max(axis=(2, 3))
    saturation_ratio = (delta_abs > 0.9 * phase_max).mean(axis=(2, 3))
    z_mean = np.full(8, np.nan, dtype=np.float64)
    z_std = np.full(8, np.nan, dtype=np.float64)
    z_rms = np.full(8, np.nan, dtype=np.float64)
    z_cosine = np.full(8, np.nan, dtype=np.float64)
    z_change_rms = np.full(8, np.nan, dtype=np.float64)
    z_mean[state_valid] = states[state_valid].mean(axis=(1, 2))
    z_std[state_valid] = states[state_valid].std(axis=(1, 2))
    z_rms[state_valid] = np.sqrt(np.mean(states[state_valid] ** 2, axis=(1, 2)))
    for chunk_index in range(1, 8):
        if state_valid[chunk_index] and state_valid[chunk_index - 1]:
            current = states[chunk_index].reshape(-1)
            previous = states[chunk_index - 1].reshape(-1)
            current_norm = np.linalg.norm(current)
            previous_norm = np.linalg.norm(previous)
            if current_norm > 0.0 and previous_norm > 0.0:
                z_cosine[chunk_index] = float(
                    np.dot(current, previous) / (current_norm * previous_norm)
                )
            z_change_rms[chunk_index] = float(
                np.sqrt(np.mean((current - previous) ** 2))
            )
    effective_display = _wrap_phase(effective_phases)
    rows = []
    for chunk_index in range(8):
        for layer_position, layer_index in enumerate(active_indices):
            delta = feedback_phases[chunk_index, layer_position]
            effective = effective_display[chunk_index, layer_index]
            rows.append(
                {
                    "chunk_index": chunk_index,
                    "physical_slm_index": layer_index + 1,
                    "code_layer_index": layer_index,
                    "feedback_used": bool(feedback_used[chunk_index]),
                    "z_mean": z_mean[chunk_index],
                    "z_std": z_std[chunk_index],
                    "z_rms": z_rms[chunk_index],
                    "z_cosine_prev": z_cosine[chunk_index],
                    "z_change_rms": z_change_rms[chunk_index],
                    "delta_mean": delta.mean(),
                    "delta_mean_abs": delta_mean_abs[chunk_index, layer_position],
                    "delta_rms": delta_rms[chunk_index, layer_position],
                    "delta_max_abs": delta_max_abs[chunk_index, layer_position],
                    "feedback_saturation_ratio": saturation_ratio[
                        chunk_index, layer_position
                    ],
                    "effective_phase_mean": effective.mean(),
                    "effective_phase_std": effective.std(),
                }
            )
    return (
        rows,
        delta_mean_abs,
        delta_rms,
        delta_max_abs,
        z_mean,
        z_std,
        z_rms,
        z_cosine,
        z_change_rms,
        saturation_ratio,
    )


def _plot_curves(output_dir, metrics, active_indices, layer_position=None,
                 filename="figure_D1_feedback_curves.png"):
    (
        _, delta_mean_abs, delta_rms, delta_max_abs,
        z_mean, z_std, z_rms, z_cosine, z_change_rms,
        saturation_ratio,
    ) = metrics
    positions = (
        list(range(len(active_indices)))
        if layer_position is None
        else [int(layer_position)]
    )
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(positions), 1)))
    figure, axes = plt.subplots(3, 3, figsize=(16, 12), squeeze=False)
    names = [
        ("mean(|Δφ|)", delta_mean_abs),
        ("RMS(Δφ)", delta_rms),
        ("max(|Δφ|)", delta_max_abs),
    ]
    for axis, (title, values) in zip(axes[0], names):
        for color_position, position in enumerate(positions):
            layer_index = active_indices[position]
            axis.plot(
                range(8), values[:, position], marker="o",
                color=colors[color_position],
                label=f"physical SLM{layer_index + 1}"
            )
        axis.set_title(title)
        axis.set_xlabel("chunk index t")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].plot(range(8), z_mean, marker="o", label="Z mean")
    axes[1, 0].plot(range(8), z_std, marker="o", label="Z std")
    axes[1, 0].plot(range(8), z_rms, marker="o", label="Z RMS")
    axes[1, 0].set_title("LayerNorm checks (not feedback magnitude)")
    axes[1, 0].set_xlabel("chunk index t")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].plot(range(8), z_cosine, marker="o")
    axes[1, 1].set_title("cos(Z_t, Z_{t-1})")
    axes[1, 1].set_xlabel("chunk index t")
    axes[1, 1].set_ylim(-1.05, 1.05)
    axes[1, 1].grid(alpha=0.25)
    axes[1, 2].plot(range(8), z_change_rms, marker="o")
    axes[1, 2].set_title("RMS(Z_t - Z_{t-1})")
    axes[1, 2].set_xlabel("chunk index t")
    axes[1, 2].grid(alpha=0.25)
    for color_position, position in enumerate(positions):
        layer_index = active_indices[position]
        axes[2, 0].plot(
            range(8), saturation_ratio[:, position], marker="o",
            color=colors[color_position],
            label=f"physical SLM{layer_index + 1}"
        )
    axes[2, 0].set_title("feedback saturation ratio")
    axes[2, 0].set_xlabel("chunk index t")
    axes[2, 0].set_ylim(0.0, 1.0)
    axes[2, 0].grid(alpha=0.25)
    axes[2, 0].legend(fontsize=8)
    axes[2, 1].axis("off")
    axes[2, 2].axis("off")
    figure.tight_layout()
    figure.savefig(
        output_dir / filename,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def _write_csv(path, rows):
    fields = [
        "chunk_index", "physical_slm_index", "code_layer_index",
        "feedback_used", "z_mean", "z_std", "z_rms", "z_cosine_prev",
        "z_change_rms", "delta_mean", "delta_mean_abs", "delta_rms",
        "delta_max_abs", "feedback_saturation_ratio",
        "effective_phase_mean", "effective_phase_std",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _save_trace(output_dir, traces, active_indices):
    base_phases = _sample_phase_stack(traces[0]["base_phases"]).astype(np.float32)
    base_phases_wrapped = _wrap_phase(base_phases).astype(np.float32)
    base_phase_deviations = (base_phases - np.pi).astype(np.float32)
    input_phases = np.stack([
        _sample_tensor(item["input_phase"]) for item in traces
    ])
    input_amplitudes = np.stack([
        _sample_tensor(item["input_amplitude"]) for item in traces
    ])
    input_phase_differences = _wrap_phase(
        input_phases - input_phases[0:1]
    ).astype(np.float32)
    chunk_inputs = np.stack([_sample_tensor(item["input_chunk"]) for item in traces])
    outputs = np.stack([_sample_tensor(item["output"]) for item in traces])
    intensities = np.stack([
        _sample_tensor(item["detector_intensity"]) for item in traces
    ])
    fields = np.stack([
        _sample_tensor(item["detector_field"]) for item in traces
    ])
    detector_phases = np.stack([
        _sample_tensor(item["detector_phase"]) for item in traces
    ])
    effective_phases = np.stack([
        _sample_phase_stack(item["effective_phases"]) for item in traces
    ])
    effective_phases_wrapped = _wrap_phase(effective_phases).astype(np.float32)
    chunk_output_deltas = (outputs - outputs[0:1]).astype(np.float32)
    detector_phase_differences = _wrap_phase(
        detector_phases - detector_phases[0:1]
    ).astype(np.float32)
    if active_indices:
        effective_phase_deviations = _wrap_phase(
            effective_phases[:, active_indices]
            - base_phases[active_indices][None, ...]
        ).astype(np.float32)
    else:
        effective_phase_deviations = np.zeros(
            (8, 0, outputs.shape[-2], outputs.shape[-1]), dtype=np.float32
        )
    feedback_used = np.asarray(
        [bool(item.get("feedback_used", False)) for item in traces],
        dtype=bool,
    )
    height, width = outputs.shape[-2:]
    states = np.full((8, height, width), np.nan, dtype=np.float32)
    state_valid = np.zeros(8, dtype=bool)
    previous_outputs = np.full_like(states, np.nan)
    for index, item in enumerate(traces):
        state = _sample_tensor(item.get("feedback_state"))
        raw = _sample_tensor(item.get("feedback_source_raw"))
        if state is not None:
            states[index] = state
            state_valid[index] = True
        if raw is not None:
            previous_outputs[index] = raw
    if active_indices:
        feedback_phases = np.stack([
            np.stack([
                _sample_tensor(item["feedback_phases"].get(layer_index))
                if item["feedback_phases"].get(layer_index) is not None
                else np.zeros((height, width), dtype=np.float32)
                for layer_index in active_indices
            ])
            for item in traces
        ])
    else:
        feedback_phases = np.zeros(
            (8, 0, height, width), dtype=np.float32
        )
    np.savez(
        output_dir / "visualization_trace.npz",
        base_phases=base_phases,
        base_phases_wrapped=base_phases_wrapped,
        base_phase_deviations=base_phase_deviations,
        input_phases=input_phases,
        input_amplitudes=input_amplitudes,
        input_phase_differences=input_phase_differences,
        chunk_inputs=chunk_inputs,
        chunk_output_deltas=chunk_output_deltas,
        chunk_outputs=outputs,
        detector_intensities=intensities,
        detector_fields_real=fields.real.astype(np.float32),
        detector_fields_imag=fields.imag.astype(np.float32),
        detector_phases=detector_phases,
        detector_phase_differences=detector_phase_differences,
        feedback_states=states,
        feedback_state_valid=state_valid,
        previous_outputs=previous_outputs,
        feedback_phases=feedback_phases,
        effective_phases=effective_phases,
        effective_phases_wrapped=effective_phases_wrapped,
        effective_phase_deviations=effective_phase_deviations,
        feedback_used=feedback_used,
    )
    return {
        "base_phases": base_phases,
        "base_phases_wrapped": base_phases_wrapped,
        "base_phase_deviations": base_phase_deviations,
        "input_phases": input_phases,
        "input_amplitudes": input_amplitudes,
        "input_phase_differences": input_phase_differences,
        "chunk_inputs": chunk_inputs,
        "chunk_output_deltas": chunk_output_deltas,
        "outputs": outputs,
        "intensities": intensities,
        "fields": fields,
        "detector_phases": detector_phases,
        "detector_phase_differences": detector_phase_differences,
        "states": states,
        "state_valid": state_valid,
        "feedback_phases": feedback_phases,
        "effective_phases": effective_phases,
        "effective_phases_wrapped": effective_phases_wrapped,
        "effective_phase_deviations": effective_phase_deviations,
        "feedback_used": feedback_used,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--video-id")
    parser.add_argument(
        "--output-root",
        help="training-result directory that will contain visualizations",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    config = _checkpoint_config(checkpoint)
    video_id, resolved_index = _select_video_id(
        checkpoint, args.sample_index, args.video_id
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    runtime_config = copy.deepcopy(config)
    runtime_config.setdefault("data", {})["batch_size"] = 1
    encoder, target_encoder, predictor = init_model(
        device=device,
        pretrained=os.path.join(
            runtime_config["pretrain"]["folder"],
            runtime_config["pretrain"]["checkpoint"],
        ),
        model_name=runtime_config["pretrain"]["model_name"],
        patch_size=runtime_config["pretrain"].get("patch_size", 16),
        crop_size=runtime_config["data"].get("resolution", 224),
        frames_per_clip=runtime_config["data"].get("frames_per_clip", 16),
        tubelet_size=runtime_config["pretrain"].get("tubelet_size", 2),
        use_sdpa=runtime_config["pretrain"].get("use_sdpa", True),
        use_SiLU=runtime_config["pretrain"].get("use_silu", False),
        wide_SiLU=runtime_config["pretrain"].get("wide_silu", True),
        is_causal=runtime_config["pretrain"].get("is_causal", False),
        pred_is_causal=runtime_config["pretrain"].get("pred_is_causal", False),
        uniform_power=runtime_config["pretrain"].get("uniform_power", False),
        enc_checkpoint_key=runtime_config["pretrain"].get(
            "enc_checkpoint_key", "encoder"
        ),
        pred_checkpoint_key=runtime_config["pretrain"].get(
            "pred_checkpoint_key", "predictor"
        ),
        pred_embed_dim=runtime_config["predictor"].get("predictor_dim", 384),
        output_mode=runtime_config["predictor"].get("output_mode", "mlp"),
        direct_384_loss=runtime_config["predictor"].get(
            "direct_384_loss", False
        ),
        pred_depth=runtime_config["pretrain"].get("pred_depth", 12),
        optical_qkv=runtime_config.get("optical_qkv", {}),
        predictor_checkpoint=str(checkpoint_path),
        predictor_type="onn_feedback",
        onn_feedback_config=runtime_config["onn"],
    )
    encoder.eval()
    target_encoder.eval()
    predictor.eval()
    loader = _make_jepa_loader(
        runtime_config,
        [video_id],
        deterministic=True,
        collator=_make_jepa_mask_collator(runtime_config),
        world_size=1,
        rank=0,
    )
    batch = next(iter(loader))
    with torch.no_grad():
        _, context, targets, masks_ctxt, masks_tgt = _prepare_jepa_batch(
            batch, runtime_config, encoder, target_encoder, device
        )
        predictor(context, targets, masks_ctxt, masks_tgt, collect_trace=True)
    predictor_core = predictor.backbone
    traces = predictor_core.last_trace.get("chunks")
    if not traces or len(traces) != 8:
        raise RuntimeError("expected 8 chunk trace records")
    onn_config = runtime_config["onn"]
    active_indices = _active_indices(onn_config)
    output_root = Path(args.output_root) if args.output_root else _default_output_root(
        checkpoint_path
    )
    checkpoint_name = checkpoint_path.stem
    sample_id = f"sample_{resolved_index:04d}_{video_id}"
    output_dir = output_root / "visualizations" / sample_id
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = _save_trace(output_dir, traces, active_indices)
    _plot_base(output_dir, arrays["base_phases"])
    _plot_base_contrast(output_dir, arrays["base_phases"])
    _plot_detector(output_dir, arrays["outputs"], arrays["detector_phases"])
    _plot_detector_contrast(
        output_dir, arrays["outputs"], arrays["detector_phases"]
    )
    _plot_input_phase(output_dir, arrays["input_phases"])
    _plot_input_amplitude(output_dir, arrays["input_amplitudes"])
    _plot_input_phase_difference(
        output_dir, arrays["input_phase_differences"]
    )
    feedback_files = []
    if active_indices:
        metrics = _feedback_metrics(
            arrays["states"],
            arrays["state_valid"],
            arrays["feedback_phases"],
            arrays["effective_phases"],
            active_indices,
            arrays["feedback_used"],
            float(onn_config["feedback_phase_max_rad"]),
        )
        _write_csv(output_dir / "feedback_metrics.csv", metrics[0])
        feedback_files.append("feedback_metrics.csv")
        if len(active_indices) == 1:
            _plot_feedback(
                output_dir,
                arrays["states"],
                arrays["state_valid"],
                arrays["feedback_phases"],
                arrays["effective_phases"],
                active_indices,
                float(onn_config["feedback_phase_max_rad"]),
            )
            _plot_effective_phase_contrast(
                output_dir,
                arrays["base_phases"],
                arrays["effective_phases"],
                active_indices,
                float(onn_config["feedback_phase_max_rad"]),
            )
            _plot_curves(output_dir, metrics, active_indices)
            feedback_files.extend([
                "figure_C_feedback_and_effective_phase.png",
                "figure_C_effective_phase_contrast.png",
                "figure_D1_feedback_curves.png",
            ])
        else:
            for layer_position, layer_index in enumerate(active_indices):
                layer_number = layer_position + 1
                physical_number = layer_index + 1
                layer_feedback = arrays["feedback_phases"][
                    :, layer_position:layer_position + 1
                ]
                c_filename = (
                    f"figure_C{layer_number}_feedback_slm{physical_number}.png"
                )
                c_contrast_filename = (
                    f"figure_C{layer_number}_effective_phase_contrast_"
                    f"slm{physical_number}.png"
                )
                d_filename = (
                    f"figure_D1_{layer_number}_feedback_curves_"
                    f"slm{physical_number}.png"
                )
                _plot_feedback(
                    output_dir,
                    arrays["states"],
                    arrays["state_valid"],
                    layer_feedback,
                    arrays["effective_phases"],
                    [layer_index],
                    float(onn_config["feedback_phase_max_rad"]),
                    filename=c_filename,
                )
                _plot_effective_phase_contrast(
                    output_dir,
                    arrays["base_phases"],
                    arrays["effective_phases"],
                    [layer_index],
                    float(onn_config["feedback_phase_max_rad"]),
                    filename=c_contrast_filename,
                )
                _plot_curves(
                    output_dir,
                    metrics,
                    active_indices,
                    layer_position=layer_position,
                    filename=d_filename,
                )
                feedback_files.extend([
                    c_filename, c_contrast_filename, d_filename,
                ])
                layer_rows = [
                    row for row in metrics[0]
                    if int(row["code_layer_index"]) == layer_index
                ]
                layer_csv = f"feedback_metrics_slm{physical_number}.csv"
                _write_csv(output_dir / layer_csv, layer_rows)
                feedback_files.append(layer_csv)

    metadata = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_mode": checkpoint.get("mode"),
        "sample_index": resolved_index,
        "video_id": video_id,
        "sample_id": sample_id,
        "num_chunks": 8,
        "chunk_tokens": 196,
        "num_slm_layers": int(onn_config["num_slm_layers"]),
        "feedback_layer_mode": onn_config.get("feedback_layer_mode"),
        "feedback_layer_indices": active_indices,
        "physical_feedback_layers": [index + 1 for index in active_indices],
        "feedback_memory_enabled": bool(
            onn_config.get("feedback_memory_enabled", False)
        ),
        "feedback_memory_alpha": float(
            onn_config.get("feedback_memory_alpha", 0.8)
        ),
        "feedback_phase_max_rad": float(
            onn_config["feedback_phase_max_rad"]
        ),
        "phase_display_range": [-math.pi, math.pi],
        "output_definition": "intensity_minus_learnable_offset",
        "detector_intensity_definition": "|detector_field|^2",
        "detector_phase_definition": "angle(detector_field)",
        "effective_phase_definition": "base_phase_plus_feedback_delta",
        "feedback_state_definition": (
            "memory_state"
            if onn_config.get("feedback_memory_enabled", False)
            else "normalize_feedback_output(previous_output)"
        ),
        "feedback_state_valid": arrays["state_valid"].tolist(),
        "feedback_visualization_files": feedback_files,
        "feedback_visualizations_skipped": not bool(active_indices),
        "feedback_source_kind": [
            item.get("feedback_source_kind", "none") for item in traces
        ],
        "prediction_shape": list(
            predictor_core.last_trace.get("prediction_shape", ())
        ),
        "config": _jsonable(config),
        "input_visualization": {
            "input_phase": "angle(encoded_input)",
            "input_amplitude": "abs(encoded_input)",
            "input_phase_difference": "wrapped(input_phase_t - input_phase_0)",
        },
        "contrast_figures": {
            "base_phase": "base_phase - pi",
            "detector_output": "y_t - y_0",
            "detector_phase": "wrapped phase_t - phase_0",
            "effective_phase": "wrapped(phi_eff - phi_base)",
        },
        "contrast_scale_note": (
            "contrast figures use global symmetric percentile limits; "
            "they are for visibility and not a replacement for absolute-scale figures"
        ),
        "base_phase_statistics": _phase_statistics(arrays["base_phases"]),
        "output_statistics": _value_stats(arrays["outputs"]),
        "input_statistics": {
            "input_phase": _value_stats(arrays["input_phases"]),
            "input_amplitude": _value_stats(arrays["input_amplitudes"]),
            "input_phase_difference": _value_stats(
                arrays["input_phase_differences"]
            ),
        },
        "contrast_statistics": {
            "base_phase_deviation": _value_stats(arrays["base_phase_deviations"]),
            "chunk_output_delta": _value_stats(arrays["chunk_output_deltas"]),
            "detector_phase_difference": _value_stats(
                arrays["detector_phase_differences"]
            ),
            "effective_phase_deviation": _value_stats(
                arrays["effective_phase_deviations"]
            ),
        },
        "output_color_scale": {
            "method": "global 0.5-99.5 percentile",
            "centered_at_zero_if_signed": True,
            "vmin": _robust_limits(arrays["outputs"], center_zero=True)[0],
            "vmax": _robust_limits(arrays["outputs"], center_zero=True)[1],
        },
        "feedback_state_color_scale": {
            "method": "global 0.5-99.5 percentile, zero-centered",
            "vmin": _robust_limits(
                arrays["states"][arrays["state_valid"]], center_zero=True
            )[0] if arrays["state_valid"].any() else -1.0,
            "vmax": _robust_limits(
                arrays["states"][arrays["state_valid"]], center_zero=True
            )[1] if arrays["state_valid"].any() else 1.0,
        },
    }
    (output_dir / "visualization_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False)
    )
    print(output_dir)


if __name__ == "__main__":
    main()
