"""Optical 384-to-1024 output mapper for the ONN feedback Predictor."""

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.fsonn import PhaseSLM, band_limited_angular_spectrum


@dataclass(frozen=True)
class OpticalOutputConfig:
    """Fixed two-SLM optical mapper configuration."""

    num_slm_layers: int = 2
    grid_size: int = 32
    input_height: int = 16
    input_width: int = 24
    wavelength_nm: float = 532.0
    pixel_size_um: float = 8.0
    input_distance_um: float = 8000.0
    slm_intervals_um: Tuple[float, ...] = (8000.0,)
    output_distance_um: float = 8000.0
    asm_padding_factor: float = 2.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, object] | None):
        values = dict(values or {})
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(values).difference(allowed))
        if unknown:
            raise ValueError(
                "unknown optical output config fields: "
                + ", ".join(unknown)
            )
        normalized = dict(values)
        if "slm_intervals_um" in normalized:
            normalized["slm_intervals_um"] = tuple(
                float(distance) for distance in normalized["slm_intervals_um"]
            )
        return cls(**normalized)

    @property
    def input_dim(self) -> int:
        return self.input_height * self.input_width

    @property
    def input_padding(self) -> Tuple[int, int, int, int]:
        pad_height = self.grid_size - self.input_height
        pad_width = self.grid_size - self.input_width
        top = pad_height // 2
        left = pad_width // 2
        return (
            left,
            pad_width - left,
            top,
            pad_height - top,
        )

    @property
    def output_dim(self) -> int:
        return self.grid_size * self.grid_size

    @property
    def all_distances_um(self) -> Tuple[float, ...]:
        return (
            float(self.input_distance_um),
            *tuple(float(distance) for distance in self.slm_intervals_um),
            float(self.output_distance_um),
        )

    def __post_init__(self):
        if self.num_slm_layers != 2:
            raise ValueError("optical output requires num_slm_layers == 2")
        if self.grid_size != 32:
            raise ValueError("optical output requires grid_size == 32")
        if self.input_height * self.input_width != 384:
            raise ValueError(
                "optical output requires input_height * input_width == 384"
            )
        if self.grid_size * self.grid_size != 1024:
            raise ValueError("optical output requires grid_size * grid_size == 1024")
        if self.input_height > self.grid_size or self.input_width > self.grid_size:
            raise ValueError(
                "input_height and input_width must not exceed grid_size"
            )
        if len(self.slm_intervals_um) != self.num_slm_layers - 1:
            raise ValueError(
                "slm_intervals_um must contain one distance between the two SLMs"
            )
        if self.input_distance_um <= 0:
            raise ValueError("input_distance_um must be positive")
        if self.output_distance_um <= 0:
            raise ValueError("output_distance_um must be positive")
        if any(distance <= 0 for distance in self.slm_intervals_um):
            raise ValueError("all SLM interval distances must be positive")
        if self.pixel_size_um <= 0:
            raise ValueError("pixel_size_um must be positive")
        if self.wavelength_nm <= 0:
            raise ValueError("wavelength_nm must be positive")
        if self.asm_padding_factor < 1.0:
            raise ValueError("asm_padding_factor must be at least one")


class OpticalOutputMapper(nn.Module):
    """Shared two-SLM 384-to-1024 optical output mapper."""

    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = OpticalOutputConfig()
        elif not isinstance(config, OpticalOutputConfig):
            config = OpticalOutputConfig.from_mapping(config)
        self.config = config
        self.slm_layers = nn.ModuleList(
            [
                PhaseSLM(config.grid_size, config.grid_size)
                for _ in range(config.num_slm_layers)
            ]
        )

    @staticmethod
    def _autocast_disabled(device):
        if device.type in {"cpu", "cuda"}:
            return torch.autocast(device_type=device.type, enabled=False)
        return nullcontext()

    def _prepare_input(self, x: torch.Tensor):
        if x.ndim != 3 or x.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"expected input tensor with shape [B, Nt, {self.config.input_dim}]"
            )
        x_float = x.float()
        grid = x_float.reshape(-1, self.config.input_height, self.config.input_width)
        grid = F.pad(
            grid,
            self.config.input_padding,
            mode="constant",
            value=0.0,
        )
        assert grid.shape[-2:] == (self.config.grid_size, self.config.grid_size)
        amplitude = grid.abs()
        phase = torch.where(
            grid < 0,
            torch.full_like(grid, torch.pi),
            torch.zeros_like(grid),
        )
        input_field = torch.polar(amplitude.float(), phase.float())
        if input_field.dtype != torch.complex64:
            input_field = input_field.to(torch.complex64)
        return grid, amplitude, phase, input_field

    def inspect_input(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        grid, amplitude, phase, input_field = self._prepare_input(x)
        del input_field
        return {
            "optical_input_grid": grid,
            "optical_input_amplitude": amplitude,
            "optical_input_phase": phase,
        }

    @staticmethod
    def center_intensity(intensity_flat: torch.Tensor) -> torch.Tensor:
        if intensity_flat.ndim != 2 or intensity_flat.shape[-1] != 1024:
            raise ValueError("intensity_flat must have shape [N, 1024]")
        output_mean = intensity_flat.mean(dim=-1, keepdim=True)
        return intensity_flat - output_mean

    def forward(self, x: torch.Tensor, return_debug: bool = False):
        batch_size, num_target_tokens, _ = x.shape
        with self._autocast_disabled(x.device):
            grid, amplitude, phase, field = self._prepare_input(x)
            field = band_limited_angular_spectrum(
                field,
                self.config.input_distance_um,
                self.config.pixel_size_um,
                self.config.wavelength_nm,
                self.config.asm_padding_factor,
            )
            for layer_index, slm in enumerate(self.slm_layers):
                field = slm(field)
                if layer_index < len(self.slm_layers) - 1:
                    field = band_limited_angular_spectrum(
                        field,
                        self.config.slm_intervals_um[layer_index],
                        self.config.pixel_size_um,
                        self.config.wavelength_nm,
                        self.config.asm_padding_factor,
                    )
            detector_field = band_limited_angular_spectrum(
                field,
                self.config.output_distance_um,
                self.config.pixel_size_um,
                self.config.wavelength_nm,
                self.config.asm_padding_factor,
            )
            intensity = detector_field.abs().square()
            intensity_flat = intensity.reshape(-1, self.config.output_dim)
            output_mean = intensity_flat.mean(dim=-1, keepdim=True)
            output_flat = intensity_flat - output_mean
            output = output_flat.reshape(
                batch_size,
                num_target_tokens,
                self.config.output_dim,
            )
            assert output.shape == (
                batch_size,
                num_target_tokens,
                self.config.output_dim,
            )

        if not return_debug:
            return output

        slm_phases = torch.stack(
            [
                2.0 * torch.pi * torch.sigmoid(slm.phase_logits)
                for slm in self.slm_layers
            ]
        )
        debug = {
            "optical_input_grid": grid,
            "optical_input_amplitude": amplitude,
            "optical_input_phase": phase,
            "optical_slm_phases": slm_phases,
            "optical_detector_field": detector_field,
            "optical_intensity": intensity,
            "optical_output_mean": output_mean,
            "optical_output": output,
        }
        return output, debug
