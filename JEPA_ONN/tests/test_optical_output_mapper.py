import unittest

import torch

from src.models.optical_output import OpticalOutputConfig, OpticalOutputMapper


class OpticalOutputMapperTests(unittest.TestCase):
    def test_default_config_has_two_slms_and_three_8000_um_segments(self):
        config = OpticalOutputConfig.from_mapping({})
        self.assertEqual(config.num_slm_layers, 2)
        self.assertEqual(config.grid_size, 32)
        self.assertEqual(config.input_height * config.input_width, 384)
        self.assertEqual(config.grid_size * config.grid_size, 1024)
        self.assertEqual(config.slm_intervals_um, (8000.0,))
        self.assertEqual(config.input_distance_um, 8000.0)
        self.assertEqual(config.output_distance_um, 8000.0)
        self.assertEqual(config.all_distances_um, (8000.0, 8000.0, 8000.0))

    def test_invalid_fixed_geometry_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "num_slm_layers"):
            OpticalOutputConfig(num_slm_layers=3)
        with self.assertRaisesRegex(ValueError, "grid_size"):
            OpticalOutputConfig(grid_size=16)
        with self.assertRaisesRegex(ValueError, "input_height"):
            OpticalOutputConfig(input_height=15)
        with self.assertRaisesRegex(ValueError, "slm_intervals_um"):
            OpticalOutputConfig(slm_intervals_um=(8000.0, 8000.0))

    def test_signed_input_encoding_preserves_values_and_zero_padding(self):
        mapper = OpticalOutputMapper()
        x = torch.zeros(1, 1, 384)
        x[0, 0, 0] = 2.0
        x[0, 0, 1] = -3.0

        debug = mapper.inspect_input(x)
        grid = debug["optical_input_grid"]
        amplitude = debug["optical_input_amplitude"]
        phase = debug["optical_input_phase"]

        self.assertEqual(tuple(grid.shape), (1, 32, 32))
        self.assertEqual(tuple(amplitude.shape), (1, 32, 32))
        self.assertEqual(tuple(phase.shape), (1, 32, 32))
        self.assertEqual(grid[0, 8, 4].item(), 2.0)
        self.assertEqual(grid[0, 8, 5].item(), -3.0)
        self.assertEqual(amplitude[0, 8, 4].item(), 2.0)
        self.assertEqual(amplitude[0, 8, 5].item(), 3.0)
        self.assertEqual(phase[0, 8, 4].item(), 0.0)
        self.assertAlmostEqual(phase[0, 8, 5].item(), float(torch.pi), places=6)
        self.assertEqual(amplitude[0, 0, 0].item(), 0.0)
        self.assertEqual(amplitude[0, 31, 31].item(), 0.0)

    def test_output_is_per_token_centered_and_restores_target_shape(self):
        mapper = OpticalOutputMapper()
        x = torch.randn(2, 3, 384)
        output = mapper(x)

        self.assertEqual(tuple(output.shape), (2, 3, 1024))
        self.assertTrue(torch.allclose(output.mean(dim=-1), torch.zeros(2, 3), atol=1e-6))

        intensity = torch.arange(2 * 1024, dtype=torch.float32).reshape(2, 1024)
        centered = OpticalOutputMapper.center_intensity(intensity)
        expected = intensity - intensity.mean(dim=-1, keepdim=True)
        self.assertTrue(torch.equal(centered, expected))

    def test_mapper_has_two_shared_slm_parameter_planes_and_gradients(self):
        mapper = OpticalOutputMapper()
        self.assertEqual(len(mapper.slm_layers), 2)
        self.assertEqual(sum(parameter.numel() for parameter in mapper.parameters()), 2048)

        x = torch.randn(2, 2, 384, requires_grad=True)
        output = mapper(x)
        output.square().mean().backward()

        self.assertIsNotNone(mapper.slm_layers[0].phase_logits.grad)
        self.assertIsNotNone(mapper.slm_layers[1].phase_logits.grad)
        self.assertTrue(torch.isfinite(mapper.slm_layers[0].phase_logits.grad).all())
        self.assertTrue(torch.isfinite(mapper.slm_layers[1].phase_logits.grad).all())


if __name__ == "__main__":
    unittest.main()
