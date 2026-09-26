"""Public smoke tests for the bundled model's D4 inference mode."""

import unittest

import torch

from model import UltimateNet
from symmetry import inverse_policy, transform_spatial


class InferenceSymmetryTests(unittest.TestCase):
    def test_d4_ensemble_is_equivariant(self):
        torch.manual_seed(7)
        model = UltimateNet(channels=8, blocks=1, value_hidden=16)
        model.d4_ensemble = True
        model.eval()
        state = torch.randn(1, 6, 9, 9)
        with torch.no_grad():
            reference_policy, reference_value = model(state)
            for transform in range(8):
                policy, value = model(transform_spatial(state, transform))
                aligned = inverse_policy(policy, transform)
                self.assertTrue(torch.allclose(reference_policy, aligned, atol=1e-5))
                self.assertTrue(torch.allclose(reference_value, value, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
