"""Release gate prevents a weak or asymmetric candidate from replacing v1.0."""

import unittest

from release_v11 import assess_release


class ReleaseGateTests(unittest.TestCase):
    def test_requires_strength_and_symmetry(self):
        league = {"match": {
            "opening_pairs": 200, "games": 400, "score_rate": 0.6,
            "score_rate_ci95": [0.53, 0.67],
            "protocol": {"simulations_per_move_each": 96},
        }}
        baseline = {"mean_policy_l1": 0.22, "mean_value_mae": 0.18}
        raw = {"mean_policy_l1": 0.20, "mean_value_mae": 0.13}
        ensemble = {"mean_policy_l1": 1e-7, "mean_value_mae": 1e-8}
        self.assertEqual(assess_release(league, baseline, raw, ensemble), [])
        league["match"]["score_rate_ci95"] = [0.49, 0.7]
        self.assertTrue(assess_release(league, baseline, raw, ensemble))
        league["match"]["score_rate_ci95"] = [0.53, 0.67]
        raw["mean_value_mae"] = 0.19
        self.assertTrue(assess_release(league, baseline, raw, ensemble))


if __name__ == "__main__":
    unittest.main()
