# -*- coding: utf-8 -*-
"""模型晋级联赛的快速测试。"""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from game import apply_move, legal_moves, new_game, terminal_info
from league import (
    export_champion,
    generate_openings,
    paired_bootstrap_ci,
    promotion_decision,
    run_league,
)
from model import UltimateNet


class _DummyModel:
    def eval(self):
        return self


def _first_legal(state, rng=None):
    return legal_moves(state)[0]


class LeagueTests(unittest.TestCase):
    def test_openings_are_reproducible_and_legal(self):
        first = generate_openings(4, opening_plies=6, seed=123)
        second = generate_openings(4, opening_plies=6, seed=123)
        self.assertEqual(first, second)
        for opening in first:
            state = new_game()
            for move in opening.moves:
                self.assertIn(move, legal_moves(state))
                state, done, _ = apply_move(state, move)
                self.assertFalse(done)
            self.assertEqual(state, opening.state)
            self.assertFalse(terminal_info(state)[0])

    def test_paired_bootstrap_is_reproducible(self):
        scores = [1.0, 0.75, 0.5, 0.25, 0.0]
        first = paired_bootstrap_ci(scores, samples=2_000, seed=9)
        second = paired_bootstrap_ci(scores, samples=2_000, seed=9)
        self.assertEqual(first, second)
        self.assertGreaterEqual(first[0], 0.0)
        self.assertLessEqual(first[1], 1.0)
        self.assertLessEqual(first[0], first[1])

    def test_every_opening_swaps_sides_with_equal_budget(self):
        candidate = _DummyModel()
        champion = _DummyModel()

        with patch("league.make_mcts_policy", return_value=_first_legal) as factory:
            report = run_league(
                candidate,
                champion,
                pairs=2,
                opening_plies=2,
                simulations=3,
                workers=1,
                c_puct=1.25,
                seed=5,
                bootstrap_samples=100,
            )

        self.assertEqual(report["games"], 4)
        self.assertEqual(report["wins"] + report["losses"] + report["draws"], 4)
        self.assertEqual(report["candidate_as_p1"]["games"], 2)
        self.assertEqual(report["candidate_as_p2"]["games"], 2)
        self.assertEqual(len(report["pairs"]), 2)
        self.assertTrue(report["protocol"]["swap_sides_for_every_opening"])
        self.assertFalse(report["protocol"]["dirichlet_noise"])
        self.assertEqual(factory.call_count, 8)
        self.assertTrue(all(call.args[1:] == (3, 1.25) for call in factory.call_args_list))

    def test_promotion_requires_both_thresholds(self):
        passing = {"score_rate": 0.60, "score_rate_ci95": [0.51, 0.70]}
        weak_ci = {"score_rate": 0.60, "score_rate_ci95": [0.49, 0.72]}
        self.assertTrue(promotion_decision(passing)["qualified"])
        self.assertFalse(promotion_decision(weak_ci)["qualified"])

    def test_export_champion_is_lightweight_and_uses_temporary_directory(self):
        torch.manual_seed(7)
        model = UltimateNet(channels=8, blocks=1, value_hidden=16)
        report = {
            "games": 4,
            "wins": 3,
            "losses": 1,
            "draws": 0,
            "score_rate": 0.75,
            "score_rate_ci95": [0.5, 1.0],
            "protocol": {"simulations_per_move_each": 2, "dirichlet_noise": False},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidate.pth"
            champion = root / "champion.pth"
            torch.save(
                {
                    "format_version": 1,
                    "model_config": dict(model.config),
                    "state_dict": model.state_dict(),
                    "iteration": 12,
                    "optimizer_state": {"large": True},
                },
                candidate,
            )
            champion.write_bytes(b"old champion")

            export_champion(candidate, champion, report)
            payload = torch.load(champion, map_location="cpu", weights_only=False)

            self.assertEqual(payload["iteration"], 12)
            self.assertIn("league_result", payload)
            self.assertNotIn("optimizer_state", payload)
            self.assertFalse(list(root.glob("*.tmp")))

    def test_invalid_candidate_does_not_replace_champion(self):
        report = {
            "games": 2,
            "wins": 2,
            "losses": 0,
            "draws": 0,
            "score_rate": 1.0,
            "score_rate_ci95": [1.0, 1.0],
            "protocol": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "bad.pth"
            champion = root / "champion.pth"
            torch.save({"wrong": True}, candidate)
            champion.write_bytes(b"keep me")
            before = champion.read_bytes()
            with self.assertRaises(ValueError):
                export_champion(candidate, champion, report)
            self.assertEqual(champion.read_bytes(), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
