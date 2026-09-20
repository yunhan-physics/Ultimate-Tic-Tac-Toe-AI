"""Coaching analysis: perspectives, terminal outcomes, and move comparison."""

import unittest
from unittest.mock import patch

import numpy as np
import torch

from game import (
    ACTION_SIZE,
    ANY_BOARD,
    DRAW,
    EMPTY,
    OPEN,
    P1,
    P2,
    GameState,
    apply_move,
    legal_moves,
    new_game,
    parts_to_action,
)
from web_analysis import analyze_position


class ZeroNet(torch.nn.Module):
    def forward(self, x):
        return torch.zeros((x.shape[0], ACTION_SIZE)), torch.zeros((x.shape[0], 1))


def last_open_board(statuses, player=P1):
    """A final empty cell closes a drawn board; the last mover need not win."""
    board = [EMPTY] * ACTION_SIZE
    pattern = [P1, P2, P1, P1, P1, P2, P2, EMPTY, P2]
    for cell, occupant in enumerate(pattern):
        board[parts_to_action(8, cell)] = occupant
    return GameState(tuple(board), tuple(statuses) + (OPEN,), player, 8)


class AnalysisTests(unittest.TestCase):
    def test_all_moves_have_equal_budget_and_correct_human_perspective(self):
        state = new_game()
        child_values = {
            apply_move(state, action)[0]: -0.8 + action / 50.0
            for action in legal_moves(state)
        }

        def controlled_search(position, model, **kwargs):
            # Deliberately useless visits: the display must use value instead.
            visits = np.zeros(ACTION_SIZE)
            visits[80] = 1.0
            return visits, 0.2 if position == state else child_values[position]

        with patch("web_analysis.search", side_effect=controlled_search) as search:
            result = analyze_position(state, ZeroNet(), P1, simulations=7, action_simulations=3)

        self.assertAlmostEqual(result["human_rate"], 60.0)
        self.assertEqual(result["human_rate"] + result["ai_rate"], 100.0)
        self.assertEqual(result["metric"], "expected_score")
        self.assertEqual([move["action"] for move in result["moves"]], list(range(81)))
        self.assertEqual([move["action"] for move in result["top_moves"]], [0, 1, 2])
        self.assertEqual([move["rank"] for move in result["top_moves"]], [1, 2, 3])
        self.assertEqual(search.call_count, 82)
        self.assertEqual(search.call_args_list[0].kwargs["simulations"], 7)
        call_cache = search.call_args_list[0].kwargs["cache"]
        for call in search.call_args_list[1:]:
            self.assertEqual(call.kwargs["simulations"], 3)
            self.assertFalse(call.kwargs["add_noise"])
            self.assertIs(call.kwargs["cache"], call_cache)
        for move in result["moves"]:
            self.assertAlmostEqual(move["human_rate"], 90.0 - move["action"])
            self.assertAlmostEqual(move["delta_pp"], move["human_rate"] - 60.0)
            self.assertEqual(move["human_rate"] + move["ai_rate"], 100.0)
            self.assertEqual((move["row"] - 1) * 9 + move["column"] - 1, move["action"])

    def test_second_player_receives_legal_recommendations_with_stable_ties(self):
        state, _, _ = apply_move(new_game(), 40)
        with patch("web_analysis.search", return_value=(np.zeros(ACTION_SIZE), 0.4)):
            result = analyze_position(state, ZeroNet(), P2)
        self.assertAlmostEqual(result["human_rate"], 70.0)
        self.assertEqual([move["action"] for move in result["moves"]], legal_moves(state))
        self.assertEqual([move["action"] for move in result["top_moves"]], legal_moves(state)[:3])
        for move in result["moves"]:
            self.assertAlmostEqual(move["human_rate"], 30.0)
            self.assertAlmostEqual(move["delta_pp"], -40.0)

    def test_ai_turn_only_evaluates_current_position(self):
        with patch("web_analysis.search", return_value=(np.zeros(ACTION_SIZE), 0.6)) as search:
            result = analyze_position(new_game(), ZeroNet(), P2)
        self.assertAlmostEqual(result["human_rate"], 20.0)
        self.assertEqual(result["moves"], [])
        self.assertEqual(result["top_moves"], [])
        self.assertEqual(search.call_count, 1)

    def test_include_moves_false_skips_move_searches(self):
        with patch("web_analysis.search", return_value=(np.zeros(ACTION_SIZE), 0.0)) as search:
            result = analyze_position(new_game(), ZeroNet(), P1, include_moves=False)
        self.assertEqual(result["top_moves"], [])
        self.assertEqual(search.call_count, 1)

    def test_terminal_results_are_exact_without_search(self):
        cases = [
            ((P1,) * 5 + (P2,) * 4, P1, 100.0),
            ((P1,) * 5 + (P2,) * 4, P2, 0.0),
            ((P1,) * 4 + (P2,) * 4 + (DRAW,), P1, 50.0),
        ]
        for statuses, human, expected in cases:
            with self.subTest(human=human, statuses=statuses):
                state = GameState((EMPTY,) * ACTION_SIZE, statuses, P2, ANY_BOARD)
                with patch("web_analysis.search", side_effect=AssertionError("No terminal search")):
                    result = analyze_position(state, ZeroNet(), human)
                self.assertEqual(result["human_rate"], expected)
                self.assertEqual(result["ai_rate"], 100.0 - expected)
                self.assertEqual(result["top_moves"], [])

    def test_real_search_last_mover_loses_in_score_only_game(self):
        state = last_open_board([P1] * 3 + [P2] * 4 + [DRAW])
        result = analyze_position(state, ZeroNet(), P1, simulations=2, action_simulations=2)
        self.assertEqual(result["human_rate"], 0.0)
        self.assertEqual(len(result["top_moves"]), 1)
        only = result["top_moves"][0]
        self.assertEqual(only["action"], parts_to_action(8, 7))
        self.assertEqual(only["human_rate"], 0.0)
        self.assertEqual(only["ai_rate"], 100.0)
        self.assertEqual(only["delta_pp"], 0.0)

    def test_real_search_terminal_draw_is_half_score(self):
        state = last_open_board([P1] * 4 + [P2] * 4)
        result = analyze_position(state, ZeroNet(), P1, simulations=2, action_simulations=2)
        self.assertEqual(result["human_rate"], 50.0)
        self.assertEqual(result["top_moves"][0]["human_rate"], 50.0)

    def test_prediction_cache_does_not_persist_across_calls(self):
        with patch("web_analysis.search", return_value=(np.zeros(ACTION_SIZE), 0.0)) as search:
            analyze_position(new_game(), ZeroNet(), P2)
            analyze_position(new_game(), ZeroNet(), P2)
        self.assertIsNot(search.call_args_list[0].kwargs["cache"], search.call_args_list[1].kwargs["cache"])

    def test_out_of_range_estimates_are_clipped(self):
        for value, expected in [(1.01, 100.0), (-1.01, 0.0)]:
            with self.subTest(value=value):
                with patch("web_analysis.search", return_value=(None, value)):
                    result = analyze_position(new_game(), ZeroNet(), P1, include_moves=False)
                self.assertEqual(result["human_rate"], expected)
                self.assertEqual(result["human_rate"] + result["ai_rate"], 100.0)

    def test_non_finite_estimates_raise_clear_error(self):
        for value in [float("nan"), float("inf"), -float("inf")]:
            with self.subTest(value=value):
                with patch("web_analysis.search", return_value=(None, value)):
                    with self.assertRaisesRegex(ValueError, "non-finite"):
                        analyze_position(new_game(), ZeroNet(), P1, include_moves=False)

    def test_invalid_parameters_are_rejected_before_search(self):
        invalid_kwargs = [
            {"human_player": 0}, {"human_player": True}, {"human_player": 1.0},
            {"simulations": 0}, {"simulations": True}, {"simulations": 1.5},
            {"action_simulations": -1}, {"action_simulations": 1.5},
            {"c_puct": 0}, {"c_puct": float("nan")}, {"c_puct": float("inf")},
            {"c_puct": "1.5"}, {"include_moves": "yes"},
        ]
        for overrides in invalid_kwargs:
            with self.subTest(overrides=overrides):
                kwargs = {"human_player": P1, **overrides}
                with patch("web_analysis.search", side_effect=AssertionError("No invalid search")):
                    with self.assertRaises(ValueError):
                        analyze_position(new_game(), ZeroNet(), **kwargs)
        with self.assertRaises(ValueError):
            analyze_position(None, ZeroNet(), P1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
