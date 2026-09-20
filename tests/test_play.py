# -*- coding: utf-8 -*-
"""人机对战入口的快速测试。"""

import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from baseline import random_action
from game import DRAW, OPEN, P1, P2, GameState, apply_move, legal_moves, new_game
from model import UltimateNet
from game_record import GameRecorder, load_record, replay_record
from play import (
    DEFAULT_CHECKPOINT,
    choose_ai_action,
    load_ai_model,
    parse_args,
    parse_move,
    parse_side,
    play_human_game,
    play_demo,
    prompt_human_move,
    render_state,
    UserQuit,
)


class PlayTests(unittest.TestCase):
    def test_parse_global_coordinates_and_action_number(self):
        self.assertEqual(parse_move("1 1"), 0)
        self.assertEqual(parse_move("9，9"), 80)
        self.assertEqual(parse_move("3/7"), 24)
        self.assertEqual(parse_move("#40"), 40)
        self.assertEqual(parse_move("40"), 40)

    def test_parse_move_rejects_bad_or_illegal_input(self):
        with self.assertRaises(ValueError):
            parse_move("0 3")
        with self.assertRaises(ValueError):
            parse_move("hello")
        state, _, _ = apply_move(new_game(), 0)
        self.assertNotIn(80, legal_moves(state))
        with self.assertRaisesRegex(ValueError, "当前不能落子"):
            parse_move("9 9", state)

    def test_parse_side_accepts_chinese_and_english(self):
        self.assertEqual(parse_side("先手"), P1)
        self.assertEqual(parse_side("FIRST"), P1)
        self.assertEqual(parse_side("后手"), P2)
        self.assertEqual(parse_side("o"), P2)

    def test_render_state_shows_board_status_target_and_score(self):
        state = GameState(
            board=(0,) * 81,
            small_boards=(P1, P2, DRAW, OPEN, OPEN, OPEN, OPEN, OPEN, OPEN),
            to_play=P2,
            next_board=4,
        )
        rendered = render_state(state)
        self.assertIn("列  1 2 3", rendered)
        self.assertIn("X O =", rendered)
        self.assertIn("比分：X 1 : 1 O", rendered)
        self.assertIn("强制区域：小棋盘 5", rendered)

    def test_prompt_retries_after_invalid_input(self):
        answers = iter(["不对", "9 9"])
        messages: list[str] = []
        move = prompt_human_move(new_game(), lambda _: next(answers), messages.append)
        self.assertEqual(move, 80)
        self.assertTrue(any("输入无效" in message for message in messages))

    def test_ai_action_is_legal(self):
        torch.manual_seed(31)
        model = UltimateNet(channels=8, blocks=1, value_hidden=16)
        state = new_game()
        move = choose_ai_action(model, state, 2, np.random.default_rng(32))
        self.assertIn(move, legal_moves(state))

    def test_default_checkpoint_loads(self):
        model, payload = load_ai_model(DEFAULT_CHECKPOINT)
        self.assertFalse(model.training)
        self.assertIn("state_dict", payload)
        self.assertEqual(model.config, payload["model_config"])

    def test_demo_is_noninteractive_and_finishes(self):
        messages: list[str] = []

        def random_ai(_model, state, _simulations, rng, **_kwargs):
            return random_action(state, rng)

        with patch("play.choose_ai_action", side_effect=random_ai):
            final_state, winner = play_demo(
                object(),
                "self",
                1,
                np.random.default_rng(33),
                output_fn=messages.append,
            )
        self.assertNotIn(winner, (None,))
        self.assertFalse(legal_moves(final_state))
        self.assertTrue(any("演示模式" in message for message in messages))
        self.assertTrue(any("最终比分" in message for message in messages))

    def test_cli_arguments(self):
        args = parse_args([
            "--side", "后手", "--simulations", "7", "--seed", "9",
            "--demo", "self", "--no-record",
        ])
        self.assertEqual(args.side, P2)
        self.assertEqual(args.simulations, 7)
        self.assertEqual(args.seed, 9)
        self.assertEqual(args.demo, "self")
        self.assertTrue(args.no_record)

    def test_human_game_integration_saves_replayable_record(self):
        with TemporaryDirectory() as temporary:
            recorder = GameRecorder(
                Path(temporary),
                human_player=P1,
                checkpoint="checkpoints/best.pth",
                model_iteration=25,
                simulations=1,
                c_puct=1.5,
                seed=34,
            )
            messages: list[str] = []

            def first_legal_human(state, *_args, **_kwargs):
                return legal_moves(state)[0]

            def first_legal_ai(_model, state, _simulations, _rng, **_kwargs):
                return legal_moves(state)[0]

            with (
                patch("play.prompt_human_move", side_effect=first_legal_human),
                patch("play.choose_ai_action", side_effect=first_legal_ai),
            ):
                final_state, _ = play_human_game(
                    object(),
                    P1,
                    1,
                    np.random.default_rng(34),
                    output_fn=messages.append,
                    recorder=recorder,
                )

            self.assertIsNotNone(recorder.saved_path)
            data = load_record(recorder.saved_path)
            self.assertEqual(data["status"], "completed")
            self.assertEqual(replay_record(recorder.saved_path), final_state)
            self.assertTrue(any("棋谱已保存" in message for message in messages))

    def test_quitting_saves_aborted_record(self):
        with TemporaryDirectory() as temporary:
            recorder = GameRecorder(
                Path(temporary),
                human_player=P1,
                checkpoint="checkpoints/best.pth",
                model_iteration=25,
                simulations=1,
                c_puct=1.5,
                seed=35,
            )
            with patch("play.prompt_human_move", side_effect=UserQuit):
                with self.assertRaises(UserQuit):
                    play_human_game(
                        object(),
                        P1,
                        1,
                        np.random.default_rng(35),
                        output_fn=lambda _message: None,
                        recorder=recorder,
                    )
            data = load_record(recorder.saved_path)
            self.assertEqual(data["status"], "aborted")
            self.assertEqual(data["reason"], "user_quit")


if __name__ == "__main__":
    unittest.main(verbosity=2)
