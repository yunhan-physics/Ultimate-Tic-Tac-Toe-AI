# -*- coding: utf-8 -*-
"""终极井字棋规则引擎的回归测试。"""

import unittest

import numpy as np

from game import (
    ACTION_SIZE,
    ANY_BOARD,
    DRAW,
    EMPTY,
    OPEN,
    P1,
    P2,
    GameState,
    action_to_parts,
    apply_move,
    encode_state,
    legal_moves,
    new_game,
    parts_to_action,
    score,
    small_board_actions,
    terminal_info,
)


def make_state(*, pieces=None, statuses=None, to_play=P1, next_board=ANY_BOARD):
    """用少量差异构造测试局面。"""
    board = [EMPTY] * ACTION_SIZE
    for small_board, cell, player in pieces or []:
        board[parts_to_action(small_board, cell)] = player
    return GameState(
        board=tuple(board),
        small_boards=tuple(statuses or [OPEN] * 9),
        to_play=to_play,
        next_board=next_board,
    )


class CoordinateTests(unittest.TestCase):
    def test_all_actions_round_trip(self):
        reconstructed = {
            parts_to_action(*action_to_parts(action))
            for action in range(ACTION_SIZE)
        }
        self.assertEqual(reconstructed, set(range(ACTION_SIZE)))

    def test_known_coordinate(self):
        # 中央小棋盘的右下角位于整个 9x9 棋盘的 (5, 5)。
        action = parts_to_action(4, 8)
        self.assertEqual(action, 5 * 9 + 5)
        self.assertEqual(action_to_parts(action), (4, 8))


class LegalMoveTests(unittest.TestCase):
    def test_first_move_can_use_all_81_cells(self):
        self.assertEqual(legal_moves(new_game()), list(range(ACTION_SIZE)))

    def test_move_routes_opponent_to_matching_small_board(self):
        state, done, winner = apply_move(new_game(), parts_to_action(4, 8))
        self.assertFalse(done)
        self.assertEqual(winner, EMPTY)
        self.assertEqual(state.next_board, 8)
        self.assertEqual(legal_moves(state), sorted(small_board_actions(8)))

    def test_closed_target_gives_free_choice(self):
        statuses = [P1] + [OPEN] * 8
        state = make_state(statuses=statuses, next_board=0)
        moves = legal_moves(state)
        self.assertEqual(len(moves), 72)
        self.assertTrue(all(action_to_parts(move)[0] != 0 for move in moves))

    def test_occupied_and_closed_cells_are_illegal(self):
        statuses = [OPEN, P2] + [OPEN] * 7
        occupied = (0, 4, P1)
        state = make_state(pieces=[occupied], statuses=statuses)
        moves = legal_moves(state)
        self.assertNotIn(parts_to_action(0, 4), moves)
        self.assertTrue(all(action_to_parts(move)[0] != 1 for move in moves))

    def test_illegal_move_raises_value_error(self):
        state = make_state(pieces=[(0, 0, P1)], next_board=0)
        with self.assertRaises(ValueError):
            apply_move(state, parts_to_action(0, 0))
        with self.assertRaises(ValueError):
            apply_move(state, parts_to_action(1, 0))


class SmallBoardTests(unittest.TestCase):
    def test_three_in_a_row_claims_and_closes_small_board(self):
        state = make_state(
            pieces=[(4, 0, P1), (4, 1, P1)],
            to_play=P1,
            next_board=4,
        )
        state, done, _ = apply_move(state, parts_to_action(4, 2))
        self.assertFalse(done)
        self.assertEqual(state.small_boards[4], P1)
        self.assertTrue(all(action_to_parts(move)[0] != 4 for move in legal_moves(state)))

    def test_closed_routing_target_becomes_any_board(self):
        statuses = [OPEN, OPEN, P2] + [OPEN] * 6
        state = make_state(
            pieces=[(4, 0, P1), (4, 1, P1)],
            statuses=statuses,
            to_play=P1,
            next_board=4,
        )
        state, _, _ = apply_move(state, parts_to_action(4, 2))
        self.assertEqual(state.next_board, ANY_BOARD)
        target_boards = {action_to_parts(move)[0] for move in legal_moves(state)}
        self.assertEqual(target_boards, {0, 1, 3, 5, 6, 7, 8})

    def test_full_small_board_without_line_is_drawn_and_closed(self):
        # 最后在格 7 落 X，得到一个没有三连的满盘。
        pieces = [
            (0, 0, P1), (0, 1, P2), (0, 2, P1),
            (0, 3, P1), (0, 4, P1), (0, 5, P2),
            (0, 6, P2),                 (0, 8, P2),
        ]
        state = make_state(pieces=pieces, to_play=P1, next_board=0)
        state, done, _ = apply_move(state, parts_to_action(0, 7))
        self.assertFalse(done)
        self.assertEqual(state.small_boards[0], DRAW)
        self.assertTrue(all(action_to_parts(move)[0] != 0 for move in legal_moves(state)))


class TerminalTests(unittest.TestCase):
    def test_big_board_line_does_not_end_score_mode_game(self):
        state = make_state(statuses=[P1, P1, P1] + [OPEN] * 6)
        self.assertEqual(terminal_info(state), (False, EMPTY))

    def test_all_closed_boards_are_scored(self):
        state = make_state(statuses=[P1] * 5 + [P2] * 3 + [DRAW])
        self.assertEqual(score(state), (5, 3))
        self.assertEqual(terminal_info(state), (True, P1))
        self.assertEqual(legal_moves(state), [])

    def test_equal_claim_count_is_a_draw(self):
        state = make_state(statuses=[P1] * 4 + [P2] * 4 + [DRAW])
        self.assertEqual(terminal_info(state), (True, EMPTY))


class EncodingTests(unittest.TestCase):
    def test_encoding_shape_type_perspective_and_legal_plane(self):
        statuses = [P1, P2, DRAW] + [OPEN] * 6
        state = make_state(
            pieces=[(3, 0, P1), (3, 1, P2)],
            statuses=statuses,
            to_play=P2,
            next_board=3,
        )
        encoded = encode_state(state)
        self.assertEqual(encoded.shape, (6, 9, 9))
        self.assertEqual(encoded.dtype, np.float32)
        self.assertEqual(encoded[0].sum(), 1.0)  # 当前方 P2 的棋子
        self.assertEqual(encoded[1].sum(), 1.0)  # 对手 P1 的棋子
        self.assertEqual(encoded[2].sum(), 9.0)  # P2 占领的小棋盘
        self.assertEqual(encoded[3].sum(), 9.0)  # P1 占领的小棋盘
        self.assertEqual(encoded[4].sum(), 9.0)  # 和棋小棋盘
        self.assertEqual(encoded[5].sum(), 7.0)  # 强制棋盘内还剩七个空格


if __name__ == "__main__":
    unittest.main(verbosity=2)
