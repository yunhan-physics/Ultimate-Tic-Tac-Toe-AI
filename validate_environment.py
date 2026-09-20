# -*- coding: utf-8 -*-
"""用大量随机对局检查规则引擎的不变量。

这不是棋力评估，而是训练前的环境验收：一旦规则引擎把非法局面送进
自对弈，神经网络会认真学习错误规则，所以这里宁可尽早失败。
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from game import (
    ANY_BOARD,
    EMPTY,
    OPEN,
    P1,
    P2,
    action_to_parts,
    apply_move,
    legal_moves,
    new_game,
    score,
    terminal_info,
)


def validate_random_games(games: int = 1000, seed: int = 20260916) -> dict[str, float]:
    """随机下完若干局；任何状态不变量被破坏时立即抛出 AssertionError。"""
    if games <= 0:
        raise ValueError("games 必须为正整数")

    rng = np.random.default_rng(seed)
    results = {P1: 0, P2: 0, EMPTY: 0}
    total_moves = 0
    shortest = 81
    longest = 0

    for _ in range(games):
        state = new_game()
        moves_made = 0

        while True:
            done, winner = terminal_info(state)
            if done:
                p1_score, p2_score = score(state)
                expected = P1 if p1_score > p2_score else P2 if p2_score > p1_score else EMPTY
                assert winner == expected
                assert all(status != OPEN for status in state.small_boards)
                assert legal_moves(state) == []
                results[winner] += 1
                break

            moves = legal_moves(state)
            assert moves, "非终局状态必须至少有一个合法动作"
            if state.next_board != ANY_BOARD and state.small_boards[state.next_board] == OPEN:
                assert all(action_to_parts(move)[0] == state.next_board for move in moves)
            assert all(state.board[move] == EMPTY for move in moves)
            assert all(state.small_boards[action_to_parts(move)[0]] == OPEN for move in moves)

            move = int(rng.choice(moves))
            old_board = state.board
            old_status = state.small_boards
            old_player = state.to_play
            state, _, _ = apply_move(state, move)
            moves_made += 1

            assert sum(value != EMPTY for value in state.board) == moves_made
            assert state.board[move] == old_player
            assert sum(a != b for a, b in zip(old_board, state.board)) == 1
            assert state.to_play != old_player
            assert all(
                before == after or before == OPEN
                for before, after in zip(old_status, state.small_boards)
            ), "封闭的小棋盘状态不得再次变化"
            assert moves_made <= 81

        total_moves += moves_made
        shortest = min(shortest, moves_made)
        longest = max(longest, moves_made)

    return {
        "games": games,
        "p1_wins": results[P1],
        "p2_wins": results[P2],
        "draws": results[EMPTY],
        "average_moves": total_moves / games,
        "shortest_game": shortest,
        "longest_game": longest,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="随机对局压力测试终极井字棋规则")
    parser.add_argument("--games", type=int, default=1000, help="随机对局数量")
    parser.add_argument("--seed", type=int, default=20260916, help="随机种子")
    return parser.parse_args()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    report = validate_random_games(args.games, args.seed)
    print("规则环境验收通过：")
    for name, value in report.items():
        print(f"  {name}: {value}")
