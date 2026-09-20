# -*- coding: utf-8 -*-
"""无需神经网络的基线对手与自动比赛工具。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from game import (
    ANY_BOARD,
    DRAW,
    EMPTY,
    OPEN,
    P1,
    P2,
    WIN_LINES,
    GameState,
    action_to_parts,
    apply_move,
    legal_moves,
    new_game,
    opponent,
    score,
    small_board_actions,
)


Policy = Callable[[GameState, np.random.Generator | None], int]


@dataclass(frozen=True, slots=True)
class GameResult:
    winner: int
    moves: int
    p1_score: int
    p2_score: int


def _rng_or_default(rng: np.random.Generator | None) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()


def random_action(state: GameState, rng: np.random.Generator | None = None) -> int:
    """从全部合法动作中均匀随机选择一步。"""
    moves = legal_moves(state)
    if not moves:
        raise ValueError("终局状态没有可选动作")
    return int(_rng_or_default(rng).choice(moves))


def _would_claim(state: GameState, move: int, player: int) -> bool:
    """判断 player 落在 move 后是否立刻占领所在小棋盘。"""
    small_board, cell = action_to_parts(move)
    if state.small_boards[small_board] != OPEN or state.board[move] != EMPTY:
        return False
    values = [state.board[action] for action in small_board_actions(small_board)]
    values[cell] = player
    return any(values[a] == values[b] == values[c] == player for a, b, c in WIN_LINES)


def _heuristic_score(state: GameState, move: int) -> float:
    """给一步棋打分；只作为训练和评估的固定参照，不追求完美。"""
    player = state.to_play
    other = opponent(player)
    small_board, cell = action_to_parts(move)
    captures = _would_claim(state, move, player)
    blocks = _would_claim(state, move, other)
    child, done, winner = apply_move(state, move)

    if done:
        if winner == player:
            return 1_000_000.0
        if winner == other:
            return -1_000_000.0
        return 50_000.0

    value = 0.0
    if captures:
        value += 10_000.0
    if blocks:
        value += 3_000.0
    if state.small_boards[small_board] == OPEN and child.small_boards[small_board] == DRAW:
        # 未能占领但至少不给对手留下这个小棋盘。
        value += 150.0

    # 局部中心/角通常参与更多三连。
    if cell == 4:
        value += 45.0
    elif cell in (0, 2, 6, 8):
        value += 25.0
    else:
        value += 10.0

    if child.next_board == ANY_BOARD:
        # 给对手自由选择通常不利。
        value -= 120.0
    else:
        replies = legal_moves(child)
        immediate_losses = sum(_would_claim(child, reply, other) for reply in replies)
        value -= immediate_losses * 2_000.0
        # 在没有直接威胁时，稍微偏好把对手送到选择较少的棋盘。
        value += max(0, 9 - len(replies)) * 8.0

    # 分数领先时，推动小棋盘封闭略有价值；落后时则相反。
    before_p1, before_p2 = score(state)
    after_p1, after_p2 = score(child)
    before_margin = before_p1 - before_p2 if player == P1 else before_p2 - before_p1
    after_margin = after_p1 - after_p2 if player == P1 else after_p2 - after_p1
    value += (after_margin - before_margin) * 500.0
    return value


def heuristic_action(state: GameState, rng: np.random.Generator | None = None) -> int:
    """优先占领、封堵并避免把对手送入可立即占领的小棋盘。"""
    probabilities = heuristic_policy(state)
    return int(_rng_or_default(rng).choice(np.arange(81), p=probabilities))


def heuristic_policy(state: GameState) -> np.ndarray:
    """返回启发式最佳动作上的均匀分布，供课程学习作为软标签。"""
    moves = legal_moves(state)
    if not moves:
        raise ValueError("终局状态没有可选动作")

    scored = [(move, _heuristic_score(state, move)) for move in moves]
    best_score = max(value for _, value in scored)
    best_moves = [move for move, value in scored if value == best_score]
    probabilities = np.zeros(81, dtype=np.float32)
    probabilities[best_moves] = 1.0 / len(best_moves)
    return probabilities


def play_game(
    p1_policy: Policy,
    p2_policy: Policy,
    rng: np.random.Generator | None = None,
) -> GameResult:
    """让两个策略完整对局一次。"""
    rng = _rng_or_default(rng)
    state = new_game()
    moves_made = 0
    while True:
        moves = legal_moves(state)
        if not moves:
            p1_score, p2_score = score(state)
            winner = P1 if p1_score > p2_score else P2 if p2_score > p1_score else EMPTY
            return GameResult(winner, moves_made, p1_score, p2_score)

        policy = p1_policy if state.to_play == P1 else p2_policy
        move = policy(state, rng)
        if move not in moves:
            raise ValueError(f"策略返回了非法动作 {move}")
        state, done, winner = apply_move(state, move)
        moves_made += 1
        if done:
            p1_score, p2_score = score(state)
            return GameResult(winner, moves_made, p1_score, p2_score)


def play_match(
    policy_a: Policy,
    policy_b: Policy,
    games: int = 100,
    seed: int = 0,
) -> dict[str, int | float]:
    """轮换先后手比赛，返回从策略 A 视角统计的胜、负、平。"""
    if games <= 0:
        raise ValueError("games 必须为正整数")
    rng = np.random.default_rng(seed)
    wins = losses = draws = total_moves = 0
    for game_index in range(games):
        a_is_p1 = game_index % 2 == 0
        result = play_game(
            policy_a if a_is_p1 else policy_b,
            policy_b if a_is_p1 else policy_a,
            rng,
        )
        total_moves += result.moves
        if result.winner == EMPTY:
            draws += 1
        elif (result.winner == P1) == a_is_p1:
            wins += 1
        else:
            losses += 1
    return {
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "score_rate": (wins + 0.5 * draws) / games,
        "average_moves": total_moves / games,
    }
