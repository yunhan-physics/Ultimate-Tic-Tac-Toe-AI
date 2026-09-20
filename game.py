# -*- coding: utf-8 -*-
"""终极井字棋的纯规则引擎。

本文件不依赖 PyTorch，只定义游戏状态、合法动作、落子、终局判断和
神经网络输入编码。训练、搜索和界面都必须以这里的规则为唯一依据。

棋盘采用真实的 9x9 行优先坐标，动作编号为 ``row * 9 + col``（0~80）。
每个动作同时属于一个大棋盘位置（小棋盘编号）和一个小棋盘内位置：

    小棋盘编号                  小棋盘内格子编号
    0 | 1 | 2                   0 | 1 | 2
    --+---+--                   --+---+--
    3 | 4 | 5                   3 | 4 | 5
    --+---+--                   --+---+--
    6 | 7 | 8                   6 | 7 | 8

项目采用规则文件指定的“胜利方式二”：不因大棋盘三连而提前结束；当九个
小棋盘全部封闭（被占领或填满成和棋）后，占领数量多的一方获胜。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EMPTY, P1, P2 = 0, 1, 2
OPEN, DRAW = 0, 3
ANY_BOARD = -1
BOARD_SIZE = 9
ACTION_SIZE = BOARD_SIZE * BOARD_SIZE

WIN_LINES = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
)


@dataclass(frozen=True, slots=True)
class GameState:
    """一个可哈希、不可变的游戏状态，方便后续用作 MCTS 缓存键。"""

    board: tuple[int, ...]
    small_boards: tuple[int, ...]
    to_play: int
    next_board: int

    def __post_init__(self) -> None:
        if len(self.board) != ACTION_SIZE:
            raise ValueError("board 必须恰好包含 81 个格子")
        if len(self.small_boards) != 9:
            raise ValueError("small_boards 必须恰好包含 9 个状态")
        if any(value not in (EMPTY, P1, P2) for value in self.board):
            raise ValueError("board 的格子只能是 0、1 或 2")
        if any(value not in (OPEN, P1, P2, DRAW) for value in self.small_boards):
            raise ValueError("小棋盘状态只能是 OPEN、P1、P2 或 DRAW")
        if self.to_play not in (P1, P2):
            raise ValueError("to_play 必须是 P1 或 P2")
        if self.next_board not in range(9) and self.next_board != ANY_BOARD:
            raise ValueError("next_board 必须是 0~8 或 ANY_BOARD")


def opponent(player: int) -> int:
    """返回另一位玩家。"""
    if player not in (P1, P2):
        raise ValueError("player 必须是 P1 或 P2")
    return 3 - player


def new_game() -> GameState:
    """创建空棋盘；玩家 1 先手，第一步可落在任意位置。"""
    return GameState(
        board=(EMPTY,) * ACTION_SIZE,
        small_boards=(OPEN,) * 9,
        to_play=P1,
        next_board=ANY_BOARD,
    )


def action_to_parts(action: int) -> tuple[int, int]:
    """把 0~80 的动作转换为 ``(小棋盘编号, 小棋盘内格子编号)``。"""
    if not isinstance(action, (int, np.integer)) or not 0 <= int(action) < ACTION_SIZE:
        raise ValueError("action 必须是 0~80 的整数")
    row, col = divmod(int(action), BOARD_SIZE)
    small_board = (row // 3) * 3 + col // 3
    cell = (row % 3) * 3 + col % 3
    return small_board, cell


def parts_to_action(small_board: int, cell: int) -> int:
    """把 ``(小棋盘编号, 小棋盘内格子编号)``转换为 0~80 的动作。"""
    if small_board not in range(9) or cell not in range(9):
        raise ValueError("small_board 和 cell 都必须是 0~8")
    board_row, board_col = divmod(small_board, 3)
    cell_row, cell_col = divmod(cell, 3)
    return (board_row * 3 + cell_row) * BOARD_SIZE + board_col * 3 + cell_col


def small_board_actions(small_board: int) -> tuple[int, ...]:
    """按小棋盘内 0~8 的顺序返回其对应的九个全局动作。"""
    if small_board not in range(9):
        raise ValueError("small_board 必须是 0~8")
    return tuple(parts_to_action(small_board, cell) for cell in range(9))


def _line_winner(values: list[int] | tuple[int, ...]) -> int:
    """检查九格棋盘中的三连；没有赢家时返回 EMPTY。"""
    for a, b, c in WIN_LINES:
        if values[a] != EMPTY and values[a] == values[b] == values[c]:
            return values[a]
    return EMPTY


def terminal_info(state: GameState) -> tuple[bool, int]:
    """返回 ``(是否终局, 赢家)``；赢家为 EMPTY 表示尚未结束或平局。"""
    if any(status == OPEN for status in state.small_boards):
        return False, EMPTY

    p1_score = state.small_boards.count(P1)
    p2_score = state.small_boards.count(P2)
    if p1_score > p2_score:
        return True, P1
    if p2_score > p1_score:
        return True, P2
    return True, EMPTY


def legal_moves(state: GameState) -> list[int]:
    """返回当前全部合法动作，按全局动作编号升序排列。"""
    if terminal_info(state)[0]:
        return []

    target = state.next_board
    if target != ANY_BOARD and state.small_boards[target] == OPEN:
        forced_moves = [
            action for action in small_board_actions(target)
            if state.board[action] == EMPTY
        ]
        # 合法状态下 OPEN 小棋盘一定有空位；这里的回退令函数对手工构造的
        # “已满但尚未标 DRAW”状态也遵守“目标已满即可自由落子”的规则。
        if forced_moves:
            return forced_moves

    return [
        action
        for action, value in enumerate(state.board)
        if value == EMPTY and state.small_boards[action_to_parts(action)[0]] == OPEN
    ]


def apply_move(state: GameState, move: int) -> tuple[GameState, bool, int]:
    """执行一步合法落子，返回 ``(新状态, 是否终局, 赢家)``。"""
    try:
        small_board, cell = action_to_parts(move)
    except ValueError as exc:
        raise ValueError(f"非法动作 {move!r}") from exc

    if move not in legal_moves(state):
        raise ValueError(f"动作 {move} 在当前局面中不合法")

    board = list(state.board)
    board[move] = state.to_play

    small_boards = list(state.small_boards)
    local_actions = small_board_actions(small_board)
    local_values = [board[action] for action in local_actions]
    local_winner = _line_winner(local_values)
    if local_winner != EMPTY:
        small_boards[small_board] = local_winner
    elif all(value != EMPTY for value in local_values):
        small_boards[small_board] = DRAW

    # 这一手在小棋盘内的位置，决定对手被送往哪个小棋盘。如果目标已经
    # 被占领或和棋封闭，则对手下一步可以在任意仍开放的小棋盘落子。
    next_board = cell if small_boards[cell] == OPEN else ANY_BOARD
    next_state = GameState(
        board=tuple(board),
        small_boards=tuple(small_boards),
        to_play=opponent(state.to_play),
        next_board=next_board,
    )
    done, winner = terminal_info(next_state)
    return next_state, done, winner


def score(state: GameState) -> tuple[int, int]:
    """返回玩家 1 和玩家 2 已占领的小棋盘数量。"""
    return state.small_boards.count(P1), state.small_boards.count(P2)


def encode_state(state: GameState) -> np.ndarray:
    """从当前行动方视角编码为 ``(6, 9, 9)`` 的 float32 数组。

    六个通道依次表示：我方棋子、对方棋子、我方占领的小棋盘、对方占领的
    小棋盘、和棋封闭的小棋盘、当前合法落子。小棋盘状态会铺满其 3x3 区域。
    """
    encoded = np.zeros((6, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    other = opponent(state.to_play)

    for action, value in enumerate(state.board):
        row, col = divmod(action, BOARD_SIZE)
        if value == state.to_play:
            encoded[0, row, col] = 1.0
        elif value == other:
            encoded[1, row, col] = 1.0

    for small_board, status in enumerate(state.small_boards):
        board_row, board_col = divmod(small_board, 3)
        rows = slice(board_row * 3, board_row * 3 + 3)
        cols = slice(board_col * 3, board_col * 3 + 3)
        if status == state.to_play:
            encoded[2, rows, cols] = 1.0
        elif status == other:
            encoded[3, rows, cols] = 1.0
        elif status == DRAW:
            encoded[4, rows, cols] = 1.0

    for action in legal_moves(state):
        row, col = divmod(action, BOARD_SIZE)
        encoded[5, row, col] = 1.0
    return encoded


def board_string(state: GameState) -> str:
    """把 9x9 棋盘格式化为适合终端显示的字符串。"""
    symbols = {EMPTY: ".", P1: "X", P2: "O"}
    lines: list[str] = []
    for row in range(BOARD_SIZE):
        chunks = []
        for board_col in range(3):
            start = row * BOARD_SIZE + board_col * 3
            chunks.append(" ".join(symbols[state.board[start + offset]] for offset in range(3)))
        lines.append(" | ".join(chunks))
        if row in (2, 5):
            lines.append("------+-------+------")
    return "\n".join(lines)
