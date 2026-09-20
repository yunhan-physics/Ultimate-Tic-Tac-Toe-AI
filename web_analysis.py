"""Position and move estimates for the browser's coaching mode.

The value head learns an expected game result in [-1, 1]. Its conversion to
percentages is an expected score (a draw counts as half), not a calibrated win
probability. MCTS visit frequencies are never used as move win percentages.
"""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any

import torch

from game import EMPTY, P1, P2, GameState, apply_move, legal_moves, terminal_info
from mcts import search


def _positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _human_rate(
    state: GameState,
    model: torch.nn.Module,
    human_player: int,
    *,
    simulations: int,
    c_puct: float,
    device: str | torch.device,
    cache: dict,
) -> float:
    done, winner = terminal_info(state)
    if done:
        if winner == EMPTY:
            return 50.0
        return 100.0 if winner == human_player else 0.0

    _, value = search(
        state,
        model,
        device=device,
        simulations=simulations,
        c_puct=c_puct,
        add_noise=False,
        cache=cache,
    )
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("The model returned a non-finite position estimate.")
    human_value = value if state.to_play == human_player else -value
    return min(100.0, max(0.0, (human_value + 1.0) * 50.0))


def analyze_position(
    state: GameState,
    model: torch.nn.Module,
    human_player: int,
    *,
    simulations: int = 48,
    action_simulations: int = 12,
    c_puct: float = 1.5,
    device: str | torch.device = "cpu",
    include_moves: bool = True,
) -> dict[str, Any]:
    """Estimate a position and all legal human moves using equal search budgets.

    All rates and deltas are from the human's perspective, regardless of who
    moves first. Coordinates are global 1-based rows and columns; actions keep
    the rule engine's 0-based encoding. ``delta_pp`` is a percentage-point
    change from the current position estimate. ``moves`` is action ordered,
    while ``top_moves`` is ranked best first with stable action-order ties.

    On the AI's turn or when ``include_moves`` is false, only the current
    position is evaluated. Prediction caching is limited to this call so a
    long-running web server does not retain trees or old model predictions.
    """
    if not isinstance(state, GameState):
        raise ValueError("state must be a GameState.")
    if (
        isinstance(human_player, bool)
        or not isinstance(human_player, Integral)
        or human_player not in (P1, P2)
    ):
        raise ValueError("human_player must be P1 or P2.")
    _positive_integer("simulations", simulations)
    _positive_integer("action_simulations", action_simulations)
    if (
        isinstance(c_puct, bool)
        or not isinstance(c_puct, Real)
        or not math.isfinite(c_puct)
        or c_puct <= 0
    ):
        raise ValueError("c_puct must be a finite positive number.")
    if not isinstance(include_moves, bool):
        raise ValueError("include_moves must be a boolean.")

    cache: dict = {}
    human_rate = _human_rate(
        state,
        model,
        human_player,
        simulations=simulations,
        c_puct=c_puct,
        device=device,
        cache=cache,
    )
    moves: list[dict[str, Any]] = []
    if include_moves and state.to_play == human_player:
        for action in legal_moves(state):
            child, _, _ = apply_move(state, action)
            child_rate = _human_rate(
                child,
                model,
                human_player,
                simulations=action_simulations,
                c_puct=c_puct,
                device=device,
                cache=cache,
            )
            row, column = divmod(action, 9)
            moves.append(
                {
                    "action": action,
                    "row": row + 1,
                    "column": column + 1,
                    "human_rate": child_rate,
                    "ai_rate": 100.0 - child_rate,
                    "delta_pp": child_rate - human_rate,
                }
            )

    top_moves = sorted(moves, key=lambda move: (-move["human_rate"], move["action"]))[:3]
    for rank, move in enumerate(top_moves, 1):
        move["rank"] = rank
    return {
        "human_rate": human_rate,
        "ai_rate": 100.0 - human_rate,
        "top_moves": top_moves,
        "moves": moves,
        "metric": "expected_score",
        "method": "MCTS value estimates; equal search budget per legal move; draws count as half.",
    }
