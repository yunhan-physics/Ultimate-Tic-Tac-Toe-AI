# -*- coding: utf-8 -*-
"""用固定基线和轮换先后手比赛评估检查点棋力。"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np
import torch

from baseline import GameResult, Policy, heuristic_action, play_game, random_action
from game import EMPTY, P1, legal_moves
from mcts import search
from model import UltimateNet, count_parameters
from self_play import select_action


def load_model(path: str | Path) -> tuple[UltimateNet, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = UltimateNet(**payload["model_config"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def make_mcts_policy(
    model: UltimateNet,
    simulations: int,
    c_puct: float,
) -> Policy:
    """每局调用一次以获得该局独享的预测缓存。"""
    cache = {}

    def policy(state, rng=None):
        rng = rng if rng is not None else np.random.default_rng()
        visits, _ = search(
            state,
            model,
            device="cpu",
            simulations=simulations,
            c_puct=c_puct,
            add_noise=False,
            cache=cache,
            rng=rng,
        )
        return select_action(visits, legal_moves(state), 0.0, rng)

    return policy


def _score_confidence_interval(outcomes: list[float]) -> tuple[float, float]:
    """给 1/0.5/0 得分计算 Wilson 风格的保守近似 95% 区间。"""
    if not outcomes:
        return 0.0, 0.0
    if len(outcomes) == 1:
        value = outcomes[0] if outcomes else 0.0
        return value, value
    sample_size = len(outcomes)
    mean = float(np.mean(outcomes))
    z = 1.96
    denominator = 1.0 + z * z / sample_size
    center = (mean + z * z / (2 * sample_size)) / denominator
    margin = z * np.sqrt(
        mean * (1.0 - mean) / sample_size + z * z / (4 * sample_size**2)
    ) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _side_report(outcomes: list[float]) -> dict[str, int | float]:
    return {
        "games": len(outcomes),
        "wins": outcomes.count(1.0),
        "losses": outcomes.count(0.0),
        "draws": outcomes.count(0.5),
        "score_rate": float(np.mean(outcomes)) if outcomes else 0.0,
    }


def evaluate_against(
    model: UltimateNet,
    opponent: Policy,
    games: int,
    simulations: int,
    workers: int = 4,
    c_puct: float = 1.5,
    seed: int = 0,
) -> dict[str, int | float | list[float]]:
    """AI 与固定对手逐局轮换先后手，所有统计均从 AI 视角计算。"""
    if games <= 0:
        raise ValueError("games 必须为正整数")
    seed_rng = np.random.default_rng(seed)
    seeds = [int(value) for value in seed_rng.integers(0, 2**32 - 1, size=games)]
    model.eval()

    def run(item: tuple[int, int]) -> tuple[GameResult, bool]:
        index, game_seed = item
        ai_is_p1 = index % 2 == 0
        ai_policy = make_mcts_policy(model, simulations, c_puct)
        rng = np.random.default_rng(game_seed)
        result = play_game(
            ai_policy if ai_is_p1 else opponent,
            opponent if ai_is_p1 else ai_policy,
            rng,
        )
        return result, ai_is_p1

    items = list(enumerate(seeds))
    if workers <= 1:
        results = [run(item) for item in items]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(run, items))

    wins = losses = draws = total_moves = 0
    score_margins = []
    outcomes = []
    p1_outcomes = []
    p2_outcomes = []
    for result, ai_is_p1 in results:
        total_moves += result.moves
        ai_score = result.p1_score if ai_is_p1 else result.p2_score
        opponent_score = result.p2_score if ai_is_p1 else result.p1_score
        score_margins.append(ai_score - opponent_score)
        if result.winner == EMPTY:
            draws += 1
            outcome = 0.5
        elif (result.winner == P1) == ai_is_p1:
            wins += 1
            outcome = 1.0
        else:
            losses += 1
            outcome = 0.0
        outcomes.append(outcome)
        (p1_outcomes if ai_is_p1 else p2_outcomes).append(outcome)

    lower, upper = _score_confidence_interval(outcomes)
    return {
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "score_rate": float(np.mean(outcomes)),
        "score_rate_ci95": [lower, upper],
        "as_first_player": _side_report(p1_outcomes),
        "as_second_player": _side_report(p2_outcomes),
        "average_small_board_margin": float(np.mean(score_margins)),
        "average_moves": total_moves / games,
    }


def evaluate_suite(
    model: UltimateNet,
    games: int,
    simulations: int,
    workers: int,
    c_puct: float,
    seed: int,
) -> dict:
    random_report = evaluate_against(
        model, random_action, games, simulations, workers, c_puct, seed
    )
    heuristic_report = evaluate_against(
        model, heuristic_action, games, simulations, workers, c_puct, seed + 1
    )
    # 初版晋级门槛不是“终极棋力”：要求稳定碾压随机，并已能对强手写基线
    # 取得一定得分。后续版本应改为与当前冠军模型直接比赛。
    passed = (
        random_report["score_rate"] >= 0.80
        and heuristic_report["score_rate"] >= 0.25
    )
    return {
        "random": random_report,
        "heuristic": heuristic_report,
        "thresholds": {"random_score_rate": 0.80, "heuristic_score_rate": 0.25},
        "passed_initial_gate": passed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估终极井字棋模型")
    parser.add_argument("checkpoint")
    parser.add_argument("--games", type=int, default=40, help="每种对手的对局数")
    parser.add_argument("--simulations", type=int, default=96)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--output", default="checkpoints/evaluation.json")
    return parser.parse_args()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    torch.set_num_threads(args.torch_threads)
    model, checkpoint = load_model(args.checkpoint)
    report = evaluate_suite(
        model,
        games=args.games,
        simulations=args.simulations,
        workers=args.workers,
        c_puct=args.c_puct,
        seed=args.seed,
    )
    report["checkpoint"] = str(Path(args.checkpoint))
    report["checkpoint_iteration"] = int(checkpoint.get("iteration", -1))
    report["model_parameters"] = count_parameters(model)
    report["simulations"] = args.simulations
    print(json.dumps(report, ensure_ascii=False, indent=2))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
