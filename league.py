# -*- coding: utf-8 -*-
"""候选模型与当前冠军之间的公平晋级联赛。

每个随机合法开局恰好进行两盘，候选模型分别执 P1 和 P2。双方使用完全
相同的 MCTS 参数、零温度落子并关闭根节点噪声。置信区间按“开局对”进行
配对 bootstrap，避免把共享同一开局的两盘误当成独立样本。
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

from baseline import GameResult, Policy
from evaluate import load_model, make_mcts_policy
from game import EMPTY, P1, P2, GameState, apply_move, legal_moves, new_game, score, terminal_info
from model import count_parameters


@dataclass(frozen=True, slots=True)
class Opening:
    """一个可复现的、尚未终局的合法开局。"""

    seed: int
    moves: tuple[int, ...]
    state: GameState


def generate_openings(
    pairs: int,
    opening_plies: int,
    seed: int,
    max_attempts_per_opening: int = 200,
) -> list[Opening]:
    """生成 ``pairs`` 个随机合法开局；同一 seed 会得到相同结果。"""
    if pairs <= 0:
        raise ValueError("pairs 必须为正整数")
    if not 0 <= opening_plies <= 80:
        raise ValueError("opening_plies 必须位于 0~80")
    if seed < 0:
        raise ValueError("seed 必须为非负整数")
    if max_attempts_per_opening <= 0:
        raise ValueError("max_attempts_per_opening 必须为正整数")

    master_rng = np.random.default_rng(seed)
    openings: list[Opening] = []
    for index in range(pairs):
        for _ in range(max_attempts_per_opening):
            opening_seed = int(master_rng.integers(0, 2**32 - 1))
            rng = np.random.default_rng(opening_seed)
            state = new_game()
            moves: list[int] = []
            completed = True
            for _ in range(opening_plies):
                legal = legal_moves(state)
                if not legal:
                    completed = False
                    break
                move = int(rng.choice(legal))
                state, done, _ = apply_move(state, move)
                moves.append(move)
                if done:
                    completed = False
                    break
            if completed and not terminal_info(state)[0]:
                openings.append(Opening(opening_seed, tuple(moves), state))
                break
        else:
            raise RuntimeError(
                f"无法为第 {index + 1} 个开局生成 {opening_plies} 手后仍未终局的状态；"
                "请减少 --opening-plies"
            )
    return openings


def play_from_state(
    initial_state: GameState,
    p1_policy: Policy,
    p2_policy: Policy,
    rng: np.random.Generator,
) -> GameResult:
    """从给定合法开局继续比赛，返回包含开局手数的完整结果。"""
    if terminal_info(initial_state)[0]:
        raise ValueError("不能从终局状态开始联赛")
    state = initial_state
    moves_made = sum(value != EMPTY for value in state.board)
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


def paired_bootstrap_ci(
    pair_scores: list[float] | np.ndarray,
    samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """对每个开局的两盘平均得分进行非参数 bootstrap。"""
    values = np.asarray(pair_scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("pair_scores 必须是一维非空序列")
    if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("pair_scores 中的值必须是 [0, 1] 内的有限数")
    if samples <= 0:
        raise ValueError("samples 必须为正整数")
    if not 0 < confidence < 1:
        raise ValueError("confidence 必须位于 (0, 1)")
    if seed < 0:
        raise ValueError("seed 必须为非负整数")
    if values.size == 1 or np.all(values == values[0]):
        return float(values.mean()), float(values.mean())

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(means, [tail, 1.0 - tail])
    return float(lower), float(upper)


def _outcome(result: GameResult, candidate_is_p1: bool) -> float:
    if result.winner == EMPTY:
        return 0.5
    return 1.0 if (result.winner == P1) == candidate_is_p1 else 0.0


def _side_report(outcomes: list[float]) -> dict[str, int | float]:
    return {
        "games": len(outcomes),
        "wins": outcomes.count(1.0),
        "losses": outcomes.count(0.0),
        "draws": outcomes.count(0.5),
        "score_rate": float(np.mean(outcomes)) if outcomes else 0.0,
    }


def run_league(
    candidate: torch.nn.Module,
    champion: torch.nn.Module,
    *,
    pairs: int = 32,
    opening_plies: int = 4,
    simulations: int = 96,
    workers: int = 4,
    c_puct: float = 1.5,
    seed: int = 20260916,
    bootstrap_samples: int = 10_000,
) -> dict:
    """进行候选对冠军联赛，所有结果均从候选模型视角统计。"""
    if simulations <= 0:
        raise ValueError("simulations 必须为正整数")
    if workers <= 0:
        raise ValueError("workers 必须为正整数")
    if c_puct <= 0:
        raise ValueError("c_puct 必须为正数")

    openings = generate_openings(pairs, opening_plies, seed)
    game_seed_rng = np.random.default_rng(seed ^ 0xA5A5A5A5)
    game_seeds = [int(value) for value in game_seed_rng.integers(0, 2**32 - 1, size=pairs)]
    candidate.eval()
    champion.eval()

    # 每个元素是一组共享开局和随机种子的交换对局；executor.map 保留顺序。
    def run_pair(item: tuple[int, Opening, int]) -> tuple[GameResult, GameResult]:
        _, opening, game_seed = item

        candidate_p1 = make_mcts_policy(candidate, simulations, c_puct)
        champion_p2 = make_mcts_policy(champion, simulations, c_puct)
        first = play_from_state(
            opening.state,
            candidate_p1,
            champion_p2,
            np.random.default_rng(game_seed),
        )

        champion_p1 = make_mcts_policy(champion, simulations, c_puct)
        candidate_p2 = make_mcts_policy(candidate, simulations, c_puct)
        second = play_from_state(
            opening.state,
            champion_p1,
            candidate_p2,
            np.random.default_rng(game_seed),
        )
        return first, second

    items = [(index, opening, game_seeds[index]) for index, opening in enumerate(openings)]
    if workers == 1:
        results = [run_pair(item) for item in items]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(run_pair, items))

    all_outcomes: list[float] = []
    p1_outcomes: list[float] = []
    p2_outcomes: list[float] = []
    pair_scores: list[float] = []
    score_margins: list[int] = []
    move_counts: list[int] = []
    pair_records: list[dict] = []
    for index, ((first, second), opening) in enumerate(zip(results, openings)):
        first_outcome = _outcome(first, candidate_is_p1=True)
        second_outcome = _outcome(second, candidate_is_p1=False)
        pair_score = (first_outcome + second_outcome) / 2.0
        all_outcomes.extend((first_outcome, second_outcome))
        p1_outcomes.append(first_outcome)
        p2_outcomes.append(second_outcome)
        pair_scores.append(pair_score)
        score_margins.extend(
            (first.p1_score - first.p2_score, second.p2_score - second.p1_score)
        )
        move_counts.extend((first.moves, second.moves))
        pair_records.append(
            {
                "opening_index": index,
                "opening_seed": opening.seed,
                "opening_moves": list(opening.moves),
                "game_seed": game_seeds[index],
                "candidate_as_p1": {
                    "outcome": first_outcome,
                    "winner": first.winner,
                    "p1_score": first.p1_score,
                    "p2_score": first.p2_score,
                    "moves": first.moves,
                },
                "candidate_as_p2": {
                    "outcome": second_outcome,
                    "winner": second.winner,
                    "p1_score": second.p1_score,
                    "p2_score": second.p2_score,
                    "moves": second.moves,
                },
                "pair_score": pair_score,
            }
        )

    lower, upper = paired_bootstrap_ci(
        pair_scores,
        samples=bootstrap_samples,
        confidence=0.95,
        seed=seed ^ 0x5A5A5A5A,
    )
    summary = _side_report(all_outcomes)
    return {
        **summary,
        "opening_pairs": pairs,
        "opening_plies": opening_plies,
        "score_rate_ci95": [lower, upper],
        "ci_method": "paired_percentile_bootstrap_by_opening",
        "bootstrap_samples": bootstrap_samples,
        "candidate_as_p1": _side_report(p1_outcomes),
        "candidate_as_p2": _side_report(p2_outcomes),
        "average_small_board_margin": float(np.mean(score_margins)),
        "average_moves": float(np.mean(move_counts)),
        "protocol": {
            "simulations_per_move_each": simulations,
            "c_puct_each": c_puct,
            "dirichlet_noise": False,
            "temperature": 0.0,
            "swap_sides_for_every_opening": True,
            "seed": seed,
        },
        "pairs": pair_records,
    }


def promotion_decision(
    match_report: dict,
    min_score_rate: float = 0.55,
    min_ci_lower: float = 0.50,
) -> dict[str, bool | float | str]:
    """只有点估计和配对区间下界同时达标，候选模型才有资格晋级。"""
    if not 0 <= min_score_rate <= 1 or not 0 <= min_ci_lower <= 1:
        raise ValueError("晋级门槛必须位于 [0, 1]")
    score_rate = float(match_report["score_rate"])
    ci_lower = float(match_report["score_rate_ci95"][0])
    qualified = score_rate >= min_score_rate and ci_lower >= min_ci_lower
    reason = (
        "候选模型的得分率与配对 95% CI 下界均达到门槛"
        if qualified
        else "候选模型未同时达到得分率与配对 95% CI 下界门槛"
    )
    return {
        "qualified": qualified,
        "promoted": False,
        "min_score_rate": min_score_rate,
        "min_ci95_lower": min_ci_lower,
        "observed_score_rate": score_rate,
        "observed_ci95_lower": ci_lower,
        "reason": reason,
    }


def _atomic_torch_save(payload: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        # 在替换冠军前确认临时文件可完整读取，旧冠军因此始终保持可恢复。
        verified = torch.load(temporary, map_location="cpu", weights_only=False)
        if not {"model_config", "state_dict"}.issubset(verified):
            raise ValueError("导出的冠军检查点缺少必要字段")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def export_champion(
    candidate_path: str | Path,
    champion_path: str | Path,
    league_report: dict,
) -> None:
    """把合格候选原子导出为轻量冠军；不会复制优化器和回放缓冲。"""
    source = Path(candidate_path)
    destination = Path(champion_path)
    if source.resolve() == destination.resolve():
        raise ValueError("候选检查点与冠军路径必须不同")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not {"model_config", "state_dict"}.issubset(payload):
        raise ValueError("候选检查点缺少 model_config 或 state_dict")
    slim = {
        "format_version": int(payload.get("format_version", 1)),
        "inference_symmetry": payload.get("inference_symmetry"),
        "model_identity": dict(payload.get("model_identity", {})),
        "model_config": dict(payload["model_config"]),
        "state_dict": {
            name: tensor.detach().cpu() for name, tensor in payload["state_dict"].items()
        },
        "iteration": int(payload.get("iteration", -1)),
        "source_checkpoint": str(source),
        "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
        "league_result": {
            "games": int(league_report["games"]),
            "wins": int(league_report["wins"]),
            "losses": int(league_report["losses"]),
            "draws": int(league_report["draws"]),
            "score_rate": float(league_report["score_rate"]),
            "score_rate_ci95": list(league_report["score_rate_ci95"]),
            "protocol": dict(league_report["protocol"]),
        },
    }
    _atomic_torch_save(slim, destination)


def _atomic_json_write(report: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="候选模型挑战当前冠军：固定开局交换先后手、配对置信区间和安全晋级"
    )
    parser.add_argument("candidate", type=Path, help="待挑战的候选检查点")
    parser.add_argument("champion", type=Path, help="当前冠军；晋级时在此路径原子替换")
    parser.add_argument("--pairs", type=int, default=32, help="随机开局数；总对局数为其 2 倍")
    parser.add_argument("--opening-plies", type=int, default=4, help="每个随机开局的预置合法手数")
    parser.add_argument("--simulations", type=int, default=96, help="双方每步相同的 MCTS 模拟数")
    parser.add_argument("--c-puct", type=float, default=1.5, help="双方相同的 PUCT 探索系数")
    parser.add_argument("--workers", type=int, default=4, help="并行开局对数量")
    parser.add_argument("--torch-threads", type=int, default=1, help="PyTorch CPU 线程数")
    parser.add_argument("--seed", type=int, default=20260916, help="开局、平手选择和 bootstrap 种子")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--min-score-rate", type=float, default=0.55, help="晋级所需最低得分率")
    parser.add_argument(
        "--min-ci-lower", type=float, default=0.50,
        help="晋级所需配对 95%% bootstrap 区间最低下界",
    )
    parser.add_argument("--report", type=Path, default=Path("checkpoints/league_report.json"))
    parser.add_argument("--no-promote", action="store_true", help="只比赛和报告，不替换冠军")
    parser.add_argument("--candidate-no-symmetry", action="store_true",
                        help="消融实验：只使用候选网络本身，不使用其 D4 推理集成")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.torch_threads <= 0:
        raise ValueError("torch_threads 必须为正整数")
    if args.candidate.resolve() == args.champion.resolve():
        raise ValueError("候选检查点与冠军路径必须不同")
    if args.report.resolve() in (args.candidate.resolve(), args.champion.resolve()):
        raise ValueError("报告路径不能覆盖候选或冠军检查点")

    torch.set_num_threads(args.torch_threads)
    candidate, candidate_payload = load_model(args.candidate)
    champion, champion_payload = load_model(args.champion)
    if args.candidate_no_symmetry:
        candidate.d4_ensemble = False
    match = run_league(
        candidate,
        champion,
        pairs=args.pairs,
        opening_plies=args.opening_plies,
        simulations=args.simulations,
        workers=args.workers,
        c_puct=args.c_puct,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    report = {
        "candidate": {
            "path": str(args.candidate),
            "iteration": int(candidate_payload.get("iteration", -1)),
            "parameters": count_parameters(candidate),
            "inference_symmetry": "none" if args.candidate_no_symmetry else candidate_payload.get("inference_symmetry", "none"),
        },
        "champion": {
            "path": str(args.champion),
            "iteration": int(champion_payload.get("iteration", -1)),
            "parameters": count_parameters(champion),
        },
        "match": match,
    }
    decision = promotion_decision(match, args.min_score_rate, args.min_ci_lower)
    report["promotion"] = decision
    if decision["qualified"] and not args.no_promote:
        export_champion(args.candidate, args.champion, match)
        decision["promoted"] = True
        decision["reason"] = "候选模型达到双门槛，已原子导出为新冠军"
    elif decision["qualified"]:
        decision["reason"] = "候选模型达到双门槛；--no-promote 已阻止文件替换"

    _atomic_json_write(report, args.report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
