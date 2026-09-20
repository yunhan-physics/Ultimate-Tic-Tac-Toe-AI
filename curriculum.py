# -*- coding: utf-8 -*-
"""启发式教师课程：补足纯自对弈早期难以发现的局部战术。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import torch

from baseline import heuristic_policy, random_action
from game import EMPTY, P1, P2, apply_move, encode_state, legal_moves, new_game, score
from self_play import TrainingExample
from train import load_checkpoint, resolve_device, save_checkpoint, train_updates


def generate_teacher_game(
    rng: np.random.Generator,
    max_random_opening: int = 8,
) -> list[TrainingExample]:
    """先随机扰动开局，再由启发式双方完成棋局并生成策略/价值标签。"""
    if max_random_opening < 0:
        raise ValueError("max_random_opening 不能为负")
    state = new_game()
    opening_moves = int(rng.integers(max_random_opening + 1))
    for _ in range(opening_moves):
        if not legal_moves(state):
            break
        state, done, _ = apply_move(state, random_action(state, rng))
        if done:
            break

    history: list[tuple[np.ndarray, np.ndarray, int]] = []
    winner = EMPTY
    while legal_moves(state):
        policy = heuristic_policy(state)
        history.append((encode_state(state), policy, state.to_play))
        move = int(rng.choice(np.arange(81), p=policy))
        state, done, winner = apply_move(state, move)
        if done:
            break
    if winner == EMPTY:
        p1_score, p2_score = score(state)
        winner = P1 if p1_score > p2_score else P2 if p2_score > p1_score else EMPTY

    return [
        TrainingExample(
            encoded,
            policy,
            0.0 if winner == EMPTY else (1.0 if winner == to_play else -1.0),
        )
        for encoded, policy, to_play in history
    ]


def run_curriculum(args: argparse.Namespace) -> dict:
    torch.set_num_threads(args.torch_threads)
    device = resolve_device(args.train_device)
    net, optimizer, replay, next_iteration, history, rng, saved_args = load_checkpoint(
        Path(args.checkpoint), device
    )
    started = time.perf_counter()
    teacher_examples = []
    for game_index in range(args.teacher_games):
        teacher_examples.extend(generate_teacher_game(rng, args.max_random_opening))
        if (game_index + 1) % 100 == 0:
            print(
                f"教师棋局 {game_index + 1}/{args.teacher_games}，样本 {len(teacher_examples)}",
                flush=True,
            )
    replay.add(teacher_examples)

    metrics = train_updates(
        net,
        optimizer,
        replay,
        steps=args.train_steps,
        batch_size=args.batch_size,
        device=device,
        rng=rng,
        grad_clip=float(saved_args.get("grad_clip", 5.0)),
    )
    record = {
        "phase": "heuristic_curriculum",
        "iteration": next_iteration - 1,
        "teacher_games": args.teacher_games,
        "teacher_examples": len(teacher_examples),
        "buffer": len(replay),
        **metrics,
        "seconds": time.perf_counter() - started,
    }
    history.append(record)

    checkpoint_args = argparse.Namespace(**saved_args)
    save_checkpoint(
        Path(args.output),
        net,
        optimizer,
        replay,
        next_iteration - 1,
        history,
        checkpoint_args,
        rng,
    )
    print(
        f"课程训练完成：{args.teacher_games} 局、{len(teacher_examples)} 个教师样本，"
        f"loss={metrics['loss']:.4f}，已保存 {args.output}"
    )
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="给终极井字棋模型加入启发式课程")
    parser.add_argument("checkpoint")
    parser.add_argument("--output", default="checkpoints/curriculum.pth")
    parser.add_argument("--teacher-games", type=int, default=400)
    parser.add_argument("--max-random-opening", type=int, default=8)
    parser.add_argument("--train-steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-device", default="auto")
    parser.add_argument("--torch-threads", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    run_curriculum(parse_args())
