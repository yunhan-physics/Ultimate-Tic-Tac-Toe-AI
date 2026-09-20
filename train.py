# -*- coding: utf-8 -*-
"""从随机权重开始进行 AlphaZero 式自对弈训练。"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from model import UltimateNet, count_parameters
from self_play import ReplayBuffer, SelfPlayResult, play_one_game


def policy_value_loss(
    policy_logits: torch.Tensor,
    value_prediction: torch.Tensor,
    target_policy: torch.Tensor,
    target_value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """软标签策略交叉熵 + 价值均方误差。"""
    policy_loss = -(target_policy * F.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
    value_loss = F.mse_loss(value_prediction, target_value)
    return policy_loss + value_loss, policy_loss, value_loss


def train_updates(
    net: UltimateNet,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    steps: int,
    batch_size: int,
    device: torch.device,
    rng: np.random.Generator,
    grad_clip: float = 5.0,
) -> dict[str, float]:
    """从回放缓冲随机抽样并更新网络若干步。"""
    if steps <= 0:
        raise ValueError("steps 必须为正整数")
    if len(replay) == 0:
        raise ValueError("回放缓冲为空")
    actual_batch = min(batch_size, len(replay))
    net.train()
    totals = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "grad_norm": 0.0}

    for _ in range(steps):
        states, policies, values = replay.sample(actual_batch, rng, augment=True)
        states_tensor = torch.from_numpy(states).to(device)
        policies_tensor = torch.from_numpy(policies).to(device)
        values_tensor = torch.from_numpy(values).to(device)

        optimizer.zero_grad(set_to_none=True)
        logits, value_prediction = net(states_tensor)
        loss, policy_loss, value_loss = policy_value_loss(
            logits, value_prediction, policies_tensor, values_tensor
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("训练损失出现 NaN 或无穷大")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("梯度出现 NaN 或无穷大")
        optimizer.step()

        totals["loss"] += float(loss.detach())
        totals["policy_loss"] += float(policy_loss.detach())
        totals["value_loss"] += float(value_loss.detach())
        totals["grad_norm"] += float(grad_norm.detach())

    return {name: value / steps for name, value in totals.items()}


def collect_self_play(
    net: UltimateNet,
    games: int,
    workers: int,
    simulations: int,
    c_puct: float,
    temperature_moves: int,
    seeds: list[int],
) -> list[SelfPlayResult]:
    """在 CPU 上并行采集互相独立的自对弈棋局。"""
    if len(seeds) != games:
        raise ValueError("每局自对弈都必须有独立随机种子")
    net.eval()

    def run(seed: int) -> SelfPlayResult:
        return play_one_game(
            net,
            device="cpu",
            simulations=simulations,
            c_puct=c_puct,
            temperature_moves=temperature_moves,
            cache={},
            rng=np.random.default_rng(seed),
        )

    if workers <= 1:
        return [run(seed) for seed in seeds]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(run, seeds))


def cpu_state_dict(net: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu() for name, tensor in net.state_dict().items()}


def save_checkpoint(
    path: Path,
    net: UltimateNet,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    iteration: int,
    history: list[dict],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> None:
    """先写临时文件再原子替换，避免中断留下半个检查点。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model_identity": {
            "name": getattr(args, "model_name", "Tic-Tac-Toe Sophon"),
            "version": getattr(args, "model_version", None),
        },
        "model_config": dict(net.config),
        "state_dict": cpu_state_dict(net),
        "optimizer_state": optimizer.state_dict(),
        "replay": replay.state_dict(),
        "iteration": iteration,
        "history": history,
        "args": vars(args),
        "rng_state": rng.bit_generator.state,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(
    path: Path,
    train_device: torch.device,
) -> tuple[UltimateNet, torch.optim.Optimizer, ReplayBuffer, int, list[dict], np.random.Generator, dict]:
    payload = torch.load(path, map_location=train_device, weights_only=False)
    net = UltimateNet(**payload["model_config"]).to(train_device)
    net.load_state_dict(payload["state_dict"])
    saved_args = payload["args"]
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=float(saved_args["learning_rate"]),
        weight_decay=float(saved_args["weight_decay"]),
    )
    optimizer.load_state_dict(payload["optimizer_state"])
    replay = ReplayBuffer.from_state_dict(payload["replay"])
    rng = np.random.default_rng()
    rng.bit_generator.state = payload["rng_state"]
    return (
        net,
        optimizer,
        replay,
        int(payload["iteration"]) + 1,
        list(payload["history"]),
        rng,
        saved_args,
    )


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求了 CUDA，但当前 PyTorch 无法使用 CUDA")
    return device


def run_training(args: argparse.Namespace) -> list[dict]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    train_device = resolve_device(args.train_device)
    checkpoint_dir = Path(args.checkpoint_dir)

    if args.resume:
        net, optimizer, replay, start_iteration, history, rng, saved_args = load_checkpoint(
            Path(args.resume), train_device
        )
        print(f"恢复检查点：{args.resume}，从第 {start_iteration} 轮继续")
        if dict(net.config) != {
            "input_channels": 6,
            "channels": int(saved_args["channels"]),
            "blocks": int(saved_args["blocks"]),
            "value_hidden": int(saved_args["value_hidden"]),
        }:
            raise ValueError("检查点网络配置不一致")
        # 网络形状与优化器超参必须沿用检查点；迭代数、每轮棋局数等运行规模
        # 仍允许由本次命令覆盖。这样继续保存的检查点也不会写入错误配置。
        for key in ("channels", "blocks", "value_hidden", "learning_rate", "weight_decay"):
            setattr(args, key, saved_args[key])
    else:
        net = UltimateNet(
            channels=args.channels,
            blocks=args.blocks,
            value_hidden=args.value_hidden,
        ).to(train_device)
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        replay = ReplayBuffer(args.buffer_size)
        start_iteration = 1
        history = []
        rng = np.random.default_rng(args.seed)

    self_play_net = UltimateNet(**net.config).cpu()
    self_play_net.load_state_dict(cpu_state_dict(net))
    print(
        f"模型：{args.model_name} {args.model_version or '(candidate)'}；"
        f"网络参数：{count_parameters(net):,}；自对弈=CPU/{args.workers}线程；"
        f"训练设备={train_device}；缓冲容量={replay.capacity:,}"
    )

    for iteration in range(start_iteration, args.iterations + 1):
        started = time.perf_counter()
        seeds = [int(seed) for seed in rng.integers(0, 2**32 - 1, size=args.games_per_iter)]
        games = collect_self_play(
            self_play_net,
            games=args.games_per_iter,
            workers=args.workers,
            simulations=args.simulations,
            c_puct=args.c_puct,
            temperature_moves=args.temperature_moves,
            seeds=seeds,
        )
        for game in games:
            replay.add(game.examples)

        metrics = train_updates(
            net,
            optimizer,
            replay,
            steps=args.train_steps,
            batch_size=args.batch_size,
            device=train_device,
            rng=rng,
            grad_clip=args.grad_clip,
        )
        self_play_net.load_state_dict(cpu_state_dict(net))

        p1_wins = sum(game.winner == 1 for game in games)
        p2_wins = sum(game.winner == 2 for game in games)
        draws = len(games) - p1_wins - p2_wins
        record = {
            "iteration": iteration,
            "games": len(games),
            "examples": sum(len(game.examples) for game in games),
            "buffer": len(replay),
            "p1_wins": p1_wins,
            "p2_wins": p2_wins,
            "draws": draws,
            "average_moves": float(np.mean([game.moves for game in games])),
            "simulations_per_move": args.simulations,
            **metrics,
            "seconds": time.perf_counter() - started,
        }
        history.append(record)
        print(
            f"[{iteration:03d}/{args.iterations:03d}] "
            f"棋局 {len(games)}（{p1_wins}/{p2_wins}/{draws}） "
            f"样本 {record['examples']} 缓冲 {len(replay)} "
            f"loss {record['loss']:.4f}="
            f"{record['policy_loss']:.4f}+{record['value_loss']:.4f} "
            f"耗时 {record['seconds']:.1f}s",
            flush=True,
        )

        save_checkpoint(
            checkpoint_dir / "last.pth",
            net,
            optimizer,
            replay,
            iteration,
            history,
            args,
            rng,
        )
        (checkpoint_dir / "training_log.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    save_checkpoint(
        checkpoint_dir / "trained.pth",
        net,
        optimizer,
        replay,
        args.iterations,
        history,
        args,
        rng,
    )
    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从零训练终极井字棋 AlphaZero")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--games-per-iter", type=int, default=12)
    parser.add_argument("--simulations", type=int, default=96)
    parser.add_argument("--train-steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=30_000)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--value-hidden", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--temperature-moves", type=int, default=18)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--train-device", default="auto")
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--model-name", default="Tic-Tac-Toe Sophon")
    parser.add_argument("--model-version", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    run_training(parse_args())
