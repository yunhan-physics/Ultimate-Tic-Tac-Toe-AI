# -*- coding: utf-8 -*-
"""终极井字棋的人机对战与无交互演示入口。

默认从 ``checkpoints/best.pth`` 加载模型。玩家可以用全局 1~9 行、
1~9 列输入落子，例如 ``3 7``；也可以直接输入动作编号 0~80。
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
import re
import sys

import numpy as np
import torch

from baseline import heuristic_action
from game import (
    ANY_BOARD,
    DRAW,
    EMPTY,
    OPEN,
    P1,
    P2,
    GameState,
    action_to_parts,
    apply_move,
    legal_moves,
    new_game,
    score,
    terminal_info,
)
from game_record import GameRecorder
from mcts import search
from model import UltimateNet, configure_inference
from self_play import select_action


DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "best.pth"
DEFAULT_RECORDS_DIR = Path(__file__).resolve().parent / "records" / "human_games"
InputFunction = Callable[[str], str]
OutputFunction = Callable[[str], None]


class UserQuit(Exception):
    """用户主动退出当前对局。"""


def positive_int(value: str) -> int:
    """供 argparse 使用的正整数解析器。"""
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("必须是整数") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("必须大于 0")
    return result


def parse_side(value: str) -> int:
    """把常见的先后手写法转换成 P1/P2。"""
    normalized = value.strip().lower()
    first = {"1", "先", "先手", "first", "p1", "x"}
    second = {"2", "后", "后手", "second", "p2", "o"}
    if normalized in first:
        return P1
    if normalized in second:
        return P2
    raise ValueError("请输入 1/先手，或 2/后手")


def parse_move(text: str, state: GameState | None = None) -> int:
    """解析玩家输入。

    两个整数表示全局 ``行 列``（均为 1~9）；一个整数表示 0~80 的
    动作编号。传入 state 时还会检查该动作在当前局面是否合法。
    """
    cleaned = text.strip().lower()
    if not cleaned:
        raise ValueError("输入不能为空")

    # 允许 ``3 7``、``3,7``、``3，7``、``3/7`` 等自然写法。
    coordinate_text = re.sub(r"[,，/;；]", " ", cleaned)
    parts = coordinate_text.split()
    if len(parts) == 2:
        try:
            row, column = (int(part) for part in parts)
        except ValueError as exc:
            raise ValueError("坐标必须是两个整数，例如：3 7") from exc
        if row not in range(1, 10) or column not in range(1, 10):
            raise ValueError("行和列都必须在 1~9 之间")
        move = (row - 1) * 9 + column - 1
    elif len(parts) == 1:
        action_text = parts[0][1:] if parts[0].startswith("#") else parts[0]
        try:
            move = int(action_text)
        except ValueError as exc:
            raise ValueError("请输入“行 列”（1~9），或动作编号（0~80）") from exc
        if move not in range(81):
            raise ValueError("动作编号必须在 0~80 之间")
    else:
        raise ValueError("请输入两个坐标，例如：3 7")

    if state is not None and move not in legal_moves(state):
        row, column = divmod(move, 9)
        raise ValueError(f"位置（{row + 1}, {column + 1}）当前不能落子")
    return move


def action_label(move: int) -> str:
    """返回适合展示给玩家的全局坐标和动作编号。"""
    if move not in range(81):
        raise ValueError("动作编号必须在 0~80 之间")
    row, column = divmod(move, 9)
    return f"（{row + 1}, {column + 1}），动作 {move}"


def _small_board_label(index: int) -> str:
    board_row, board_column = divmod(index, 3)
    return f"小棋盘 {index + 1}（大棋盘第 {board_row + 1} 行、第 {board_column + 1} 列）"


def render_board(state: GameState) -> str:
    """绘制带全局 1~9 行列坐标的九宫格棋盘。"""
    symbols = {EMPTY: ".", P1: "X", P2: "O"}
    lines = ["       列  1 2 3 | 4 5 6 | 7 8 9"]
    for row in range(9):
        chunks: list[str] = []
        for block_column in range(3):
            start = row * 9 + block_column * 3
            chunks.append(" ".join(symbols[state.board[start + offset]] for offset in range(3)))
        lines.append(f"行 {row + 1:>2}     " + " | ".join(chunks))
        if row in (2, 5):
            lines.append("           ------+-------+------")
    return "\n".join(lines)


def render_small_boards(state: GameState) -> str:
    """绘制九个小棋盘的占领/封闭状态。"""
    symbols = {OPEN: "·", P1: "X", P2: "O", DRAW: "="}
    lines = ["小棋盘状态（·开放，X/O 已占领，= 和棋封闭）："]
    for row in range(3):
        lines.append("  " + " ".join(symbols[state.small_boards[row * 3 + column]] for column in range(3)))
    return "\n".join(lines)


def render_state(state: GameState, last_move: int | None = None) -> str:
    """汇总棋盘、比分、行动方和强制落子区域。"""
    p1_score, p2_score = score(state)
    done, _ = terminal_info(state)
    lines = [render_board(state), "", render_small_boards(state), f"比分：X {p1_score} : {p2_score} O"]
    if last_move is not None:
        lines.append(f"上一手：{action_label(last_move)}")
    if done:
        lines.append("当前状态：对局结束")
    else:
        player = "X（先手）" if state.to_play == P1 else "O（后手）"
        lines.append(f"当前行动：{player}")
        if state.next_board == ANY_BOARD:
            lines.append("强制区域：无，可在任意开放的小棋盘落子")
        else:
            lines.append(f"强制区域：{_small_board_label(state.next_board)}")
    return "\n".join(lines)


def load_ai_model(
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    device: str | torch.device = "cpu",
) -> tuple[UltimateNet, dict]:
    """加载轻量或完整训练检查点，返回已进入推理模式的网络。"""
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"找不到模型检查点：{path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "model_config" not in payload or "state_dict" not in payload:
        raise ValueError("检查点缺少 model_config 或 state_dict")
    model = UltimateNet(**payload["model_config"])
    model.load_state_dict(payload["state_dict"])
    configure_inference(model, payload)
    model.to(device)
    model.eval()
    return model, payload


def choose_ai_action(
    model: UltimateNet,
    state: GameState,
    simulations: int,
    rng: np.random.Generator,
    *,
    device: str | torch.device = "cpu",
    c_puct: float = 1.5,
    cache: dict | None = None,
) -> int:
    """用无探索噪声的 MCTS 为实战选择一步合法棋。"""
    if simulations <= 0:
        raise ValueError("simulations 必须大于 0")
    moves = legal_moves(state)
    if not moves:
        raise ValueError("终局状态没有可选动作")
    visits, _ = search(
        state,
        model,
        device=device,
        simulations=simulations,
        c_puct=c_puct,
        add_noise=False,
        cache=cache,
        rng=rng,
    )
    # 温度为 0：总是选访问次数最多的动作；并列时由 seed 可复现地决定。
    return select_action(visits, moves, temperature=0.0, rng=rng)


def prompt_human_move(
    state: GameState,
    input_fn: InputFunction = input,
    output_fn: OutputFunction = print,
) -> int:
    """持续询问直到得到合法落子；q/quit/退出会结束对局。"""
    while True:
        try:
            raw = input_fn("你的落子（行 列，如 3 7；或动作 0~80；q 退出）：")
        except (EOFError, KeyboardInterrupt) as exc:
            raise UserQuit from exc
        if raw.strip().lower() in {"q", "quit", "exit", "退出"}:
            raise UserQuit
        try:
            return parse_move(raw, state)
        except ValueError as exc:
            output_fn(f"输入无效：{exc}")


def prompt_human_side(
    input_fn: InputFunction = input,
    output_fn: OutputFunction = print,
) -> int:
    """持续询问玩家选择先手或后手。"""
    while True:
        try:
            raw = input_fn("请选择先后手（1=先手 X，2=后手 O）：")
        except (EOFError, KeyboardInterrupt) as exc:
            raise UserQuit from exc
        if raw.strip().lower() in {"q", "quit", "exit", "退出"}:
            raise UserQuit
        try:
            return parse_side(raw)
        except ValueError as exc:
            output_fn(f"输入无效：{exc}")


def result_message(state: GameState, winner: int, human_player: int | None = None) -> str:
    """生成含最终小棋盘比分的结果说明。"""
    p1_score, p2_score = score(state)
    if winner == EMPTY:
        outcome = "平局"
    elif human_player is None:
        outcome = f"{'X' if winner == P1 else 'O'} 获胜"
    elif winner == human_player:
        outcome = "你获胜了"
    else:
        outcome = "AI 获胜"
    return f"{outcome}。最终比分：X {p1_score} : {p2_score} O"


def finish_record_safely(
    recorder: GameRecorder | None,
    state: GameState,
    winner: int | None,
    *,
    status: str,
    reason: str | None,
    output_fn: OutputFunction,
) -> Path | None:
    """保存棋谱并把失败明确告诉玩家，但不让保存问题破坏已经完成的对局。"""
    if recorder is None:
        return None
    try:
        path = recorder.finish(state, winner, status=status, reason=reason)
    except (OSError, ValueError, RuntimeError) as exc:
        output_fn(f"警告：棋谱保存失败：{exc}")
        return None
    output_fn(f"棋谱已保存：{path}")
    return path


def play_human_game(
    model: UltimateNet,
    human_player: int,
    simulations: int,
    rng: np.random.Generator,
    *,
    device: str | torch.device = "cpu",
    c_puct: float = 1.5,
    input_fn: InputFunction = input,
    output_fn: OutputFunction = print,
    recorder: GameRecorder | None = None,
) -> tuple[GameState, int]:
    """进行一局人机对战，返回最终状态和赢家。"""
    if human_player not in (P1, P2):
        raise ValueError("human_player 必须是 P1 或 P2")
    state = new_game()
    cache: dict = {}
    output_fn("你执 " + ("X（先手）" if human_player == P1 else "O（后手）"))
    output_fn(render_state(state))

    try:
        while True:
            state_before = state
            if state.to_play == human_player:
                move = prompt_human_move(state, input_fn, output_fn)
                actor = "你"
                record_actor = "human"
            else:
                output_fn(f"AI 思考中（{simulations} 次 MCTS 模拟）……")
                move = choose_ai_action(
                    model,
                    state,
                    simulations,
                    rng,
                    device=device,
                    c_puct=c_puct,
                    cache=cache,
                )
                actor = "AI"
                record_actor = "ai"
            state, done, winner = apply_move(state, move)
            if recorder is not None:
                recorder.record_move(state_before, move, record_actor, state)
            output_fn(f"\n{actor}落在 {action_label(move)}")
            output_fn(render_state(state, last_move=move))
            if done:
                output_fn(result_message(state, winner, human_player))
                finish_record_safely(
                    recorder,
                    state,
                    winner,
                    status="completed",
                    reason=None,
                    output_fn=output_fn,
                )
                return state, winner
    except UserQuit:
        finish_record_safely(
            recorder,
            state,
            None,
            status="aborted",
            reason="user_quit",
            output_fn=output_fn,
        )
        raise


def play_demo(
    model: UltimateNet,
    mode: str,
    simulations: int,
    rng: np.random.Generator,
    *,
    device: str | torch.device = "cpu",
    c_puct: float = 1.5,
    output_fn: OutputFunction = print,
) -> tuple[GameState, int]:
    """运行无需输入的 AI 对启发式或 AI 自战演示。"""
    normalized = mode.strip().lower()
    if normalized in {"heuristic", "ai-vs-heuristic"}:
        demo_mode = "heuristic"
        output_fn("演示模式：训练 AI（X）对启发式 AI（O）")
    elif normalized in {"self", "ai", "ai-vs-ai"}:
        demo_mode = "self"
        output_fn("演示模式：训练 AI 自战（X 对 O）")
    else:
        raise ValueError("演示模式必须是 heuristic 或 self")

    state = new_game()
    cache: dict = {}
    output_fn(render_state(state))
    while True:
        if demo_mode == "heuristic" and state.to_play == P2:
            move = heuristic_action(state, rng)
            actor = "启发式 AI（O）"
        else:
            move = choose_ai_action(
                model,
                state,
                simulations,
                rng,
                device=device,
                c_puct=c_puct,
                cache=cache,
            )
            actor = "训练 AI（X）" if state.to_play == P1 else "训练 AI（O）"
        state, done, winner = apply_move(state, move)
        output_fn(f"\n{actor}落在 {action_label(move)}")
        output_fn(render_state(state, last_move=move))
        if done:
            output_fn(result_message(state, winner))
            return state, winner


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="与终极井字棋 AI 对战")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="模型检查点（默认 checkpoints/best.pth）",
    )
    parser.add_argument(
        "--side",
        "--human-side",
        dest="side",
        help="玩家先后手：first/second、1/2、先手/后手；省略时现场询问",
    )
    parser.add_argument("--simulations", type=positive_int, default=128, help="AI 每步 MCTS 模拟次数")
    parser.add_argument("--seed", type=int, default=20260916, help="随机种子")
    parser.add_argument("--c-puct", type=float, default=1.5, help="MCTS 探索系数")
    parser.add_argument("--torch-threads", type=positive_int, default=1, help="PyTorch CPU 线程数")
    parser.add_argument(
        "--records-dir",
        type=Path,
        default=DEFAULT_RECORDS_DIR,
        help="真人棋谱保存目录（默认 records/human_games）",
    )
    parser.add_argument("--no-record", action="store_true", help="本局不保存真人棋谱")
    parser.add_argument(
        "--demo",
        nargs="?",
        const="heuristic",
        choices=("heuristic", "self", "ai-vs-heuristic", "ai-vs-ai"),
        help="无交互演示；默认 AI 对启发式，self/ai-vs-ai 表示 AI 自战",
    )
    args = parser.parse_args(argv)
    if not np.isfinite(args.c_puct) or args.c_puct <= 0:
        parser.error("--c-puct 必须是有限正数")
    if args.side is not None:
        try:
            args.side = parse_side(args.side)
        except ValueError as exc:
            parser.error(str(exc))
    return args


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    torch.set_num_threads(args.torch_threads)
    rng = np.random.default_rng(args.seed)
    try:
        model, payload = load_ai_model(args.checkpoint)
    except (FileNotFoundError, OSError, KeyError, ValueError, RuntimeError) as exc:
        print(f"无法加载模型：{exc}", file=sys.stderr)
        return 2

    iteration = payload.get("iteration", "未知")
    print(f"已加载模型：{Path(args.checkpoint)}（训练轮次：{iteration}）")
    try:
        if args.demo is not None:
            play_demo(model, args.demo, args.simulations, rng, c_puct=args.c_puct)
        else:
            human_player = args.side if args.side is not None else prompt_human_side()
            recorder = None
            if not args.no_record:
                recorder = GameRecorder(
                    args.records_dir,
                    human_player=human_player,
                    checkpoint=str(Path(args.checkpoint).expanduser().resolve()),
                    model_iteration=payload.get("iteration"),
                    simulations=args.simulations,
                    c_puct=args.c_puct,
                    seed=args.seed,
                )
                print(f"棋谱记录已开启：{args.records_dir}")
            play_human_game(
                model,
                human_player,
                args.simulations,
                rng,
                c_puct=args.c_puct,
                recorder=recorder,
            )
    except UserQuit:
        print("已退出本局。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
