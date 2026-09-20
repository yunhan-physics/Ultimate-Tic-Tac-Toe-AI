# -*- coding: utf-8 -*-
"""AlphaZero 风格的 PUCT 蒙特卡洛树搜索。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from game import EMPTY, GameState, apply_move, encode_state, legal_moves, terminal_info
from model import ACTION_SIZE, masked_policy


@dataclass(slots=True)
class Node:
    """树节点中的价值始终站在该节点 ``state.to_play`` 的视角。"""

    state: GameState
    prior: float = 0.0
    visit_count: int = 0
    value_sum: float = 0.0
    expanded: bool = False
    children: dict[int, "Node"] = field(default_factory=dict)

    @property
    def value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0


def terminal_value(state: GameState, winner: int) -> float:
    """把终局赢家转换成当前节点行动方视角的 -1/0/+1。"""
    if winner == EMPTY:
        return 0.0
    return 1.0 if winner == state.to_play else -1.0


def _predict(
    net: torch.nn.Module,
    state: GameState,
    device: torch.device | str,
    cache: dict[GameState, tuple[np.ndarray, float]] | None,
) -> tuple[np.ndarray, float]:
    """预测合法动作先验与当前方价值；网络更新后必须清空 cache。"""
    if cache is not None and state in cache:
        return cache[state]

    encoded = torch.from_numpy(encode_state(state)).unsqueeze(0).to(device)
    mask = torch.zeros((1, ACTION_SIZE), dtype=torch.bool, device=device)
    mask[0, legal_moves(state)] = True

    was_training = net.training
    net.eval()
    with torch.inference_mode():
        logits, value = net(encoded)
        probabilities = masked_policy(logits, mask)
    if was_training:
        net.train()

    prediction = probabilities[0].detach().cpu().numpy().astype(np.float32), float(value.item())
    if cache is not None:
        cache[state] = prediction
    return prediction


def _expand(
    node: Node,
    net: torch.nn.Module,
    device: torch.device | str,
    cache: dict[GameState, tuple[np.ndarray, float]] | None,
) -> float:
    """用网络扩展非终局叶节点，返回该叶节点行动方视角的价值。"""
    done, winner = terminal_info(node.state)
    if done:
        node.expanded = True
        return terminal_value(node.state, winner)

    probabilities, value = _predict(net, node.state, device, cache)
    for move in legal_moves(node.state):
        child_state, _, _ = apply_move(node.state, move)
        node.children[move] = Node(child_state, prior=float(probabilities[move]))
    node.expanded = True
    return value


def _add_dirichlet_noise(
    root: Node,
    alpha: float,
    fraction: float,
    rng: np.random.Generator,
) -> None:
    if not root.children:
        return
    if alpha <= 0 or not 0 <= fraction <= 1:
        raise ValueError("dirichlet_alpha 必须为正，noise_fraction 必须位于 [0, 1]")
    actions = list(root.children)
    noise = rng.dirichlet(np.full(len(actions), alpha))
    for action, sample in zip(actions, noise):
        child = root.children[action]
        child.prior = (1.0 - fraction) * child.prior + fraction * float(sample)


def _select_child(node: Node, c_puct: float) -> tuple[int, Node]:
    """最大化 ``-Q_child + U``；负号用于切换双方视角。"""
    if not node.children:
        raise ValueError("无法从没有子节点的节点中选择")
    sqrt_parent = np.sqrt(max(1, node.visit_count))
    best_action = -1
    best_child: Node | None = None
    best_score = float("-inf")
    for action, child in node.children.items():
        exploration = c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
        score = -child.value + exploration
        if score > best_score:
            best_score = score
            best_action = action
            best_child = child
    assert best_child is not None
    return best_action, best_child


def _backpropagate(path: list[Node], leaf_value: float) -> None:
    """沿路径回传，每跨过一手棋就翻转一次价值符号。"""
    value = leaf_value
    for node in reversed(path):
        node.visit_count += 1
        node.value_sum += value
        value = -value


def search(
    state: GameState,
    net: torch.nn.Module,
    device: torch.device | str = "cpu",
    simulations: int = 100,
    c_puct: float = 1.5,
    add_noise: bool = False,
    dirichlet_alpha: float = 0.3,
    noise_fraction: float = 0.25,
    cache: dict[GameState, tuple[np.ndarray, float]] | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float]:
    """执行 MCTS，返回 81 维访问分布和根节点价值。"""
    if simulations <= 0:
        raise ValueError("simulations 必须为正整数")
    if c_puct <= 0:
        raise ValueError("c_puct 必须为正数")

    done, winner = terminal_info(state)
    if done:
        return np.zeros(ACTION_SIZE, dtype=np.float32), terminal_value(state, winner)

    rng = rng if rng is not None else np.random.default_rng()
    root = Node(state)
    _expand(root, net, device, cache)
    if add_noise:
        _add_dirichlet_noise(root, dirichlet_alpha, noise_fraction, rng)

    for _ in range(simulations):
        node = root
        path = [root]

        while node.expanded and node.children:
            _, node = _select_child(node, c_puct)
            path.append(node)

        done, winner = terminal_info(node.state)
        if done:
            leaf_value = terminal_value(node.state, winner)
            node.expanded = True
        else:
            leaf_value = _expand(node, net, device, cache)
        _backpropagate(path, leaf_value)

    visits = np.zeros(ACTION_SIZE, dtype=np.float32)
    total = sum(child.visit_count for child in root.children.values())
    if total:
        for action, child in root.children.items():
            visits[action] = child.visit_count / total
    return visits, root.value
