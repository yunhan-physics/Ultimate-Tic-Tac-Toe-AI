# -*- coding: utf-8 -*-
"""自对弈、D4 数据增强与回放缓冲。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from game import EMPTY, P1, P2, GameState, apply_move, encode_state, legal_moves, new_game, score
from mcts import search


@dataclass(frozen=True, slots=True)
class TrainingExample:
    state: np.ndarray
    policy: np.ndarray
    value: float


@dataclass(frozen=True, slots=True)
class SelfPlayResult:
    examples: list[TrainingExample]
    winner: int
    moves: int
    p1_score: int
    p2_score: int


def select_action(
    policy: np.ndarray,
    legal: list[int],
    temperature: float,
    rng: np.random.Generator,
) -> int:
    """按温度从 MCTS 访问分布中选择一个合法动作。"""
    if policy.shape != (81,):
        raise ValueError("policy 必须是长度 81 的一维数组")
    if not legal:
        raise ValueError("没有合法动作可供选择")
    if not np.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature 必须是有限的非负数")

    legal_array = np.asarray(legal, dtype=np.int64)
    weights = np.asarray(policy[legal_array], dtype=np.float64)
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
    if temperature <= 1e-8:
        best = legal_array[np.isclose(weights, weights.max(), rtol=1e-7, atol=1e-12)]
        return int(rng.choice(best))

    if not np.isclose(temperature, 1.0):
        positive = weights > 0
        scaled = np.zeros_like(weights)
        if positive.any():
            log_weights = np.log(weights[positive]) / temperature
            log_weights -= log_weights.max()
            scaled[positive] = np.exp(log_weights)
        weights = scaled
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        weights = np.ones(len(legal_array), dtype=np.float64) / len(legal_array)
    else:
        weights /= total
    return int(rng.choice(legal_array, p=weights))


def _d4_transform(
    state: np.ndarray,
    policy: np.ndarray,
    transform: int,
) -> tuple[np.ndarray, np.ndarray]:
    """应用编号 0~7 的一个 D4 变换。"""
    if state.shape != (6, 9, 9):
        raise ValueError("state 形状必须是 (6, 9, 9)")
    if policy.shape != (81,):
        raise ValueError("policy 形状必须是 (81,)")
    if transform not in range(8):
        raise ValueError("transform 必须是 0~7")

    policy_board = policy.reshape(9, 9)
    mirrored, rotations = divmod(transform, 4)
    base_state = np.flip(state, axis=2) if mirrored else state
    base_policy = np.flip(policy_board, axis=1) if mirrored else policy_board
    transformed_state = np.rot90(base_state, rotations, axes=(1, 2))
    transformed_policy = np.rot90(base_policy, rotations)
    return (
        np.ascontiguousarray(transformed_state, dtype=np.float32),
        np.ascontiguousarray(transformed_policy.reshape(81), dtype=np.float32),
    )


def d4_augment(state: np.ndarray, policy: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """生成正方形棋盘的四次旋转及其镜像，共八个样本。"""
    return [_d4_transform(state, policy, transform) for transform in range(8)]


class ReplayBuffer:
    """固定容量环形缓冲；只存原样本，在抽样时随机应用 D4 变换。"""

    def __init__(self, capacity: int = 50_000):
        if capacity <= 0:
            raise ValueError("capacity 必须为正整数")
        self.capacity = capacity
        self._data: list[TrainingExample | None] = [None] * capacity
        self._size = 0
        self._next = 0

    def __len__(self) -> int:
        return self._size

    @staticmethod
    def _validated_copy(example: TrainingExample) -> TrainingExample:
        if example.state.shape != (6, 9, 9) or example.policy.shape != (81,):
            raise ValueError("训练样本形状不正确")
        if not np.isfinite(example.state).all() or not np.isfinite(example.policy).all():
            raise ValueError("训练样本不能包含 NaN 或无穷大")
        if (example.policy < 0).any() or not np.isclose(example.policy.sum(), 1.0, atol=1e-5):
            raise ValueError("策略目标必须是非负且归一化的概率分布")
        if example.value not in (-1.0, 0.0, 1.0):
            raise ValueError("价值标签必须是 -1、0 或 1")
        return TrainingExample(
            np.array(example.state, dtype=np.float32, copy=True, order="C"),
            np.array(example.policy, dtype=np.float32, copy=True, order="C"),
            float(example.value),
        )

    def add(self, examples: list[TrainingExample]) -> None:
        for example in examples:
            self._data[self._next] = self._validated_copy(example)
            self._next = (self._next + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
        augment: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if batch_size <= 0:
            raise ValueError("batch_size 必须为正整数")
        if batch_size > self._size:
            raise ValueError("回放缓冲中的样本不足")
        indices = rng.choice(self._size, size=batch_size, replace=False)
        selected = [self._data[int(index)] for index in indices]
        assert all(example is not None for example in selected)
        transformed = []
        for example in selected:
            assert example is not None
            if augment:
                transformed.append(_d4_transform(example.state, example.policy, int(rng.integers(8))))
            else:
                transformed.append((example.state, example.policy))
        states = np.stack([item[0] for item in transformed]).astype(np.float32, copy=False)
        policies = np.stack([item[1] for item in transformed]).astype(np.float32, copy=False)
        typed_selected = [example for example in selected if example is not None]
        values = np.asarray([example.value for example in typed_selected], dtype=np.float32).reshape(-1, 1)
        return states, policies, values

    def as_list(self) -> list[TrainingExample]:
        """返回浅拷贝，供检查点选择性保存。"""
        if self._size < self.capacity:
            return [example for example in self._data[:self._size] if example is not None]
        ordered = self._data[self._next:] + self._data[:self._next]
        return [example for example in ordered if example is not None]

    def state_dict(self) -> dict:
        """导出原始样本，供训练检查点完整恢复。"""
        examples = self.as_list()
        if examples:
            states = np.stack([example.state for example in examples]).astype(np.float32, copy=False)
            policies = np.stack([example.policy for example in examples]).astype(np.float32, copy=False)
            values = np.asarray([example.value for example in examples], dtype=np.float32)
        else:
            states = np.empty((0, 6, 9, 9), dtype=np.float32)
            policies = np.empty((0, 81), dtype=np.float32)
            values = np.empty((0,), dtype=np.float32)
        return {
            "capacity": self.capacity,
            "states": states,
            "policies": policies,
            "values": values,
        }

    @classmethod
    def from_state_dict(cls, data: dict) -> "ReplayBuffer":
        buffer = cls(int(data["capacity"]))
        examples = [
            TrainingExample(state, policy, float(value))
            for state, policy, value in zip(data["states"], data["policies"], data["values"])
        ]
        buffer.add(examples)
        return buffer


def play_one_game(
    net: torch.nn.Module,
    device: torch.device | str = "cpu",
    simulations: int = 100,
    c_puct: float = 1.5,
    temperature_moves: int = 18,
    cache: dict | None = None,
    rng: np.random.Generator | None = None,
    initial_state: GameState | None = None,
) -> SelfPlayResult:
    """让当前网络与自身完成一局，生成当前行动方视角的训练样本。"""
    if temperature_moves < 0:
        raise ValueError("temperature_moves 不能为负")
    rng = rng if rng is not None else np.random.default_rng()
    state = initial_state if initial_state is not None else new_game()
    history: list[tuple[np.ndarray, np.ndarray, int]] = []
    moves_made = 0

    while legal_moves(state):
        policy, _ = search(
            state,
            net,
            device=device,
            simulations=simulations,
            c_puct=c_puct,
            add_noise=True,
            cache=cache,
            rng=rng,
        )
        history.append((encode_state(state), policy.copy(), state.to_play))
        temperature = 1.0 if moves_made < temperature_moves else 0.0
        move = select_action(policy, legal_moves(state), temperature, rng)

        state, done, winner = apply_move(state, move)
        moves_made += 1
        if done:
            break
    else:
        p1_score, p2_score = score(state)
        winner = P1 if p1_score > p2_score else P2 if p2_score > p1_score else EMPTY

    examples = []
    for encoded, policy, to_play in history:
        value = 0.0 if winner == EMPTY else (1.0 if winner == to_play else -1.0)
        examples.append(TrainingExample(encoded, policy, value))

    p1_score, p2_score = score(state)
    return SelfPlayResult(examples, winner, moves_made, p1_score, p2_score)
