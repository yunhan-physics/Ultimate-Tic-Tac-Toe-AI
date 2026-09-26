# -*- coding: utf-8 -*-
"""终极井字棋的 AlphaZero 风格策略价值网络。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from symmetry import inverse_policy, transform_spatial


ACTION_SIZE = 81


def _group_count(channels: int) -> int:
    """选择一个能整除通道数、且不超过 8 的 GroupNorm 分组数。"""
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock(nn.Module):
    """两个 3x3 卷积组成的残差块；使用适合单样本推理的 GroupNorm。"""

    def __init__(self, channels: int):
        super().__init__()
        groups = _group_count(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.norm1(self.conv1(x)), inplace=False)
        x = self.norm2(self.conv2(x))
        return F.relu(x + residual, inplace=False)


class UltimateNet(nn.Module):
    """输入 ``(B, 6, 9, 9)``，输出 81 个策略 logit 和一个价值。"""

    def __init__(
        self,
        input_channels: int = 6,
        channels: int = 64,
        blocks: int = 6,
        value_hidden: int = 128,
    ):
        super().__init__()
        if min(input_channels, channels, blocks, value_hidden) <= 0:
            raise ValueError("网络各项尺寸必须为正整数")

        groups = _group_count(channels)
        self.config = {
            "input_channels": input_channels,
            "channels": channels,
            "blocks": blocks,
            "value_hidden": value_hidden,
        }
        self.d4_ensemble = False
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.ReLU(inplace=False),
        )
        self.body = nn.Sequential(*(ResidualBlock(channels) for _ in range(blocks)))

        # 策略头保留空间布局，最终每个全局格子对应一个 logit。
        self.policy_conv = nn.Conv2d(channels, 4, 1, bias=False)
        self.policy_norm = nn.GroupNorm(1, 4)
        self.policy_linear = nn.Linear(4 * 9 * 9, ACTION_SIZE)

        # 价值始终站在“当前轮到的一方”视角，范围限制在 [-1, 1]。
        self.value_conv = nn.Conv2d(channels, 4, 1, bias=False)
        self.value_norm = nn.GroupNorm(1, 4)
        self.value_linear1 = nn.Linear(4 * 9 * 9, value_hidden)
        self.value_linear2 = nn.Linear(value_hidden, 1)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # 两个最终输出层不接 ReLU，较小初始权重让随机网络的分布更平稳。
        nn.init.normal_(self.policy_linear.weight, std=0.01)
        nn.init.normal_(self.value_linear2.weight, std=0.01)

    def _forward_raw(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4 or x.shape[1:] != (self.config["input_channels"], 9, 9):
            raise ValueError(
                f"输入形状必须是 (B, {self.config['input_channels']}, 9, 9)，实际为 {tuple(x.shape)}"
            )
        features = self.body(self.stem(x))

        policy = F.relu(self.policy_norm(self.policy_conv(features)), inplace=False)
        policy_logits = self.policy_linear(policy.flatten(1))

        value = F.relu(self.value_norm(self.value_conv(features)), inplace=False)
        value = F.relu(self.value_linear1(value.flatten(1)), inplace=False)
        value = torch.tanh(self.value_linear2(value))
        return policy_logits, value

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.d4_ensemble or self.training:
            return self._forward_raw(x)
        if x.ndim != 4 or x.shape[1:] != (self.config["input_channels"], 9, 9):
            return self._forward_raw(x)
        batch = x.shape[0]
        transformed = torch.cat([transform_spatial(x, index) for index in range(8)])
        logits, values = self._forward_raw(transformed)
        logits = torch.stack([
            inverse_policy(logits[index * batch:(index + 1) * batch], index)
            for index in range(8)
        ]).mean(dim=0)
        values = values.reshape(8, batch, 1).mean(dim=0)
        return logits, values


def configure_inference(model: UltimateNet, payload: dict) -> UltimateNet:
    """Enable release-time symmetry averaging when recorded in a checkpoint."""
    mode = payload.get("inference_symmetry")
    if mode not in (None, "none", "d4"):
        raise ValueError(f"Unsupported inference symmetry: {mode!r}")
    model.d4_ensemble = mode == "d4"
    return model


def masked_policy(policy_logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """只在合法动作上 softmax；每个样本必须至少有一个合法动作。"""
    if policy_logits.ndim != 2 or policy_logits.shape[1] != ACTION_SIZE:
        raise ValueError("policy_logits 形状必须是 (B, 81)")
    if legal_mask.shape != policy_logits.shape:
        raise ValueError("legal_mask 必须与 policy_logits 同形状")
    mask = legal_mask.to(device=policy_logits.device, dtype=torch.bool)
    if not torch.all(mask.any(dim=1)):
        raise ValueError("每个非终局样本至少需要一个合法动作")

    masked_logits = policy_logits.masked_fill(~mask, torch.finfo(policy_logits.dtype).min)
    probabilities = F.softmax(masked_logits, dim=1)
    probabilities = probabilities * mask.to(probabilities.dtype)
    return probabilities / probabilities.sum(dim=1, keepdim=True)


def count_parameters(model: nn.Module) -> int:
    """统计可训练参数数量。"""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
