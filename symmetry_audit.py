"""Measure policy equivariance and value invariance on fixed legal states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate import load_model
from game import apply_move, encode_state, legal_moves, new_game
from symmetry import inverse_policy, transform_spatial


def sample_states(seed: int = 20260926) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    samples = [encode_state(new_game())]
    for _ in range(5):
        state = new_game()
        for ply in range(1, 50):
            state, done, _ = apply_move(state, int(rng.choice(legal_moves(state))))
            if ply in (1, 2, 4, 8, 16, 24, 32, 40, 48) and not done:
                samples.append(encode_state(state))
            if done:
                break
    return samples


def audit_model(model: torch.nn.Module, samples: list[np.ndarray]) -> dict:
    model.eval()
    differences = []
    for state in samples:
        tensor = torch.from_numpy(state).unsqueeze(0)
        orbit = torch.cat([transform_spatial(tensor, index) for index in range(8)])
        with torch.inference_mode():
            logits, values = model(orbit)
        aligned = torch.stack([
            inverse_policy(logits[index:index + 1], index)[0] for index in range(8)
        ])
        legal = tensor[0, 5].bool().reshape(81)
        probabilities = torch.softmax(aligned.masked_fill(~legal, -1e9), dim=-1)
        differences.append({
            "policy_l1": float((probabilities - probabilities.mean(dim=0)).abs().sum(dim=-1).mean()),
            "value_mae": float((values - values.mean()).abs().mean()),
        })
    return {
        "positions": len(samples),
        "mean_policy_l1": float(np.mean([item["policy_l1"] for item in differences])),
        "mean_value_mae": float(np.mean([item["value_mae"] for item in differences])),
        "opening": differences[0],
        "per_position": differences,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--raw", action="store_true", help="Measure the network without release-time D4 averaging")
    args = parser.parse_args()
    torch.set_num_threads(1)
    model, payload = load_model(args.checkpoint)
    if args.raw:
        model.d4_ensemble = False
    report = {
        "checkpoint": str(args.checkpoint),
        "version": payload.get("model_identity", {}).get("version"),
        "inference_symmetry": payload.get("inference_symmetry", "none"),
        "measured_mode": "raw" if args.raw else "release",
        **audit_model(model, sample_states()),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "per_position"}, indent=2))


if __name__ == "__main__":
    main()
