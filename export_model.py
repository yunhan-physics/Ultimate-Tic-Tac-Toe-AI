# -*- coding: utf-8 -*-
"""从可恢复训练检查点导出轻量推理模型。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch


def export_model(source: Path, destination: Path, evaluation: Path | None = None) -> None:
    payload = torch.load(source, map_location="cpu", weights_only=False)
    slim = {
        "format_version": 1,
        "model_config": payload["model_config"],
        "state_dict": payload["state_dict"],
        "iteration": payload.get("iteration", -1),
        "inference_symmetry": payload.get("inference_symmetry"),
        "source_checkpoint": str(source),
    }
    if evaluation is not None:
        slim["evaluation"] = json.loads(evaluation.read_text(encoding="utf-8"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(slim, temporary)
    os.replace(temporary, destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="导出不含优化器与回放缓冲的推理模型")
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--evaluation", type=Path)
    args = parser.parse_args()
    export_model(args.source, args.destination, args.evaluation)
