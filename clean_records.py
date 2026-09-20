# -*- coding: utf-8 -*-
"""Audit human game records and quarantine data that must not enter training.

The command is deliberately conservative: completed assisted games remain
available for qualitative analysis, while aborted or invalid files can be
moved to a recoverable quarantine directory.  No record is ever deleted.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Any

from game import EMPTY
from game_record import load_record, replay_record


@dataclass(frozen=True, slots=True)
class RecordAudit:
    path: str
    category: str
    status: str | None = None
    reason: str | None = None
    moves: int | None = None
    mode: str | None = None
    assisted: bool | None = None
    human_outcome: str | None = None
    checkpoint: str | None = None
    model_iteration: int | str | None = None
    model_name: str | None = None
    model_version: str | None = None
    error: str | None = None


def audit_record(path: str | os.PathLike[str]) -> RecordAudit:
    """Strictly validate one record and classify its fitness for evaluation."""
    source = Path(path)
    try:
        record = load_record(source)
        replay_record(record)
    except (OSError, TypeError, ValueError) as exc:
        return RecordAudit(str(source), "invalid", error=str(exc))

    status = record["status"]
    metadata = record.get("metadata", {})
    identity = metadata.get("model_identity", {})
    mode = metadata.get("mode")
    assisted = bool(metadata.get("assisted", False))
    human_outcome = None
    if status == "completed":
        winner = record["result"]["winner"]
        human = record["human_player"]
        human_outcome = "draw" if winner == EMPTY else "win" if winner == human else "loss"

    if status != "completed":
        category = "incomplete"
    elif assisted or mode == "coach":
        category = "assisted_completed"
    else:
        category = "clean_match"
    return RecordAudit(
        path=str(source),
        category=category,
        status=status,
        reason=record.get("reason"),
        moves=int(record["move_count"]),
        mode=mode,
        assisted=assisted,
        human_outcome=human_outcome,
        checkpoint=record.get("checkpoint"),
        model_iteration=record.get("model_iteration"),
        model_name=identity.get("name") if isinstance(identity, dict) else None,
        model_version=identity.get("version") if isinstance(identity, dict) else None,
    )


def _benchmark(records: list[RecordAudit]) -> dict[str, Any]:
    human = {"wins": 0, "losses": 0, "draws": 0}
    for record in records:
        key = {"win": "wins", "loss": "losses", "draw": "draws"}[record.human_outcome]
        human[key] += 1
    games = len(records)
    human.update(
        games=games,
        non_loss_rate=(human["wins"] + human["draws"]) / games if games else None,
    )
    ai = {
        "wins": human["losses"],
        "losses": human["wins"],
        "draws": human["draws"],
        "games": games,
        "non_loss_rate": (human["losses"] + human["draws"]) / games if games else None,
        "minimum_games_for_v1_gate": 20,
        "enough_games_for_v1_gate": games >= 20,
    }
    return {"human": human, "ai": ai}


def audit_directory(directory: str | os.PathLike[str]) -> dict[str, Any]:
    """Audit top-level JSON records in a directory and return a stable report."""
    root = Path(directory).expanduser().resolve()
    records = [audit_record(path) for path in sorted(root.glob("*.json"))]
    categories: dict[str, int] = {}
    for record in records:
        categories[record.category] = categories.get(record.category, 0) + 1

    clean_matches = [record for record in records if record.category == "clean_match"]
    overall = _benchmark(clean_matches)
    grouped: dict[tuple[str | None, str | None, int | str | None, str | None], list[RecordAudit]] = {}
    for record in clean_matches:
        key = (record.model_name, record.model_version, record.model_iteration, record.checkpoint)
        grouped.setdefault(key, []).append(record)
    model_benchmarks = []
    for (name, version, iteration, checkpoint), group in grouped.items():
        model_benchmarks.append(
            {
                "model": {
                    "name": name,
                    "version": version,
                    "iteration": iteration,
                    "checkpoint": checkpoint,
                },
                **_benchmark(group),
            }
        )
    v1_records = [record for record in clean_matches if record.model_version == "v1.0"]
    return {
        "schema_version": 1,
        "source_directory": str(root),
        "total_files": len(records),
        "categories": categories,
        "unassisted_human_benchmark": overall["human"],
        "unassisted_ai_benchmark": overall["ai"],
        "model_benchmarks": model_benchmarks,
        "v1_release_human_benchmark": _benchmark(v1_records),
        "records": [asdict(record) for record in records],
    }


def quarantine_records(report: dict[str, Any], quarantine_dir: str | os.PathLike[str]) -> list[str]:
    """Move incomplete/invalid records into category folders without overwriting."""
    source_root = Path(report["source_directory"]).resolve()
    target_root = Path(quarantine_dir).expanduser().resolve()
    if target_root == source_root or source_root in target_root.parents:
        pass
    elif target_root in source_root.parents:
        raise ValueError("quarantine directory cannot contain the source directory")

    moved: list[str] = []
    for item in report["records"]:
        category = item["category"]
        if category not in {"incomplete", "invalid"}:
            continue
        source = Path(item["path"]).resolve()
        if source.parent != source_root:
            raise ValueError(f"record is outside the audited directory: {source}")
        destination = target_root / category / source.name
        if destination.exists():
            raise FileExistsError(f"quarantine destination already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, destination)
        moved.append(str(destination))
    return moved


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="严格审计真人棋谱，并可恢复地隔离未完成/损坏文件")
    parser.add_argument("directory", nargs="?", type=Path, default=Path("records/human_games"))
    parser.add_argument("--report", type=Path, default=Path("records/audit_report.json"))
    parser.add_argument(
        "--quarantine",
        type=Path,
        default=None,
        help="移动 incomplete/invalid 文件到该目录；省略时只审计，不改动棋谱",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = audit_directory(args.directory)
    if args.quarantine is not None:
        report["quarantined"] = quarantine_records(report, args.quarantine)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["categories"].get("invalid", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
