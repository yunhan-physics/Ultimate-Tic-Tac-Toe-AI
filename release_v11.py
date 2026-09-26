"""Promote Sophon v1.1 only after strength and symmetry gates pass."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from league import _atomic_json_write, export_champion, promotion_decision


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def assess_release(league: dict, baseline: dict, raw: dict, ensemble: dict) -> list[str]:
    """Return unmet requirements without changing any model files."""
    match = league["match"]
    failures = []
    if match["opening_pairs"] < 200 or match["games"] < 400:
        failures.append("The paired league needs at least 200 openings and 400 games.")
    if match["protocol"]["simulations_per_move_each"] < 96:
        failures.append("Both models need at least 96 simulations per move.")
    if not promotion_decision(match)["qualified"]:
        failures.append("The score-rate and paired confidence-interval gates did not pass.")
    if raw["mean_policy_l1"] >= baseline["mean_policy_l1"]:
        failures.append("The raw network policy symmetry did not improve.")
    if raw["mean_value_mae"] >= baseline["mean_value_mae"]:
        failures.append("The raw network value symmetry did not improve.")
    if ensemble["mean_policy_l1"] >= 1e-5 or ensemble["mean_value_mae"] >= 1e-5:
        failures.append("Release-time D4 symmetry is outside numerical tolerance.")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, default=Path("checkpoints/candidate_v11/trained.pth"))
    parser.add_argument("--champion", type=Path, default=Path("checkpoints/best.pth"))
    parser.add_argument("--named", type=Path, default=Path("checkpoints/releases/Tic-Tac-Toe_Sophon_v1.1.pth"))
    parser.add_argument("--league", type=Path, default=Path("checkpoints/candidate_v11/league_vs_v1.json"))
    parser.add_argument("--baseline", type=Path, default=Path("checkpoints/candidate_v11/v1_baseline_symmetry.json"))
    parser.add_argument("--raw", type=Path, default=Path("checkpoints/candidate_v11/v11_raw_symmetry.json"))
    parser.add_argument("--ensemble", type=Path, default=Path("checkpoints/candidate_v11/v11_ensemble_symmetry.json"))
    parser.add_argument("--report", type=Path, default=Path("checkpoints/releases/v1.1_release_report.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.named.exists() or args.report.exists():
        raise FileExistsError("The named v1.1 release or report already exists.")
    candidate = torch.load(args.candidate, map_location="cpu", weights_only=False)
    champion = torch.load(args.champion, map_location="cpu", weights_only=False)
    if candidate.get("model_identity", {}).get("version") != "v1.1":
        raise ValueError("Candidate checkpoint is not Sophon v1.1.")
    if candidate.get("inference_symmetry") != "d4":
        raise ValueError("Candidate checkpoint does not enable D4 inference.")
    if champion.get("model_identity", {}).get("version") != "v1.0":
        raise ValueError("Current champion is not the expected v1.0 release.")

    league = read_json(args.league)
    baseline = read_json(args.baseline)
    raw = read_json(args.raw)
    ensemble = read_json(args.ensemble)
    failures = assess_release(league, baseline, raw, ensemble)
    if failures:
        print(json.dumps({"qualified": False, "failures": failures}, indent=2))
        return 1

    history = candidate["history"]
    new_history = [entry for entry in history if 46 <= entry["iteration"] <= 55]
    if len(new_history) != 10 or sum(entry["games"] for entry in new_history) != 200:
        raise ValueError("The v1.1 checkpoint does not contain the planned 200 new self-play games.")
    release_report = {
        "model": "Tic-Tac-Toe Sophon",
        "version": "v1.1",
        "released_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent": "v1.0",
        "self_play_games_added": 200,
        "self_play_examples_added": sum(entry["examples"] for entry in new_history),
        "training_updates_added": len(new_history) * int(candidate["args"]["train_steps"]),
        "inference_symmetry": "d4",
        "symmetry": {"baseline": baseline, "trained_raw": raw, "released_ensemble": ensemble},
        "league_vs_v1.0": league["match"],
        "promotion": {"qualified": True, "promoted": not args.dry_run},
    }
    if args.dry_run:
        print(json.dumps({key: value for key, value in release_report.items()
                          if key not in ("symmetry", "league_vs_v1.0")}, indent=2))
        return 0

    export_champion(args.candidate, args.named, league["match"])
    export_champion(args.candidate, args.champion, league["match"])
    _atomic_json_write(release_report, args.report)
    print(f"Promoted Sophon v1.1 to {args.named} and {args.champion}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
