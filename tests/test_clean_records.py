# -*- coding: utf-8 -*-
"""Tests for record auditing and recoverable quarantine."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from game import EMPTY, P1, apply_move, legal_moves, new_game, terminal_info
from game_record import GameRecorder
from clean_records import audit_directory, quarantine_records


def _recorder(directory: Path, game_id: str) -> GameRecorder:
    return GameRecorder(
        directory,
        human_player=P1,
        checkpoint="best.pth",
        model_iteration=1,
        simulations=8,
        c_puct=1.5,
        seed=3,
        game_id=game_id,
        metadata={"mode": "match", "assisted": False},
    )


class CleanRecordTests(unittest.TestCase):
    def test_audit_separates_completed_incomplete_and_invalid(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            complete = _recorder(root, "complete")
            state = new_game()
            while legal_moves(state):
                move = legal_moves(state)[0]
                after, _, _ = apply_move(state, move)
                complete.record_move(
                    state, move, "human" if state.to_play == P1 else "ai", after
                )
                state = after
            _, winner = terminal_info(state)
            complete.finish(state, winner)

            incomplete = _recorder(root, "incomplete")
            state = new_game()
            after, _, _ = apply_move(state, 0)
            incomplete.record_move(state, 0, "human", after)
            incomplete.finish(after, EMPTY, status="aborted", reason="test")

            (root / "broken.json").write_text("{", encoding="utf-8")
            report = audit_directory(root)
            self.assertEqual(
                report["categories"],
                {"invalid": 1, "clean_match": 1, "incomplete": 1},
            )
            self.assertEqual(report["unassisted_human_benchmark"]["games"], 1)

            quarantine = root / "quarantine"
            moved = quarantine_records(report, quarantine)
            self.assertEqual(len(moved), 2)
            self.assertEqual(len(list(root.glob("*.json"))), 1)
            self.assertTrue((quarantine / "incomplete").is_dir())
            self.assertTrue((quarantine / "invalid" / "broken.json").is_file())

    def test_assisted_game_is_excluded_from_human_benchmark(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            recorder = GameRecorder(
                root,
                human_player=P1,
                checkpoint="best.pth",
                model_iteration=1,
                simulations=8,
                c_puct=1.5,
                seed=3,
                metadata={"mode": "coach", "assisted": True},
            )
            state = new_game()
            while legal_moves(state):
                move = legal_moves(state)[0]
                after, _, _ = apply_move(state, move)
                recorder.record_move(
                    state, move, "human" if state.to_play == P1 else "ai", after
                )
                state = after
            _, winner = terminal_info(state)
            recorder.finish(state, winner)

            report = audit_directory(root)
            self.assertEqual(report["categories"], {"assisted_completed": 1})
            self.assertEqual(report["unassisted_human_benchmark"]["games"], 0)

    def test_model_versions_are_benchmarked_separately(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            recorder = GameRecorder(
                root,
                human_player=P1,
                checkpoint="best.pth",
                model_iteration=45,
                simulations=128,
                c_puct=1.5,
                seed=4,
                metadata={
                    "mode": "match",
                    "assisted": False,
                    "model_identity": {"name": "Tic-Tac-Toe Sophon", "version": "v1.0"},
                },
            )
            state = new_game()
            while legal_moves(state):
                move = legal_moves(state)[0]
                after, _, _ = apply_move(state, move)
                recorder.record_move(
                    state, move, "human" if state.to_play == P1 else "ai", after
                )
                state = after
            _, winner = terminal_info(state)
            recorder.finish(state, winner)

            report = audit_directory(root)
            self.assertEqual(report["v1_release_human_benchmark"]["ai"]["games"], 1)
            self.assertEqual(report["model_benchmarks"][0]["model"]["version"], "v1.0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
