# -*- coding: utf-8 -*-
"""人机棋谱模块的回归测试。"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from game import ANY_BOARD, EMPTY, P1, P2, GameState, apply_move, legal_moves, new_game, terminal_info
from game_record import GameRecorder, load_record, replay_record


FIXED_TIME = datetime(2026, 9, 16, 8, 30, tzinfo=timezone.utc)


def recorder(directory: Path, *, human_player: int = P1, game_id: str = "test-game") -> GameRecorder:
    """创建参数固定、便于断言的测试记录器。"""
    return GameRecorder(
        directory,
        human_player=human_player,
        checkpoint="checkpoints/best.pth",
        model_iteration=760,
        simulations=48,
        c_puct=1.5,
        seed=20260916,
        started_at=FIXED_TIME,
        game_id=game_id,
    )


def record_moves(value: GameRecorder, moves: list[int]) -> GameState:
    """按规则执行并记录给定动作序列。"""
    state = new_game()
    for move in moves:
        actor = "human" if state.to_play == value.human_player else "ai"
        next_state, _, _ = apply_move(state, move)
        value.record_move(state, move, actor, next_state)
        state = next_state
    return state


class GameRecordTests(unittest.TestCase):
    def test_completed_game_round_trip(self):
        """完成棋谱包含元数据，并可严格回放到相同终局。"""
        with TemporaryDirectory() as temporary:
            target = Path(temporary)
            value = recorder(target)
            # 每次选编号最小的合法动作，可确定性地产生一局 64 手的完整和棋。
            state = new_game()
            while legal_moves(state):
                move = legal_moves(state)[0]
                actor = "human" if state.to_play == value.human_player else "ai"
                next_state, _, _ = apply_move(state, move)
                value.record_move(state, move, actor, next_state)
                state = next_state
            done, winner = terminal_info(state)
            self.assertTrue(done)
            path = value.finish(state, winner)

            self.assertTrue(path.is_file())
            data = load_record(path)
            self.assertEqual(data["schema_version"], 1)
            self.assertEqual(data["ruleset"], "ultimate_tic_tac_toe_score_only_v1")
            self.assertEqual(data["game_id"], "test-game")
            self.assertEqual(data["started_at"], "2026-09-16T08:30:00.000000Z")
            self.assertEqual(data["human_player"], P1)
            self.assertEqual(data["checkpoint"], "checkpoints/best.pth")
            self.assertEqual(data["model_iteration"], 760)
            self.assertEqual(data["simulations"], 48)
            self.assertEqual(data["c_puct"], 1.5)
            self.assertEqual(data["seed"], 20260916)
            self.assertEqual(data["status"], "completed")
            self.assertEqual(data["result"]["outcome"], "draw")
            self.assertEqual(replay_record(path), state)

    def test_aborted_game_can_be_replayed(self):
        """中途退出会保留已下步骤和可复现的当前局面。"""
        with TemporaryDirectory() as temporary:
            value = recorder(Path(temporary))
            # 0 把对手送往左上盘；1 是该盘内合法位置。
            final_state = record_moves(value, [0, 1])
            # play.py 在用户退出时传 None；记录文件会统一写为 EMPTY=0。
            path = value.finish(final_state, None, status="aborted", reason="user_quit")

            data = load_record(path)
            self.assertEqual(data["status"], "aborted")
            self.assertEqual(data["reason"], "user_quit")
            self.assertEqual(data["result"]["winner"], EMPTY)
            self.assertEqual(data["move_count"], 2)
            first = data["moves"][0]
            self.assertEqual(first["actor"], "human")
            self.assertEqual(first["player"], P1)
            self.assertEqual((first["global_row"], first["global_column"]), (1, 1))
            self.assertEqual((first["small_board"], first["local_cell"]), (0, 0))
            self.assertEqual(first["forced_board_before"], ANY_BOARD)
            self.assertEqual(first["after"]["score"], {"p1": 0, "p2": 0})
            self.assertEqual(replay_record(path), final_state)

    def test_tampering_is_rejected(self):
        """即使 JSON 仍可解析，修改一步动作也会被摘要检查拒绝。"""
        with TemporaryDirectory() as temporary:
            target = Path(temporary)
            value = recorder(target)
            final_state = record_moves(value, [0, 1])
            path = value.finish(final_state, EMPTY, status="aborted")

            data = json.loads(path.read_text(encoding="utf-8"))
            data["moves"][0]["move"] = 80
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "摘要"):
                replay_record(path)

    def test_atomic_write_leaves_no_temporary_file_and_names_are_unique(self):
        """同一局号重复保存不覆盖旧文件，也不遗留 .tmp。"""
        with TemporaryDirectory() as temporary:
            target = Path(temporary)
            paths: list[Path] = []
            for _ in range(2):
                value = recorder(target)
                state = record_moves(value, [0])
                paths.append(value.finish(state, EMPTY, status="aborted"))

            self.assertNotEqual(paths[0], paths[1])
            self.assertTrue(all(path.is_file() for path in paths))
            self.assertEqual(list(target.glob("*.tmp")), [])
            self.assertEqual(list(target.glob(".*.tmp")), [])

    def test_record_move_rejects_wrong_actor_and_state(self):
        with TemporaryDirectory() as temporary:
            value = recorder(Path(temporary))
            before = new_game()
            after, _, _ = apply_move(before, 0)
            with self.assertRaisesRegex(ValueError, "actor"):
                value.record_move(before, 0, "ai", after)
            with self.assertRaisesRegex(ValueError, "state_after"):
                value.record_move(before, 0, "human", before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
