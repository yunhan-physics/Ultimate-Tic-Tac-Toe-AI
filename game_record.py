# -*- coding: utf-8 -*-
"""人机对局棋谱的记录、加载与严格回放。

棋谱使用可读的 JSON 格式。记录器始终从 :func:`game.new_game` 开始，
每次落子都会核对前后局面；回放时则重新执行全部动作，避免把损坏或被修改
的棋谱误当作有效训练数据。
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any
from uuid import uuid4

from game import (
    ANY_BOARD,
    EMPTY,
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


SCHEMA_VERSION = 1
RULESET_ID = "ultimate_tic_tac_toe_score_only_v1"
_VALID_ACTORS = {"human", "ai"}
_VALID_STATUSES = {"completed", "aborted"}


def _is_int(value: object) -> bool:
    """排除 ``bool``；JSON 中布尔值也是 Python 的整数子类。"""
    return isinstance(value, int) and not isinstance(value, bool)


def _utc_text(value: datetime | str | None) -> str:
    """把时间标准化为带 ``Z`` 后缀的 UTC ISO 8601 字符串。"""
    if value is None:
        moment = datetime.now(timezone.utc)
    elif isinstance(value, datetime):
        moment = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("started_at 不能为空")
        try:
            moment = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
        except ValueError as exc:
            raise ValueError("started_at 必须是有效的 ISO 8601 时间") from exc
    else:
        raise TypeError("started_at 必须是 datetime、ISO 8601 字符串或 None")

    # 无时区的测试时间按 UTC 解释；实际运行生成的时间始终带 UTC 时区。
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    moment = moment.astimezone(timezone.utc)
    return moment.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _state_to_dict(state: GameState) -> dict[str, Any]:
    """生成稳定、可 JSON 序列化的局面表示。"""
    return {
        "board": list(state.board),
        "small_boards": list(state.small_boards),
        "to_play": state.to_play,
        "next_board": state.next_board,
    }


def _state_from_dict(value: object, field: str) -> GameState:
    """从棋谱字段恢复局面，并把结构错误统一转换成 ValueError。"""
    if not isinstance(value, dict):
        raise ValueError(f"{field} 必须是对象")
    required = {"board", "small_boards", "to_play", "next_board"}
    if set(value) != required:
        raise ValueError(f"{field} 字段不完整或含未知字段")
    board = value["board"]
    small_boards = value["small_boards"]
    if not isinstance(board, list) or not all(_is_int(item) for item in board):
        raise ValueError(f"{field}.board 必须是整数数组")
    if not isinstance(small_boards, list) or not all(_is_int(item) for item in small_boards):
        raise ValueError(f"{field}.small_boards 必须是整数数组")
    if not _is_int(value["to_play"]) or not _is_int(value["next_board"]):
        raise ValueError(f"{field} 的 to_play 和 next_board 必须是整数")
    try:
        return GameState(
            board=tuple(board),
            small_boards=tuple(small_boards),
            to_play=value["to_play"],
            next_board=value["next_board"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} 不是有效局面：{exc}") from exc


def _canonical_bytes(record: Mapping[str, Any]) -> bytes:
    """生成摘要所用的确定性 JSON；摘要字段本身不参与计算。"""
    content = {key: value for key, value in record.items() if key != "integrity"}
    try:
        text = json.dumps(
            content,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"棋谱含不能序列化的值：{exc}") from exc
    return text.encode("utf-8")


def _digest(record: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(record)).hexdigest()


def _check_integrity(record: Mapping[str, Any]) -> None:
    integrity = record.get("integrity")
    if not isinstance(integrity, dict):
        raise ValueError("棋谱缺少 integrity 摘要")
    if integrity.get("algorithm") != "sha256":
        raise ValueError("棋谱使用了不支持的摘要算法")
    digest = integrity.get("digest")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("棋谱的 SHA-256 摘要格式无效")
    if not hmac.compare_digest(digest, _digest(record)):
        raise ValueError("棋谱内容摘要不匹配，文件可能已损坏或被修改")


def _expected_actor(player: int, human_player: int) -> str:
    return "human" if player == human_player else "ai"


def _move_entry(
    ply: int,
    state_before: GameState,
    move: int,
    actor: str,
    state_after: GameState,
) -> dict[str, Any]:
    row, column = divmod(move, 9)
    small_board, local_cell = action_to_parts(move)
    p1_score, p2_score = score(state_after)
    return {
        "ply": ply,
        "actor": actor,
        "player": state_before.to_play,
        "move": move,
        # 全局行列按人类习惯使用 1~9；内部动作、小棋盘和局部格使用 0 起始编号。
        "global_row": row + 1,
        "global_column": column + 1,
        "small_board": small_board,
        "local_cell": local_cell,
        "forced_board_before": state_before.next_board,
        "after": {
            "score": {"p1": p1_score, "p2": p2_score},
            "small_boards": list(state_after.small_boards),
        },
    }


class GameRecorder:
    """从空棋盘开始记录一局人机对战，并以原子方式保存 JSON。"""

    def __init__(
        self,
        directory: str | os.PathLike[str],
        *,
        human_player: int,
        checkpoint: str | os.PathLike[str],
        model_iteration: int | str | None,
        simulations: int,
        c_puct: float,
        seed: int,
        started_at: datetime | str | None = None,
        game_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if human_player not in (P1, P2):
            raise ValueError("human_player 必须是 P1 或 P2")
        if not _is_int(simulations) or simulations <= 0:
            raise ValueError("simulations 必须是正整数")
        if isinstance(c_puct, bool) or not isinstance(c_puct, (int, float)):
            raise ValueError("c_puct 必须是有限正数")
        c_puct = float(c_puct)
        if not math.isfinite(c_puct) or c_puct <= 0:
            raise ValueError("c_puct 必须是有限正数")
        if not _is_int(seed):
            raise ValueError("seed 必须是整数")
        if model_iteration is not None and not isinstance(model_iteration, (int, str)):
            raise ValueError("model_iteration 必须是整数、字符串或 None")
        if isinstance(model_iteration, bool):
            raise ValueError("model_iteration 不能是布尔值")

        identifier = uuid4().hex if game_id is None else str(game_id).strip()
        if not identifier:
            raise ValueError("game_id 不能为空")

        self.directory = Path(directory).expanduser()
        self.human_player = human_player
        self.checkpoint = str(checkpoint)
        self.model_iteration = model_iteration
        self.simulations = simulations
        self.c_puct = c_puct
        self.seed = seed
        self.started_at = _utc_text(started_at)
        self.game_id = identifier
        # Optional provenance (for example assisted web play) is covered by the checksum.
        self.metadata = json.loads(json.dumps(dict(metadata or {}), allow_nan=False))
        self.saved_path: Path | None = None
        self._state = new_game()
        self._moves: list[dict[str, Any]] = []

    @property
    def state(self) -> GameState:
        """返回记录器当前期望的局面，便于调用方诊断集成错误。"""
        return self._state

    def record_move(
        self,
        state_before: GameState,
        move: int,
        actor: str,
        state_after: GameState,
    ) -> None:
        """记录并验证一步棋；不接受断链、非法或伪造的前后局面。"""
        if self.saved_path is not None:
            raise RuntimeError("棋谱已经结束，不能继续记录落子")
        if actor not in _VALID_ACTORS:
            raise ValueError("actor 只能是 'human' 或 'ai'")
        if not isinstance(state_before, GameState) or not isinstance(state_after, GameState):
            raise TypeError("state_before 和 state_after 必须是 GameState")
        if state_before != self._state:
            raise ValueError("state_before 与棋谱当前局面不一致")
        expected_actor = _expected_actor(state_before.to_play, self.human_player)
        if actor != expected_actor:
            raise ValueError(
                f"actor 与执棋方不一致：玩家 {state_before.to_play} 应由 {expected_actor} 落子"
            )
        if not _is_int(move) or move not in legal_moves(state_before):
            raise ValueError(f"动作 {move!r} 在当前局面中不合法")
        expected_state, _, _ = apply_move(state_before, move)
        if state_after != expected_state:
            raise ValueError("state_after 与规则引擎计算出的局面不一致")

        self._moves.append(
            _move_entry(len(self._moves) + 1, state_before, int(move), actor, state_after)
        )
        self._state = state_after

    def finish(
        self,
        final_state: GameState,
        winner: int | None,
        *,
        status: str = "completed",
        reason: str | None = None,
    ) -> Path:
        """结束并保存棋谱，返回最终 JSON 路径。

        ``completed`` 必须对应规则引擎认可的终局和赢家；``aborted`` 表示
        玩家中途退出，因此没有赢家（winner 可传 None 或 EMPTY）。
        """
        if self.saved_path is not None:
            raise RuntimeError(f"棋谱已经保存到 {self.saved_path}")
        if not isinstance(final_state, GameState):
            raise TypeError("final_state 必须是 GameState")
        if final_state != self._state:
            raise ValueError("final_state 与最后记录的局面不一致")
        if status not in _VALID_STATUSES:
            raise ValueError("status 只能是 'completed' 或 'aborted'")
        if reason is not None and not isinstance(reason, str):
            raise ValueError("reason 必须是字符串或 None")

        done, rules_winner = terminal_info(final_state)
        if status == "completed":
            if not _is_int(winner) or winner not in (EMPTY, P1, P2):
                raise ValueError("completed 棋谱的 winner 必须是 EMPTY、P1 或 P2")
            if not done:
                raise ValueError("completed 棋谱的 final_state 必须是终局")
            if winner != rules_winner:
                raise ValueError("winner 与规则引擎计算结果不一致")
            stored_winner = winner
        else:
            if winner is not None and winner != EMPTY:
                raise ValueError("aborted 棋谱没有赢家，winner 必须是 None 或 EMPTY")
            # 文件内统一沿用规则引擎的 EMPTY=0 表示“没有赢家”。
            stored_winner = EMPTY

        p1_score, p2_score = score(final_state)
        if status == "aborted":
            outcome = "aborted"
        elif stored_winner == P1:
            outcome = "p1_win"
        elif stored_winner == P2:
            outcome = "p2_win"
        else:
            outcome = "draw"

        record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "ruleset": RULESET_ID,
            "game_id": self.game_id,
            "started_at": self.started_at,
            "finished_at": _utc_text(None),
            "status": status,
            "reason": reason,
            "human_player": self.human_player,
            "checkpoint": self.checkpoint,
            "model_iteration": self.model_iteration,
            "simulations": self.simulations,
            "c_puct": self.c_puct,
            "seed": self.seed,
            "initial_state": _state_to_dict(new_game()),
            "move_count": len(self._moves),
            "moves": list(self._moves),
            "final_state": _state_to_dict(final_state),
            "result": {
                "winner": stored_winner,
                "outcome": outcome,
                "terminal": done,
                "score": {"p1": p1_score, "p2": p2_score},
            },
        }
        if self.metadata:
            record["metadata"] = self.metadata
        record["integrity"] = {"algorithm": "sha256", "digest": _digest(record)}

        path = self._write_atomic(record)
        self.saved_path = path
        return path

    def _write_atomic(self, record: Mapping[str, Any]) -> Path:
        """在目标目录内写临时文件，再原子替换为唯一的正式文件。"""
        self.directory.mkdir(parents=True, exist_ok=True)
        safe_id = re.sub(r"[^0-9A-Za-z._-]+", "-", self.game_id).strip(".-") or "game"
        stamp = self.started_at.replace("-", "").replace(":", "").replace(".", "")
        stamp = stamp.replace("+", "").replace("Z", "Z")
        # 随机后缀保证即使注入相同 game_id 和 started_at，也不会覆盖旧棋谱。
        destination = self.directory / f"human_{stamp}_{safe_id}_{uuid4().hex[:10]}.json"
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=self.directory,
                prefix=f".{destination.stem}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                json.dump(record, handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            return destination
        except BaseException:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
            raise


def load_record(path: str | os.PathLike[str]) -> dict[str, Any]:
    """加载棋谱并校验格式版本与内容摘要。"""
    source = Path(path).expanduser()
    try:
        with source.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取棋谱 {source}：{exc}") from exc
    if not isinstance(record, dict):
        raise ValueError("棋谱根节点必须是 JSON 对象")
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"不支持的棋谱 schema_version：{record.get('schema_version')!r}")
    _check_integrity(record)
    return record


def replay_record(record_or_path: Mapping[str, Any] | str | os.PathLike[str]) -> GameState:
    """严格重放棋谱，成功时返回最终 :class:`GameState`。

    除动作合法性外，本函数还核对每步派生字段、执棋者、最终状态与赢家。
    任一不一致都会抛出 ``ValueError``。
    """
    if isinstance(record_or_path, (str, os.PathLike)):
        record = load_record(record_or_path)
    elif isinstance(record_or_path, Mapping):
        record = dict(record_or_path)
        if record.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"不支持的棋谱 schema_version：{record.get('schema_version')!r}")
        _check_integrity(record)
    else:
        raise TypeError("record_or_path 必须是棋谱路径或 Mapping")

    human_player = record.get("human_player")
    if record.get("ruleset") != RULESET_ID:
        raise ValueError(f"棋谱规则版本不匹配：{record.get('ruleset')!r}")
    if not _is_int(human_player) or human_player not in (P1, P2):
        raise ValueError("human_player 必须是 P1 或 P2")
    if record.get("initial_state") != _state_to_dict(new_game()):
        raise ValueError("initial_state 不是标准空棋盘")

    moves = record.get("moves")
    if not isinstance(moves, list):
        raise ValueError("moves 必须是数组")
    move_count = record.get("move_count")
    if not _is_int(move_count) or move_count != len(moves):
        raise ValueError("move_count 与 moves 长度不一致")

    state = new_game()
    for index, entry in enumerate(moves, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"第 {index} 手记录必须是对象")
        if entry.get("ply") != index:
            raise ValueError(f"第 {index} 手的 ply 不连续")
        actor = entry.get("actor")
        expected_actor = _expected_actor(state.to_play, human_player)
        if actor not in _VALID_ACTORS or actor != expected_actor:
            raise ValueError(f"第 {index} 手 actor 与执棋方不一致")
        if entry.get("player") != state.to_play:
            raise ValueError(f"第 {index} 手 player 与当前行动方不一致")

        move = entry.get("move")
        if not _is_int(move) or move not in legal_moves(state):
            raise ValueError(f"第 {index} 手动作 {move!r} 不合法")
        row, column = divmod(move, 9)
        small_board, local_cell = action_to_parts(move)
        derived = {
            "global_row": row + 1,
            "global_column": column + 1,
            "small_board": small_board,
            "local_cell": local_cell,
            "forced_board_before": state.next_board,
        }
        for field, expected in derived.items():
            if entry.get(field) != expected:
                raise ValueError(f"第 {index} 手的 {field} 与动作或局面不一致")

        state_after, _, _ = apply_move(state, move)
        p1_score, p2_score = score(state_after)
        expected_after = {
            "score": {"p1": p1_score, "p2": p2_score},
            "small_boards": list(state_after.small_boards),
        }
        if entry.get("after") != expected_after:
            raise ValueError(f"第 {index} 手的落子后状态摘要不一致")
        state = state_after

    final_state = _state_from_dict(record.get("final_state"), "final_state")
    if final_state != state:
        raise ValueError("final_state 与动作回放结果不一致")

    status = record.get("status")
    if status not in _VALID_STATUSES:
        raise ValueError("status 只能是 'completed' 或 'aborted'")
    result = record.get("result")
    if not isinstance(result, dict):
        raise ValueError("result 必须是对象")
    winner = result.get("winner")
    if not _is_int(winner) or winner not in (EMPTY, P1, P2):
        raise ValueError("result.winner 无效")
    done, rules_winner = terminal_info(state)
    p1_score, p2_score = score(state)
    if status == "completed":
        if not done or winner != rules_winner:
            raise ValueError("completed 棋谱的终局或赢家不正确")
        outcome = "p1_win" if winner == P1 else "p2_win" if winner == P2 else "draw"
    else:
        if winner != EMPTY:
            raise ValueError("aborted 棋谱不能记录赢家")
        outcome = "aborted"
    expected_result = {
        "winner": winner,
        "outcome": outcome,
        "terminal": done,
        "score": {"p1": p1_score, "p2": p2_score},
    }
    if result != expected_result:
        raise ValueError("result 与最终局面不一致")
    return state


__all__ = ["GameRecorder", "RULESET_ID", "SCHEMA_VERSION", "load_record", "replay_record"]
