# -*- coding: utf-8 -*-
"""本地网页服务的状态机、棋谱持久化和 HTTP 安全边界测试。"""

from __future__ import annotations

from http.client import HTTPConnection
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.parse import quote

from game import legal_moves, terminal_info
from game_record import load_record, replay_record
from web_server import GameService, LocalServer, WEB_ROOT


class _DummyModel:
    """搜索在本测试中被替换，因此模型只充当服务依赖。"""


def _choose_first_legal(model, state, simulations, rng, **kwargs):
    del model, simulations, rng, kwargs
    return legal_moves(state)[0]


class WebServerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.records = self.root / "records"
        self.service = GameService(
            _DummyModel(),
            {"iteration": 760},
            checkpoint=self.root / "champion.pth",
            records_dir=self.records,
            simulations=3,
            analysis_simulations=2,
            c_puct=1.25,
        )
        self.ai_patch = patch("web_server.choose_ai_action", side_effect=_choose_first_legal)
        self.ai_patch.start()
        self.server = LocalServer(("127.0.0.1", 0), self.service)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.ai_patch.stop()
        self.temporary.cleanup()

    def request(self, method, path, body=None, headers=None):
        headers = dict(headers or {})
        if isinstance(body, (dict, list, int, float, bool)) or body is None and method == "POST":
            body = json.dumps(body).encode("utf-8")
            headers.setdefault("Content-Type", "application/json")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        connection = HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            content = response.read()
            response_headers = {name.lower(): value for name, value in response.getheaders()}
            content_type = response_headers.get("content-type", "")
            value = json.loads(content) if "application/json" in content_type else content
            return response.status, response_headers, value
        finally:
            connection.close()

    def new_game(self, *, human_player=1, mode="match", previous_session=None):
        body = {"human_player": human_player, "mode": mode}
        if previous_session is not None:
            body["previous_session"] = previous_session
        status, _, value = self.request("POST", "/api/new", body)
        self.assertEqual(status, 200, value)
        return value

    @staticmethod
    def signature(session):
        record = session.record_path.read_bytes() if session.record_path else None
        return (
            session.state,
            session.revision,
            session.aborted,
            tuple(tuple(sorted(item.items())) for item in session.history),
            record,
        )

    def assert_saved_snapshot(self, identifier):
        session = self.service.session(identifier)
        self.assertIsNotNone(session.record_path)
        self.assertTrue(session.record_path.is_file())
        self.assertEqual(replay_record(session.record_path), session.state)
        self.assertEqual(
            list(self.records.glob(f"web_{identifier}.json")),
            [session.record_path],
        )
        self.assertFalse(list(self.records.glob("*.tmp")))
        self.assertFalse(list(self.records.glob(".*.tmp")))

    def test_first_player_flow_and_refreshed_state(self):
        view = self.new_game(human_player=1)
        identifier = view["session"]
        human_action = view["legal_moves"][40]
        status, _, after_human = self.request(
            "POST", "/api/move",
            {"session": identifier, "revision": view["revision"], "action": human_action},
        )
        self.assertEqual(status, 200)
        self.assertEqual(after_human["revision"], 1)
        self.assertEqual(after_human["history"][-1]["actor"], "human")

        status, _, after_ai = self.request(
            "POST", "/api/ai",
            {"session": identifier, "revision": after_human["revision"]},
        )
        self.assertEqual(status, 200)
        self.assertEqual([move["actor"] for move in after_ai["history"]], ["human", "ai"])
        self.assertEqual(after_ai["to_play"], 1)

        status, _, refreshed = self.request(
            "GET", f"/api/state?session={quote(identifier)}"
        )
        self.assertEqual(status, 200)
        self.assertEqual(refreshed, after_ai)
        self.assert_saved_snapshot(identifier)

    def test_second_player_flow_starts_with_ai(self):
        view = self.new_game(human_player=2)
        identifier = view["session"]
        session = self.service.session(identifier)
        before = self.signature(session)

        status, _, error = self.request(
            "POST", "/api/move",
            {"session": identifier, "revision": 0, "action": view["legal_moves"][0]},
        )
        self.assertEqual(status, 409)
        self.assertIn("turn", error["error"].lower())
        self.assertEqual(self.signature(session), before)

        status, _, after_ai = self.request(
            "POST", "/api/ai", {"session": identifier, "revision": 0}
        )
        self.assertEqual(status, 200)
        self.assertEqual(after_ai["history"][0]["actor"], "ai")
        self.assertEqual(after_ai["to_play"], 2)
        action = after_ai["legal_moves"][0]
        status, _, after_human = self.request(
            "POST", "/api/move",
            {"session": identifier, "revision": 1, "action": action},
        )
        self.assertEqual(status, 200)
        self.assertEqual([move["actor"] for move in after_human["history"]], ["ai", "human"])
        self.assert_saved_snapshot(identifier)

    def test_wrong_turn_illegal_and_stale_requests_do_not_mutate_state(self):
        view = self.new_game(human_player=1)
        identifier = view["session"]
        session = self.service.session(identifier)

        for endpoint, body, expected_status in [
            ("/api/ai", {"session": identifier, "revision": 0}, 409),
            ("/api/move", {"session": identifier, "revision": 0, "action": 81}, 400),
            ("/api/move", {"session": identifier, "revision": 0, "action": True}, 400),
        ]:
            with self.subTest(endpoint=endpoint, body=body):
                before = self.signature(session)
                status, _, _ = self.request("POST", endpoint, body)
                self.assertEqual(status, expected_status)
                self.assertEqual(self.signature(session), before)

        status, _, current = self.request(
            "POST", "/api/move",
            {"session": identifier, "revision": 0, "action": view["legal_moves"][0]},
        )
        self.assertEqual(status, 200)

        for body in [
            {"session": identifier, "revision": 0, "action": current["legal_moves"][0]},
            {"session": identifier, "revision": True, "action": current["legal_moves"][0]},
            {"session": identifier, "revision": 1, "action": current["legal_moves"][0]},
        ]:
            with self.subTest(body=body):
                before = self.signature(session)
                status, _, _ = self.request("POST", "/api/move", body)
                self.assertEqual(status, 409)
                self.assertEqual(self.signature(session), before)

    def test_each_accepted_move_replaces_one_replayable_snapshot(self):
        view = self.new_game(human_player=1)
        identifier = view["session"]
        self.assert_saved_snapshot(identifier)
        path = self.service.session(identifier).record_path
        self.assertEqual(len(list(self.records.glob("*.json"))), 1)

        for _ in range(8):
            session = self.service.session(identifier)
            if session.state.to_play == session.human_player:
                view = self.service.move(
                    {"session": identifier, "revision": session.revision,
                     "action": legal_moves(session.state)[0]}
                )
            else:
                view = self.service.move(
                    {"session": identifier, "revision": session.revision}, ai=True
                )
            self.assertEqual(self.service.session(identifier).record_path, path)
            self.assertEqual(view["ply"], self.service.session(identifier).revision)
            self.assert_saved_snapshot(identifier)
            self.assertEqual(len(list(self.records.glob("*.json"))), 1)

        status, headers, downloaded = self.request(
            "GET", f"/api/record?session={quote(identifier)}"
        )
        self.assertEqual(status, 200)
        self.assertEqual(downloaded, json.loads(path.read_text(encoding="utf-8")))
        self.assertIn(path.name, headers["content-disposition"])

    def test_full_game_is_saved_completed_and_replayable(self):
        view = self.new_game(human_player=1)
        identifier = view["session"]
        while not view["done"]:
            session = self.service.session(identifier)
            request = {"session": identifier, "revision": session.revision}
            if session.state.to_play == session.human_player:
                request["action"] = legal_moves(session.state)[0]
                view = self.service.move(request)
            else:
                view = self.service.move(request, ai=True)

        session = self.service.session(identifier)
        self.assertTrue(terminal_info(session.state)[0])
        record = load_record(session.record_path)
        self.assertEqual(record["status"], "completed")
        self.assertIsNone(record["reason"])
        self.assertTrue(record["result"]["terminal"])
        self.assertEqual(replay_record(record), session.state)
        self.assertEqual(len(list(self.records.glob("*.json"))), 1)

    def test_restart_and_abort_save_distinct_replayable_games(self):
        first = self.new_game(human_player=1)
        old_id = first["session"]
        first = self.service.move(
            {"session": old_id, "revision": 0, "action": first["legal_moves"][0]}
        )
        first = self.service.move(
            {"session": old_id, "revision": first["revision"]}, ai=True
        )
        old_state = self.service.session(old_id).state

        second = self.new_game(human_player=2, mode="coach", previous_session=old_id)
        new_id = second["session"]
        old = self.service.session(old_id)
        self.assertTrue(old.aborted)
        old_record = load_record(old.record_path)
        self.assertEqual(old_record["status"], "aborted")
        self.assertEqual(old_record["reason"], "web_restart")
        self.assertEqual(replay_record(old_record), old_state)

        status, _, ended = self.request(
            "POST", "/api/abort",
            {"session": new_id, "revision": second["revision"]},
        )
        self.assertEqual(status, 200)
        self.assertTrue(ended["aborted"])
        new = self.service.session(new_id)
        new_record = load_record(new.record_path)
        self.assertEqual(new_record["status"], "aborted")
        self.assertEqual(new_record["reason"], "web_user_end")
        self.assertEqual(replay_record(new_record), new.state)
        self.assertEqual(len(list(self.records.glob("*.json"))), 2)

    def test_coach_metadata_and_analysis_do_not_change_game_state(self):
        view = self.new_game(human_player=1, mode="coach")
        identifier = view["session"]
        session = self.service.session(identifier)
        record = load_record(session.record_path)
        self.assertEqual(
            record["metadata"],
            {"interface": "web", "mode": "coach", "assisted": True},
        )
        before = self.signature(session)
        analysis = {
            "human_rate": 62.5,
            "ai_rate": 37.5,
            "top_moves": [{"action": view["legal_moves"][0], "human_rate": 62.5}],
            "moves": [],
            "metric": "expected_score",
            "method": "test estimate",
        }
        with patch("web_server.analyze_position", return_value=analysis) as analyzer:
            status, _, result = self.request(
                "POST", "/api/analysis",
                {"session": identifier, "revision": view["revision"]},
            )
            self.assertEqual(status, 200)
            self.assertEqual(result["human_rate"] + result["ai_rate"], 100.0)
            self.assertEqual(result["session"], identifier)
            self.assertEqual(result["revision"], 0)
            self.assertEqual(self.signature(session), before)
            self.assertTrue(analyzer.call_args.kwargs["include_moves"])

            # 同一修订号重复请求命中会话缓存，不再次运行分析。
            status, _, repeated = self.request(
                "POST", "/api/analysis",
                {"session": identifier, "revision": view["revision"]},
            )
            self.assertEqual(status, 200)
            self.assertEqual(repeated, result)
            self.assertEqual(analyzer.call_count, 1)

    def test_info_endpoint_and_security_headers(self):
        status, headers, info = self.request("GET", "/api/info")
        self.assertEqual(status, 200)
        self.assertEqual(info["model"], {"name": "champion.pth", "iteration": 760})
        self.assertEqual(info["simulations"], 3)
        self.assertEqual(info["analysis_simulations"], 2)
        self.assertIn("ultimate_tic_tac_toe", info["ruleset"])
        self.assertEqual(headers["x-content-type-options"], "nosniff")
        self.assertEqual(headers["cache-control"], "no-store")
        self.assertIn("frame-ancestors 'none'", headers["content-security-policy"])

    def test_info_prefers_versioned_display_identity(self):
        self.service.payload["model_identity"] = {
            "name": "Tic-Tac-Toe Sophon",
            "version": "v1.0",
        }
        status, _, info = self.request("GET", "/api/info")
        self.assertEqual(status, 200)
        self.assertEqual(info["model"]["name"], "champion.pth")
        self.assertEqual(info["model"]["display_name"], "Tic-Tac-Toe Sophon")
        self.assertEqual(info["model"]["version"], "v1.0")

    def test_static_routes_serve_only_real_frontend_assets(self):
        names = ("index.html", "app.css", "app.js")
        missing = [name for name in names if not (WEB_ROOT / name).is_file()]
        if missing:
            self.skipTest("前端资源仍在并行构建：" + ", ".join(missing))

        status, _, root = self.request("GET", "/")
        self.assertEqual(status, 200)
        status, html_headers, index = self.request("GET", "/index.html")
        self.assertEqual(status, 200)
        self.assertEqual(root, index)
        self.assertIn("text/html", html_headers["content-type"])
        self.assertIn(b"app.css", index)
        self.assertIn(b"app.js", index)
        self.assertIn(b"Play against Sophon v1.0", index)
        self.assertIn(b"Recommend moves", index)
        self.assertIn(b"Position evaluation", index)
        self.assertNotIn(b"Every move opens a possibility.", index)
        self.assertNotIn(b"The story so far", index)

        for route, content_type in (("/app.css", "text/css"), ("/app.js", "text/javascript")):
            with self.subTest(route=route):
                status, headers, body = self.request("GET", route)
                self.assertEqual(status, 200)
                self.assertIn(content_type, headers["content-type"])
                self.assertGreater(len(body), 100)
                if route == "/app.js":
                    self.assertIn(b"display_name", body)
                    self.assertIn(b"model.version", body)

        for route in ("/game.py", "/../game.py", "/%2e%2e/game.py", "/web_server.py"):
            with self.subTest(route=route):
                status, headers, body = self.request("GET", route)
                self.assertEqual(status, 404)
                self.assertIn("application/json", headers["content-type"])
                self.assertNotIn(b"GameState", json.dumps(body).encode("utf-8"))

    def test_rejects_foreign_host_origin_and_cross_site_requests(self):
        status, _, _ = self.request("GET", "/api/info", headers={"Host": "evil.example"})
        self.assertEqual(status, 403)
        status, _, _ = self.request(
            "GET", "/api/info", headers={"Origin": "https://evil.example"}
        )
        self.assertEqual(status, 403)
        status, _, _ = self.request(
            "GET", "/api/info", headers={"Sec-Fetch-Site": "cross-site"}
        )
        self.assertEqual(status, 403)
        status, _, _ = self.request(
            "GET", "/api/info", headers={"Origin": f"http://localhost:{self.port}"}
        )
        self.assertEqual(status, 200)

    def test_lan_mode_allows_private_ip_but_rejects_public_or_wrong_origin(self):
        self.server.allow_lan = True
        private = f"10.22.54.77:{self.port}"
        status, _, info = self.request("GET", "/api/info", headers={"Host": private})
        self.assertEqual(status, 200, info)
        status, _, info = self.request(
            "GET", "/api/info",
            headers={"Host": private, "Origin": f"http://{private}"},
        )
        self.assertEqual(status, 200, info)
        for headers in (
            {"Host": f"8.8.8.8:{self.port}"},
            {"Host": private, "Origin": f"http://8.8.8.8:{self.port}"},
            {"Host": private, "Origin": f"http://10.22.54.77:{self.port + 1}"},
        ):
            with self.subTest(headers=headers):
                status, _, _ = self.request("GET", "/api/info", headers=headers)
                self.assertEqual(status, 403)

    def test_public_tunnel_requires_https_same_origin_and_rate_limits(self):
        self.server.public_host_suffix = "trycloudflare.com"
        host = "quiet-field-123.trycloudflare.com"
        public_headers = {"Host": host, "Origin": f"https://{host}"}

        status, headers, info = self.request("GET", "/api/info", headers={"Host": host})
        self.assertEqual(status, 200, info)
        self.assertIn("max-age=31536000", headers["strict-transport-security"])
        status, _, _ = self.request("GET", "/api/info", headers=public_headers)
        self.assertEqual(status, 200)
        status, _, _ = self.request(
            "GET", "/api/info", headers={**public_headers, "Sec-Fetch-Site": "cross-site"}
        )
        self.assertEqual(status, 200)

        rejected = (
            {"Host": host, "Origin": f"http://{host}"},
            {"Host": host, "Origin": "https://other.trycloudflare.com"},
            {"Host": "trycloudflare.com.evil.example"},
        )
        for headers in rejected:
            with self.subTest(headers=headers):
                status, _, _ = self.request("GET", "/api/info", headers=headers)
                self.assertEqual(status, 403)

        self.server.rate_limiter.LIMITS = {
            **self.server.rate_limiter.LIMITS,
            "new": 1,
        }
        status, _, _ = self.request(
            "POST", "/api/new", {"human_player": 1, "mode": "match"}, public_headers
        )
        self.assertEqual(status, 200)
        status, _, result = self.request(
            "POST", "/api/new", {"human_player": 1, "mode": "match"}, public_headers
        )
        self.assertEqual(status, 429, result)
        other_client = {
            **public_headers,
            "X-Forwarded-For": "198.51.100.10, 203.0.113.42",
        }
        status, _, result = self.request(
            "POST", "/api/new", {"human_player": 1, "mode": "match"}, other_client
        )
        self.assertEqual(status, 200, result)

    def test_rejects_invalid_or_oversized_request_bodies(self):
        cases = [
            (b"[]", {"Content-Type": "application/json"}, 400),
            (b'"text"', {"Content-Type": "application/json"}, 400),
            (b"{bad json", {"Content-Type": "application/json"}, 400),
            (b"{}", {"Content-Type": "text/plain"}, 415),
            (b"x" * 8193, {"Content-Type": "application/json"}, 400),
            (b"", {"Content-Type": "application/json"}, 400),
        ]
        for body, headers, expected in cases:
            with self.subTest(length=len(body), headers=headers):
                status, _, result = self.request("POST", "/api/new", body, headers)
                self.assertEqual(status, expected, result)
        self.assertEqual(self.service.sessions, {})

        status, _, result = self.request(
            "POST", "/api/new", {"human_player": True, "mode": "match"}
        )
        self.assertEqual(status, 400, result)
        self.assertEqual(self.service.sessions, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
