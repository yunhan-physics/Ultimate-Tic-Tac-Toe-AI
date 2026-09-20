"""Local HTML client for the champion model. Run: python web_server.py."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import copy
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import logging
import os
from pathlib import Path
import socket
import threading
import time
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

import numpy as np
import torch

from game import GameState, apply_move, legal_moves, new_game, score, terminal_info
from game_record import GameRecorder, RULESET_ID
from play import DEFAULT_CHECKPOINT, DEFAULT_RECORDS_DIR, choose_ai_action, load_ai_model
from web_analysis import analyze_position


ROOT = Path(__file__).resolve().parent
WEB_ROOT = ROOT / "web"
PRIVATE_NETWORKS = tuple(
    ipaddress.ip_network(value) for value in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
)


def private_lan_addresses() -> list[str]:
    """Return this computer's RFC1918 IPv4 addresses for phone URLs."""
    try:
        addresses = socket.gethostbyname_ex(socket.gethostname())[2]
    except OSError:
        return []
    return sorted({
        address for address in addresses
        if any(ipaddress.ip_address(address) in network for network in PRIVATE_NETWORKS)
    })


def _address_parts(value: str, *, origin: bool):
    parsed = urlsplit(value if origin else f"//{value}")
    hostname = (parsed.hostname or "").lower().rstrip(".")
    actual_port = parsed.port if parsed.port is not None else (
        443 if origin and parsed.scheme == "https" else 80 if origin else None
    )
    return parsed, hostname, actual_port


def _public_web_address(value: str, suffix: str | None, *, origin: bool) -> bool:
    if not suffix:
        return False
    try:
        parsed, hostname, actual_port = _address_parts(value, origin=origin)
    except ValueError:
        return False
    suffix = suffix.lower().strip(".")
    if not hostname.endswith(f".{suffix}") or hostname == suffix:
        return False
    if origin:
        return parsed.scheme == "https" and actual_port == 443
    return actual_port in (None, 443)


def _allowed_web_address(
    value: str,
    port: int,
    allow_lan: bool,
    *,
    origin: bool,
    public_host_suffix: str | None = None,
) -> bool:
    """Validate a Host header or same-origin URL without trusting arbitrary DNS names."""
    if _public_web_address(value, public_host_suffix, origin=origin):
        return True
    try:
        parsed, hostname, actual_port = _address_parts(value, origin=origin)
        if origin and parsed.scheme != "http":
            return False
    except ValueError:
        return False
    if actual_port != port:
        return False
    if hostname == "localhost":
        return True
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return False
    if address.is_loopback:
        return True
    return allow_lan and any(address in network for network in PRIVATE_NETWORKS)


class APIError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


@dataclass
class Session:
    identifier: str
    mode: str
    human_player: int
    recorder: GameRecorder
    rng: np.random.Generator
    state: GameState = field(default_factory=new_game)
    revision: int = 0
    aborted: bool = False
    history: list = field(default_factory=list)
    record_path: Path | None = None
    record_error: str | None = None
    analysis: dict | None = None
    lock: threading.RLock = field(default_factory=threading.RLock)


class GameService:
    """Authoritative server state: browsers submit actions, never replacement boards."""

    def __init__(self, model, payload: dict, *, checkpoint=DEFAULT_CHECKPOINT,
                 records_dir=DEFAULT_RECORDS_DIR, simulations=128,
                 analysis_simulations=24, c_puct=1.5, access_mode="local"):
        self.model = model
        self.payload = payload
        self.checkpoint = Path(checkpoint).resolve()
        self.records_dir = Path(records_dir).resolve()
        self.simulations = simulations
        self.analysis_simulations = analysis_simulations
        self.c_puct = c_puct
        self.access_mode = access_mode
        self.sessions: dict[str, Session] = {}
        self.sessions_lock = threading.RLock()
        self.inference_lock = threading.Lock()

    def info(self):
        identity = self.payload.get("model_identity", {})
        model_info = {"name": self.checkpoint.name,
                      "iteration": self.payload.get("iteration")}
        if identity:
            model_info.update(display_name=identity.get("name"), version=identity.get("version"))
        return {"model": model_info,
                "simulations": self.simulations,
                "analysis_simulations": self.analysis_simulations,
                "access_mode": self.access_mode,
                "ruleset": RULESET_ID}

    def session(self, identifier):
        if not isinstance(identifier, str):
            raise APIError("A valid game session is required.")
        with self.sessions_lock:
            value = self.sessions.get(identifier)
        if value is None:
            raise APIError("This session is no longer available. Start a new game.", 404)
        return value

    @staticmethod
    def check_revision(session, revision):
        if type(revision) is not int or revision != session.revision:
            raise APIError("The board has changed. Refresh the position and try again.", 409)

    def _save(self, session, reason=None):
        """One durable, replayable file per game, replaced after every accepted move.

        An unfinished snapshot has the existing recorder's 'aborted' format. It
        explicitly has no outcome; the live session continues independently.
        """
        done, winner = terminal_info(session.state)
        snapshot = copy.copy(session.recorder)
        try:
            temporary = snapshot.finish(
                session.state, winner if done else None,
                status="completed" if done else "aborted",
                reason=None if done else (reason or "web_autosave_incomplete"),
            )
            destination = self.records_dir / f"web_{session.identifier}.json"
            os.replace(temporary, destination)
            session.record_path = destination
            session.record_error = None
        except (OSError, ValueError, RuntimeError):
            logging.exception("Could not autosave game %s", session.identifier)
            session.record_error = "Autosave failed. Keep this page open and retry saving."

    def snapshot(self, session):
        with session.lock:
            done, winner = terminal_info(session.state)
            p1, p2 = score(session.state)
            return {
                "session": session.identifier, "revision": session.revision,
                "mode": session.mode, "human_player": session.human_player,
                "ai_player": 3 - session.human_player,
                "board": list(session.state.board),
                "small_boards": list(session.state.small_boards),
                "to_play": session.state.to_play, "next_board": session.state.next_board,
                "legal_moves": [] if session.aborted else legal_moves(session.state),
                "done": done, "winner": winner, "aborted": session.aborted,
                "ply": len(session.history),
                "last_move": session.history[-1]["action"] if session.history else None,
                "score": {"human": p1 if session.human_player == 1 else p2,
                          "ai": p2 if session.human_player == 1 else p1},
                "history": list(session.history),
                "record": {"saved": session.record_path is not None and not session.record_error,
                           "filename": session.record_path.name if session.record_path else None,
                           "error": session.record_error},
            }

    def new(self, data):
        mode = data.get("mode", "match")
        human = data.get("human_player", 1)
        if mode not in ("match", "coach"):
            raise APIError("Choose Match or Guided play.")
        if type(human) is not int or human not in (1, 2):
            raise APIError("Choose to play first or second.")
        previous = data.get("previous_session")
        if previous is not None:
            old = self.session(previous)
            with old.lock:
                if not terminal_info(old.state)[0] and not old.aborted:
                    old.aborted = True
                    old.revision += 1
                    self._save(old, "web_restart")
        identifier = uuid4().hex
        seed = int(identifier[:8], 16)
        metadata = {"interface": "web", "mode": mode, "assisted": mode == "coach"}
        identity = self.payload.get("model_identity")
        if identity:
            metadata["model_identity"] = dict(identity)
        recorder = GameRecorder(
            self.records_dir, human_player=human, checkpoint=self.checkpoint,
            model_iteration=self.payload.get("iteration"), simulations=self.simulations,
            c_puct=self.c_puct, seed=seed, game_id=identifier,
            metadata=metadata,
        )
        value = Session(identifier, mode, human, recorder, np.random.default_rng(seed))
        with self.sessions_lock:
            # Keep memory bounded; ended games remain available as JSON on disk.
            if len(self.sessions) >= 128:
                expired = next((key for key, item in self.sessions.items()
                                if item.aborted or terminal_info(item.state)[0]), None)
                if expired is None:
                    raise APIError("Too many active games. Finish a game before starting another.", 503)
                del self.sessions[expired]
            self.sessions[identifier] = value
        with value.lock:
            self._save(value)
        return self.snapshot(value)

    def _apply(self, session, action, actor):
        before = session.state
        after, _, _ = apply_move(before, action)
        session.recorder.record_move(before, action, actor, after)
        session.state = after
        session.revision += 1
        row, column = divmod(action, 9)
        session.history.append({"ply": len(session.history) + 1, "player": before.to_play,
                                "actor": actor, "action": action,
                                "row": row + 1, "column": column + 1})
        session.analysis = None
        self._save(session)

    def move(self, data, *, ai=False):
        session = self.session(data.get("session"))
        with session.lock:
            self.check_revision(session, data.get("revision"))
            if session.aborted or terminal_info(session.state)[0]:
                raise APIError("This game has ended. Start a new game.", 409)
            human_turn = session.state.to_play == session.human_player
            if human_turn == ai:
                raise APIError("It is not this player's turn.", 409)
            if ai:
                with self.inference_lock:
                    action = choose_ai_action(
                        self.model, session.state, self.simulations, session.rng,
                        c_puct=self.c_puct, cache={},
                    )
            else:
                action = data.get("action")
                if type(action) is not int or action not in legal_moves(session.state):
                    raise APIError("Choose an empty cell in a highlighted open board.")
            self._apply(session, action, "ai" if ai else "human")
            return self.snapshot(session)

    def analyze(self, data):
        session = self.session(data.get("session"))
        with session.lock:
            self.check_revision(session, data.get("revision"))
            if session.analysis is not None:
                return session.analysis
            state, revision = session.state, session.revision
            include_moves = session.mode == "coach" and not session.aborted
        # Inference may be slow; do not hold the game lock while giving advice.
        with self.inference_lock:
            result = analyze_position(
                state, self.model, session.human_player, simulations=self.simulations,
                action_simulations=self.analysis_simulations, c_puct=self.c_puct,
                include_moves=include_moves,
            )
        with session.lock:
            self.check_revision(session, revision)
            result.update(session=session.identifier, revision=revision)
            session.analysis = result
            return result

    def abort(self, data):
        session = self.session(data.get("session"))
        with session.lock:
            self.check_revision(session, data.get("revision"))
            if not terminal_info(session.state)[0] and not session.aborted:
                session.aborted = True
                session.revision += 1
                session.analysis = None
            self._save(session, "web_user_end")
            return self.snapshot(session)

    def record(self, identifier):
        session = self.session(identifier)
        with session.lock:
            if session.record_path is None or session.record_error:
                self._save(session, "web_user_end" if session.aborted else None)
            if session.record_path is None or session.record_error:
                raise APIError("The game record could not be saved. Please try again.", 500)
            return session.record_path.name, session.record_path.read_bytes()


class LocalServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, service, *, allow_lan=False, public_host_suffix=None):
        self.service = service
        self.allow_lan = allow_lan
        self.public_host_suffix = public_host_suffix
        self.rate_limiter = RequestRateLimiter()
        # GameService serializes model inference. Bound the public waiting room so
        # a burst cannot leave an unbounded number of HTTP threads blocked.
        self.public_compute_slots = threading.BoundedSemaphore(4)
        super().__init__(address, RequestHandler)


class RequestRateLimiter:
    """Small in-memory sliding-window limiter for public tunnel traffic."""

    LIMITS = {"general": 180, "write": 60, "new": 6, "compute": 12}

    def __init__(self, window_seconds=60.0):
        self.window_seconds = float(window_seconds)
        self._events = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, client: str, bucket: str) -> bool:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        key = (client, bucket)
        with self._lock:
            events = self._events[key]
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.LIMITS[bucket]:
                return False
            events.append(now)
            # Avoid retaining quiet clients forever.
            if len(self._events) > 4096:
                stale = [item for item, values in self._events.items() if not values or values[-1] <= cutoff]
                for item in stale[:1024]:
                    self._events.pop(item, None)
            return True


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "UltimateChess/1.0"

    def log_message(self, format, *args):
        logging.info("%s %s", self.address_string(), format % args)

    def _send(self, status, body, content_type="application/json; charset=utf-8", filename=None):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
        if self.server.public_host_suffix:
            self.send_header("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(body)

    def _check_local_request(self):
        port = self.server.server_port
        host = self.headers.get("Host", "")
        public = _public_web_address(
            host, self.server.public_host_suffix, origin=False
        )
        if not _allowed_web_address(
            host,
            port,
            self.server.allow_lan,
            origin=False,
            public_host_suffix=self.server.public_host_suffix,
        ):
            raise APIError("Use this app through its approved address.", 403)
        origin = self.headers.get("Origin")
        if origin is not None and not _allowed_web_address(
            origin,
            port,
            self.server.allow_lan,
            origin=True,
            public_host_suffix=self.server.public_host_suffix,
        ):
            raise APIError("Requests must come from this app.", 403)
        if public and origin is not None:
            try:
                _, host_name, _ = _address_parts(host, origin=False)
                _, origin_name, _ = _address_parts(origin, origin=True)
            except ValueError:
                raise APIError("Requests must come from this app.", 403) from None
            if host_name != origin_name:
                raise APIError("Requests must come from this app.", 403)
        if self.headers.get("Sec-Fetch-Site") == "cross-site":
            # A few mobile/in-app browsers label same-origin fetches as
            # cross-site after opening a temporary HTTPS tunnel.  The strict
            # Host/Origin checks above still apply; reject only when the
            # request does not carry the approved public origin.
            if not public or origin is None:
                raise APIError("Cross-site requests are not supported.", 403)
        self._public_request = public

    def _public_client(self):
        if not getattr(self, "_public_request", False):
            return self.client_address[0]
        # The origin is loopback-only, so these headers can only arrive through
        # the approved reverse tunnel (or another local process).  Cloudflare
        # supplies CF-Connecting-IP; localtunnel supplies X-Forwarded-For.
        candidates = []
        if self.server.public_host_suffix == "trycloudflare.com":
            candidates.append(self.headers.get("CF-Connecting-IP", "").strip())
        forwarded_for = self.headers.get("X-Forwarded-For", "")
        if forwarded_for:
            # Use the hop appended by the tunnel, rather than a client-supplied
            # value at the start of the chain.
            candidates.append(forwarded_for.rsplit(",", 1)[-1].strip())
        for forwarded in candidates:
            try:
                return str(ipaddress.ip_address(forwarded))
            except ValueError:
                continue
        return self.client_address[0]

    def _check_rate_limit(self, route: str, post: bool):
        if not getattr(self, "_public_request", False):
            return
        client = self._public_client()
        buckets = ["general"]
        if post:
            buckets.append("write")
        if route == "/api/new":
            buckets.append("new")
        if route in ("/api/ai", "/api/analysis"):
            buckets.append("compute")
        if not all(self.server.rate_limiter.allow(client, bucket) for bucket in buckets):
            raise APIError("Too many requests. Wait a minute and try again.", 429)

    def _body(self):
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            raise APIError("Send JSON request data.", 415)
        if self.headers.get("Transfer-Encoding"):
            raise APIError("Chunked requests are not supported.")
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 8192:
                raise ValueError
            self.connection.settimeout(15)
            data = json.loads(self.rfile.read(length))
        except (ValueError, UnicodeDecodeError, TimeoutError):
            raise APIError("Invalid or oversized JSON request.") from None
        if not isinstance(data, dict):
            raise APIError("The request must be a JSON object.")
        return data

    def _dispatch(self, post=False):
        try:
            self._check_local_request()
            route = urlsplit(self.path)
            self._check_rate_limit(route.path, post)
            service = self.server.service
            if post:
                data = self._body()
                endpoints = {"/api/new": service.new, "/api/move": service.move,
                             "/api/ai": lambda value: service.move(value, ai=True),
                             "/api/analysis": service.analyze, "/api/abort": service.abort}
                action = endpoints.get(route.path)
                if action is None:
                    raise APIError("Endpoint not found.", 404)
                compute = getattr(self, "_public_request", False) and route.path in (
                    "/api/ai", "/api/analysis"
                )
                if compute and not self.server.public_compute_slots.acquire(blocking=False):
                    raise APIError("The AI is busy. Try again in a moment.", 429)
                try:
                    result = action(data)
                finally:
                    if compute:
                        self.server.public_compute_slots.release()
                self._send(200, result)
                return
            params = parse_qs(route.query)
            identifier = params.get("session", [None])[0]
            if route.path == "/api/info":
                self._send(200, service.info())
            elif route.path == "/api/state":
                self._send(200, service.snapshot(service.session(identifier)))
            elif route.path == "/api/record":
                name, content = service.record(identifier)
                self._send(200, content, filename=name)
            elif route.path in ("/", "/index.html", "/app.css", "/app.js"):
                name = "index.html" if route.path == "/" else route.path[1:]
                types = {"index.html": "text/html; charset=utf-8",
                         "app.css": "text/css; charset=utf-8",
                         "app.js": "text/javascript; charset=utf-8"}
                self._send(200, (WEB_ROOT / name).read_bytes(), types[name])
            else:
                raise APIError("Page not found.", 404)
        except APIError as exc:
            self._send(exc.status, {"error": str(exc)})
        except (BrokenPipeError, ConnectionResetError):
            pass  # The browser closed; any accepted move was already autosaved.
        except Exception:
            logging.exception("Request failed")
            try:
                self._send(500, {"error": "The request failed. Refresh the position and try again."})
            except (BrokenPipeError, ConnectionResetError):
                pass

    def do_GET(self):
        self._dispatch()

    def do_POST(self):
        self._dispatch(post=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Play the trained champion in a local browser.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--lan", action="store_true",
                        help="Allow phones on the same private IPv4 network.")
    parser.add_argument(
        "--public-host-suffix",
        default=None,
        help="Allow same-origin HTTPS hosts below this suffix (for example trycloudflare.com).",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--records-dir", type=Path, default=DEFAULT_RECORDS_DIR)
    parser.add_argument("--simulations", type=int, default=128)
    parser.add_argument("--analysis-simulations", type=int, default=24,
                        help="Equal MCTS budget per legal move in guided play.")
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if min(args.simulations, args.analysis_simulations, args.torch_threads) < 1:
        parser.error("Search budgets and thread count must be positive")
    if args.lan and args.public_host_suffix:
        parser.error("--lan and --public-host-suffix cannot be used together")
    return args


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.set_num_threads(args.torch_threads)
    try:
        model, payload = load_ai_model(args.checkpoint)
        access_mode = "public" if args.public_host_suffix else "lan" if args.lan else "local"
        service = GameService(model, payload, checkpoint=args.checkpoint,
                              records_dir=args.records_dir, simulations=args.simulations,
                              analysis_simulations=args.analysis_simulations,
                              access_mode=access_mode)
        bind_address = "0.0.0.0" if args.lan else "127.0.0.1"
        server = LocalServer(
            (bind_address, args.port),
            service,
            allow_lan=args.lan,
            public_host_suffix=args.public_host_suffix,
        )
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        logging.error("Could not start the app: %s", exc)
        return 2
    logging.info("Champion loaded: %s (iteration %s)", args.checkpoint, payload.get("iteration"))
    logging.info("Open http://127.0.0.1:%s  |  Records: %s", args.port, service.records_dir)
    if args.lan:
        for address in private_lan_addresses():
            logging.info("Phone on the same Wi-Fi: http://%s:%s", address, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Server stopped. All accepted moves have been autosaved.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
