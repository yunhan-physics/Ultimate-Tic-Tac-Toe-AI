"""Start the local web app once and open it in the default browser."""

from __future__ import annotations

import argparse
import json
import ipaddress
import os
from pathlib import Path
import subprocess
import sys
import socket
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
import webbrowser


ROOT = Path(__file__).resolve().parent
RULESET = "ultimate_tic_tac_toe_score_only_v1"
PRIVATE_NETWORKS = tuple(
    ipaddress.ip_network(value) for value in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
)


def probe(url: str) -> str:
    """Return ``ours``, ``offline`` or ``foreign`` for the selected port."""
    try:
        with urlopen(f"{url}/api/info", timeout=1.5) as response:
            if response.status != 200:
                return "foreign"
            payload = json.load(response)
    except HTTPError:
        return "foreign"
    except (URLError, TimeoutError, ConnectionError, OSError):
        return "offline"
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return "foreign"
    if (
        isinstance(payload, dict)
        and payload.get("ruleset") == RULESET
        and isinstance(payload.get("model"), dict)
    ):
        return "ours"
    return "foreign"


def private_lan_addresses() -> list[str]:
    try:
        addresses = socket.gethostbyname_ex(socket.gethostname())[2]
    except OSError:
        return []
    return sorted({
        value for value in addresses
        if any(ipaddress.ip_address(value) in network for network in PRIVATE_NETWORKS)
    })


def start_server(port: int, lan: bool) -> tuple[subprocess.Popen, Path]:
    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout_path = logs / "web_server.stdout.log"
    stderr_path = logs / "web_server.stderr.log"
    creation_flags = 0
    startupinfo = None
    if os.name == "nt":
        creation_flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
    with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
        command = [sys.executable, "-B", str(ROOT / "web_server.py"), "--port", str(port)]
        if lan:
            command.append("--lan")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            creationflags=creation_flags,
            startupinfo=startupinfo,
            close_fds=True,
        )
    return process, stderr_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch the local Ultimate Tic-Tac-Toe website.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--lan", action="store_true", help="Allow phones on the same Wi-Fi.")
    parser.add_argument("--no-open", action="store_true", help="Start without opening a browser.")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    url = f"http://127.0.0.1:{args.port}"
    status = probe(url)
    process = None
    error_log = ROOT / "logs" / "web_server.stderr.log"
    if status == "foreign":
        print(f"Port {args.port} is already used by another application.", file=sys.stderr)
        print("Close that application or run: python start.py --port 8766", file=sys.stderr)
        return 2
    if status == "offline":
        process, error_log = start_server(args.port, args.lan)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            status = probe(url)
            if status == "ours":
                break
            if status == "foreign" or process.poll() is not None:
                break
            time.sleep(0.25)
    if status != "ours":
        print("The local game server could not start.", file=sys.stderr)
        print(f"Details: {error_log}", file=sys.stderr)
        return 2

    print(f"Ultimate Tic-Tac-Toe is ready: {url}")
    if args.lan:
        addresses = private_lan_addresses()
        if addresses:
            for address in addresses:
                print(f"Phone address: http://{address}:{args.port}")
        else:
            print("No private Wi-Fi address was found. Check the computer's network connection.")
    if not args.no_open:
        webbrowser.open(url, new=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
