"""Launch a rate-limited public HTTPS quick tunnel for the local game server."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, build_opener, urlopen

from launch_web import probe


ROOT = Path(__file__).resolve().parent
DEFAULT_CLOUDFLARED = ROOT / "tools" / "cloudflared.exe"
PROVIDER_SUFFIXES = {
    "cloudflare": "trycloudflare.com",
    "localtunnel": "loca.lt",
}
URL_PATTERN = re.compile(
    r"https://[a-z0-9-]+\.(?:trycloudflare\.com|loca\.lt)(?=$|[\s/])",
    re.IGNORECASE,
)


def _popen(command: list[str], stdout_path: Path, stderr_path: Path) -> subprocess.Popen:
    creation_flags = 0
    startupinfo = None
    if os.name == "nt":
        creation_flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        return subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            creationflags=creation_flags,
            startupinfo=startupinfo,
            close_fds=True,
        )


def extract_public_url(*logs: Path) -> str | None:
    matches: list[str] = []
    for path in logs:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        matches.extend(URL_PATTERN.findall(text))
    return matches[-1].lower() if matches else None


def probe_public(url: str, proxy_url: str | None = None) -> bool:
    try:
        if proxy_url:
            opener = build_opener(ProxyHandler({"http": proxy_url, "https": proxy_url}))
            response_context = opener.open(f"{url}/api/info", timeout=8)
        else:
            response_context = urlopen(f"{url}/api/info", timeout=8)
        with response_context as response:
            payload = json.load(response)
        return (
            response.status == 200
            and payload.get("ruleset") == "ultimate_tic_tac_toe_score_only_v1"
            and payload.get("access_mode") == "public"
        )
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish Sophon through an HTTPS tunnel")
    parser.add_argument("--port", type=int, default=8766, help="Loopback-only origin port")
    parser.add_argument(
        "--provider", choices=tuple(PROVIDER_SUFFIXES), default="localtunnel",
        help="Tunnel provider (localtunnel works through the campus SOCKS proxy)",
    )
    parser.add_argument("--proxy-host", default="127.0.0.1")
    parser.add_argument("--proxy-port", type=int, default=7897)
    parser.add_argument("--cloudflared", type=Path, default=DEFAULT_CLOUDFLARED)
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535 or not 1 <= args.proxy_port <= 65535 or args.timeout <= 0:
        parser.error("ports and timeout must be positive and valid")
    cloudflared = args.cloudflared.expanduser().resolve()
    if args.provider == "cloudflare" and not cloudflared.is_file():
        parser.error(f"cloudflared was not found: {cloudflared}")
    public_suffix = PROVIDER_SUFFIXES[args.provider]
    proxy_url = (
        f"http://{args.proxy_host}:{args.proxy_port}"
        if args.provider == "localtunnel" else None
    )

    logs = ROOT / "logs"
    status_path = logs / "public_status.json"
    if status_path.is_file():
        try:
            saved = json.loads(status_path.read_text(encoding="utf-8"))
            existing_url = saved.get("public_url")
            existing_provider = saved.get("provider")
        except (OSError, ValueError, json.JSONDecodeError):
            existing_url = None
            existing_provider = None
        if (
            existing_provider == args.provider
            and isinstance(existing_url, str)
            and probe_public(existing_url, proxy_url)
        ):
            print(f"Public Sophon is already ready: {existing_url}")
            return 0

    local_url = f"http://127.0.0.1:{args.port}"
    origin_process = None
    local_status = probe(local_url)
    if local_status == "foreign":
        print(f"Port {args.port} is already used by another application.", file=sys.stderr)
        return 2
    if local_status == "offline":
        origin_process = _popen(
            [
                sys.executable,
                "-B",
                str(ROOT / "web_server.py"),
                "--port",
                str(args.port),
                "--public-host-suffix",
                public_suffix,
            ],
            logs / "public_origin.stdout.log",
            logs / "public_origin.stderr.log",
        )
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and probe(local_url) != "ours":
            if origin_process.poll() is not None:
                break
            time.sleep(0.25)
        if probe(local_url) != "ours":
            print("The loopback-only public origin could not start.", file=sys.stderr)
            print(f"Details: {logs / 'public_origin.stderr.log'}", file=sys.stderr)
            return 2

    tunnel_stdout = logs / "public_tunnel.stdout.log"
    tunnel_stderr = logs / "public_tunnel.stderr.log"
    if args.provider == "cloudflare":
        tunnel_command = [
            str(cloudflared),
            "tunnel",
            "--url",
            local_url,
            "--no-autoupdate",
            "--loglevel",
            "info",
        ]
    else:
        tunnel_command = [
            sys.executable,
            "-u",
            "-B",
            str(ROOT / "localtunnel_socks.py"),
            "--port",
            str(args.port),
            "--proxy-host",
            args.proxy_host,
            "--proxy-port",
            str(args.proxy_port),
        ]
    tunnel_process = _popen(
        tunnel_command,
        tunnel_stdout,
        tunnel_stderr,
    )
    deadline = time.monotonic() + args.timeout
    public_url = None
    while time.monotonic() < deadline:
        if tunnel_process.poll() is not None:
            break
        public_url = extract_public_url(tunnel_stdout, tunnel_stderr)
        if public_url and public_url.endswith(f".{public_suffix}") and probe_public(public_url, proxy_url):
            break
        time.sleep(0.5)
    if (
        not public_url
        or not public_url.endswith(f".{public_suffix}")
        or not probe_public(public_url, proxy_url)
    ):
        print("The public tunnel did not become ready.", file=sys.stderr)
        print(f"Details: {tunnel_stderr}", file=sys.stderr)
        return 2

    status = {
        "public_url": public_url,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "origin": local_url,
        "origin_pid": origin_process.pid if origin_process is not None else None,
        "tunnel_pid": tunnel_process.pid,
        "provider": args.provider,
        "model": "Tic-Tac-Toe Sophon v1.0",
        "note": "The temporary URL changes whenever the tunnel is restarted.",
    }
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Public Sophon is ready: {public_url}")
    print("This temporary URL remains live while this computer and the tunnel process are running.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
