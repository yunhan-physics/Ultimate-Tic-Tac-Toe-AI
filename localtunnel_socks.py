"""Publish a local HTTP service through localtunnel using a SOCKS5 proxy.

The official localtunnel client obtains its public URL over HTTPS, but opens
the forwarding sockets directly on a random TCP port.  Some managed networks
block those direct sockets.  This client keeps the same protocol while routing
both the control request and forwarding sockets through a local mixed/SOCKS5
proxy.
"""

from __future__ import annotations

import argparse
import json
import logging
import selectors
import signal
import socket
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, build_opener

from socks_proxy import connect_socks5


DEFAULT_SERVER = "https://localtunnel.me"


def request_tunnel(server: str, proxy_url: str, subdomain: str | None = None) -> dict:
    """Allocate and validate a localtunnel endpoint."""
    endpoint = f"{server.rstrip('/')}/{subdomain}" if subdomain else f"{server.rstrip('/')}/?new"
    opener = build_opener(ProxyHandler({"http": proxy_url, "https": proxy_url}))
    try:
        with opener.open(endpoint, timeout=20) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(f"could not allocate a localtunnel endpoint: {error}") from error
    required = {"id", "port", "url"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise RuntimeError("localtunnel returned an incomplete endpoint description")
    if not isinstance(payload["port"], int) or not 1 <= payload["port"] <= 65535:
        raise RuntimeError("localtunnel returned an invalid forwarding port")
    if not isinstance(payload["url"], str) or not payload["url"].startswith("https://"):
        raise RuntimeError("localtunnel returned an invalid public URL")
    return payload


def relay(remote: socket.socket, local: socket.socket, stop: threading.Event) -> None:
    selector = selectors.DefaultSelector()
    try:
        remote.setblocking(False)
        local.setblocking(False)
        selector.register(remote, selectors.EVENT_READ, local)
        selector.register(local, selectors.EVENT_READ, remote)
        while not stop.is_set():
            for key, _ in selector.select(timeout=1.0):
                source = key.fileobj
                destination = key.data
                try:
                    chunk = source.recv(65536)
                except (BlockingIOError, InterruptedError):
                    continue
                if not chunk:
                    return
                destination.sendall(chunk)
    finally:
        selector.close()


def tunnel_worker(
    *,
    proxy_host: str,
    proxy_port: int,
    remote_host: str,
    remote_port: int,
    local_host: str,
    local_port: int,
    stop: threading.Event,
) -> None:
    while not stop.is_set():
        remote = None
        local = None
        try:
            remote = connect_socks5(proxy_host, proxy_port, remote_host, remote_port)
            local = socket.create_connection((local_host, local_port), timeout=10)
            remote.settimeout(None)
            local.settimeout(None)
            relay(remote, local, stop)
        except OSError as error:
            if not stop.is_set():
                logging.warning("forwarding connection ended: %s", error)
        finally:
            for connection in (remote, local):
                if connection is not None:
                    try:
                        connection.close()
                    except OSError:
                        pass
        stop.wait(0.5)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run localtunnel through a SOCKS5 proxy")
    parser.add_argument("--port", type=int, required=True, help="Local HTTP origin port")
    parser.add_argument("--local-host", default="127.0.0.1")
    parser.add_argument("--proxy-host", default="127.0.0.1")
    parser.add_argument("--proxy-port", type=int, default=7897)
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--subdomain")
    args = parser.parse_args(argv)
    for name in ("port", "proxy_port"):
        if not 1 <= getattr(args, name) <= 65535:
            parser.error(f"{name.replace('_', '-')} must be between 1 and 65535")

    proxy_url = f"http://{args.proxy_host}:{args.proxy_port}"
    endpoint = request_tunnel(args.server, proxy_url, args.subdomain)
    remote_host = endpoint.get("ip") or urlsplit(args.server).hostname
    if not isinstance(remote_host, str) or not remote_host:
        raise RuntimeError("localtunnel did not provide a valid forwarding host")
    stop = threading.Event()

    def request_stop(*_):
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    worker_count = max(1, min(int(endpoint.get("max_conn_count", 1)), 8))
    workers = [
        threading.Thread(
            target=tunnel_worker,
            kwargs={
                "proxy_host": args.proxy_host,
                "proxy_port": args.proxy_port,
                "remote_host": remote_host,
                "remote_port": endpoint["port"],
                "local_host": args.local_host,
                "local_port": args.port,
                "stop": stop,
            },
            daemon=True,
            name=f"localtunnel-{index + 1}",
        )
        for index in range(worker_count)
    ]
    for worker in workers:
        worker.start()
    print(endpoint["url"], flush=True)
    logging.info("forwarding %s with %d connection(s)", endpoint["url"], worker_count)
    try:
        while not stop.wait(1.0):
            if not any(worker.is_alive() for worker in workers):
                raise RuntimeError("all forwarding workers stopped")
    except KeyboardInterrupt:
        stop.set()
    for worker in workers:
        worker.join(timeout=2)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
