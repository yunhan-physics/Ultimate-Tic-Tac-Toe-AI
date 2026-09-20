"""Minimal SOCKS5 stdio bridge for OpenSSH ProxyCommand."""

from __future__ import annotations

import socket
import struct
import sys
import threading


def _read_exact(sock: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("SOCKS proxy closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def connect_socks5(proxy_host: str, proxy_port: int, target_host: str, target_port: int):
    sock = socket.create_connection((proxy_host, proxy_port), timeout=15)
    sock.sendall(b"\x05\x01\x00")
    if _read_exact(sock, 2) != b"\x05\x00":
        raise ConnectionError("SOCKS proxy rejected no-authentication mode")
    encoded = target_host.encode("idna")
    if len(encoded) > 255:
        raise ValueError("target hostname is too long")
    sock.sendall(b"\x05\x01\x00\x03" + bytes([len(encoded)]) + encoded + struct.pack("!H", target_port))
    header = _read_exact(sock, 4)
    if header[:2] != b"\x05\x00":
        raise ConnectionError(f"SOCKS CONNECT failed with status {header[1]}")
    address_type = header[3]
    if address_type == 1:
        _read_exact(sock, 4)
    elif address_type == 3:
        _read_exact(sock, _read_exact(sock, 1)[0])
    elif address_type == 4:
        _read_exact(sock, 16)
    else:
        raise ConnectionError("SOCKS proxy returned an unknown address type")
    _read_exact(sock, 2)
    sock.settimeout(None)
    return sock


def _stdin_to_socket(sock: socket.socket):
    try:
        while data := sys.stdin.buffer.read(65536):
            sock.sendall(data)
    except (BrokenPipeError, ConnectionError, OSError):
        pass
    try:
        sock.shutdown(socket.SHUT_WR)
    except OSError:
        pass


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 4:
        print("usage: socks_proxy.py PROXY_HOST PROXY_PORT TARGET_HOST TARGET_PORT", file=sys.stderr)
        return 2
    proxy_host, proxy_port, target_host, target_port = argv
    sock = connect_socks5(proxy_host, int(proxy_port), target_host, int(target_port))
    upstream = threading.Thread(target=_stdin_to_socket, args=(sock,), daemon=True)
    upstream.start()
    try:
        while data := sock.recv(65536):
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()
    except (BrokenPipeError, ConnectionError, OSError):
        pass
    finally:
        sock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
