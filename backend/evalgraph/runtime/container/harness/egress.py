"""Egress allowlist proxy (stdlib only), the sandbox's only way out (conf_05 Q-47).

Runs in its own container on two networks: the run's internal network (no route out) and the default bridge. The app
container sits only on the internal network and reaches the outside through this proxy, which allows CONNECT tunnels
(HTTPS) and plain HTTP requests to allowlisted hosts and logs every decision to /out/egress.jsonl.

    python egress.py --port 3128 --allow pypi.org,.huggingface.co,... --log /out/egress.jsonl
A leading dot allows the domain and its subdomains.
"""

from __future__ import annotations

import argparse
import json
import select
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

LOCK = threading.Lock()


def allowed(host: str, rules: list[str]) -> bool:
    host = host.lower().rstrip(".")
    for rule in rules:
        rule = rule.lower().strip()
        if not rule:
            continue
        if rule.startswith("."):
            if host == rule[1:] or host.endswith(rule):
                return True
        elif host == rule:
            return True
    return False


def make_handler(rules: list[str], log_path: str):
    def record(host: str, port: int, ok: bool, method: str) -> None:
        with LOCK, open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "host": host, "port": port, "allowed": ok, "method": method}) + "\n")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # decisions go to the JSONL log
            pass

        def _deny(self, host: str) -> None:
            body = f"EvalGraph sandbox: egress to {host} is not on the allowlist\n".encode()
            self.send_response(403)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

        def do_CONNECT(self):  # HTTPS: tunnel bytes, never decrypt
            host, _, port = self.path.rpartition(":")
            port = int(port or 443)
            ok = allowed(host, rules)
            record(host, port, ok, "CONNECT")
            if not ok:
                return self._deny(host)
            try:
                upstream = socket.create_connection((host, port), timeout=30)
            except OSError:
                self.send_error(502)
                return
            self.send_response(200, "Connection established")
            self.end_headers()
            conns = [self.connection, upstream]
            try:
                while True:
                    readable, _, broken = select.select(conns, [], conns, 120)
                    if broken or not readable:
                        break
                    for s in readable:
                        data = s.recv(65536)
                        if not data:
                            return
                        (upstream if s is self.connection else self.connection).sendall(data)
            finally:
                upstream.close()

        def _forward(self):  # plain HTTP with an absolute URI
            parts = urlsplit(self.path)
            host, port = parts.hostname or "", parts.port or 80
            ok = allowed(host, rules)
            record(host, port, ok, self.command)
            if not ok:
                return self._deny(host)
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            path = parts.path + (f"?{parts.query}" if parts.query else "")
            try:
                with socket.create_connection((host, port), timeout=30) as up:
                    headers = "".join(f"{k}: {v}\r\n" for k, v in self.headers.items() if k.lower() not in ("proxy-connection", "connection"))
                    up.sendall(f"{self.command} {path or '/'} HTTP/1.1\r\n{headers}Connection: close\r\n\r\n".encode() + body)
                    self.close_connection = True
                    while chunk := up.recv(65536):
                        self.wfile.write(chunk)
            except OSError:
                self.send_error(502)

        do_GET = do_POST = do_PUT = do_HEAD = do_DELETE = do_PATCH = _forward

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=3128)
    ap.add_argument("--allow", default="")
    ap.add_argument("--log", default="/out/egress.jsonl")
    args = ap.parse_args()
    rules = [r for r in args.allow.split(",") if r.strip()]
    server = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(rules, args.log))
    print(f"egress proxy on :{args.port}, {len(rules)} allow rules", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
