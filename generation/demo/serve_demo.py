#!/usr/bin/env python3
"""Serve the local RAG letter generation demo.

This uses only the Python standard library so reviewers do not need Node,
Gemini credentials, retrieval indexes, or any package installation to view
the saved demo.
"""

from __future__ import annotations

import argparse
import http.server
import socket
import socketserver
from functools import partial
from pathlib import Path


DEMO_DIR = Path(__file__).resolve().parent
DEFAULT_PORT = 8000


class DemoRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - inherited stdlib method name
        if self.path in {"/", ""}:
            self.path = "/rag_letter_demo.html"
        super().do_GET()


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def find_available_port(host: str, start_port: int, attempts: int = 50) -> int:
    for port in range(start_port, start_port + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"No available port found from {start_port} to {start_port + attempts - 1}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the CreditRiskRAG letter-generation demo.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind. Default: 127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Preferred port. Default: 8000")
    parser.add_argument(
        "--strict-port",
        action="store_true",
        help="Fail if the requested port is unavailable instead of trying the next open port.",
    )
    args = parser.parse_args()

    port = args.port if args.strict_port else find_available_port(args.host, args.port)
    handler = partial(DemoRequestHandler, directory=str(DEMO_DIR))

    with ReusableTCPServer((args.host, port), handler) as httpd:
        url = f"http://{args.host}:{port}/rag_letter_demo.html"
        print("CreditRiskRAG letter-generation demo", flush=True)
        print(f"Serving: {DEMO_DIR}", flush=True)
        print(f"Open:    {url}", flush=True)
        print(f"Letters: {url}#letters", flush=True)
        print(f"Eval:    {url}#evaluation", flush=True)
        print("Press Ctrl+C to stop.", flush=True)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped demo server.")


if __name__ == "__main__":
    main()
