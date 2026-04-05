"""Simple HTTP status server for monitoring the agent."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import Callable

import structlog

logger = structlog.get_logger()


class StatusHandler(BaseHTTPRequestHandler):
    """HTTP handler that serves agent status as JSON."""

    status_fn: Callable | None = None

    def do_GET(self):
        if self.path == "/status" or self.path == "/":
            status = self.status_fn() if self.status_fn else {}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(status, indent=2).encode())
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress default HTTP logging


def start_status_server(port: int, status_fn: Callable) -> Thread:
    """Start status HTTP server in a background thread."""
    StatusHandler.status_fn = status_fn
    server = HTTPServer(("0.0.0.0", port), StatusHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info("status_server_started", port=port)
    return thread
