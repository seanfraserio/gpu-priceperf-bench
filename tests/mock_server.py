"""Minimal OpenAI-compatible SSE server. Stands in for vLLM so the whole
path can be proven for free, on any machine, in CI."""
from __future__ import annotations

import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TOKEN_DELAY_S = 0.001


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # keep dry-run output readable
        pass

    def do_GET(self):
        if self.path != "/health":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        max_tokens = int(body.get("max_tokens", 16))

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        for _ in range(max_tokens):
            chunk = json.dumps({"choices": [{"delta": {"content": "tok "}}]})
            self._write_chunk(f"data: {chunk}\n\n")
            time.sleep(TOKEN_DELAY_S)
        self._write_chunk("data: [DONE]\n\n")
        self._write_chunk("")

    def _write_chunk(self, text: str) -> None:
        payload = text.encode()
        self.wfile.write(f"{len(payload):X}\r\n".encode() + payload + b"\r\n")
        self.wfile.flush()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
