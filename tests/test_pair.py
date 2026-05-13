"""Tests for the `vylen-gateway-pair` CLI."""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from hermes_vylen_gateway.pair import PairError, exchange, main


class _FakeCloud(BaseHTTPRequestHandler):
    """Tiny HTTP server that mimics the cloud's POST /v1/instances/pair."""

    behavior = "ok"

    def do_POST(self):  # noqa: N802
        if self.path != "/v1/instances/pair":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        code = body.get("pairing_code", "")

        if _FakeCloud.behavior == "rejected" or code == "BAD-CODE":
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "error": {"code": "PAIRING_CODE_INVALID", "message": "Pairing code is invalid"}
            }).encode())
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({
            "instance_id": "inst_test",
            "instance_token": "vyl_live_T0K3N",
        }).encode())

    def log_message(self, format, *args):  # noqa: A002
        pass


@pytest.fixture
def cloud():
    _FakeCloud.behavior = "ok"
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = HTTPServer(("127.0.0.1", port), _FakeCloud)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        thread.join(timeout=1)


def test_exchange_returns_token(cloud):
    instance_id, token = exchange(cloud, "ABC1-DEF2")
    assert instance_id == "inst_test"
    assert token == "vyl_live_T0K3N"


def test_exchange_bad_code_raises(cloud):
    with pytest.raises(PairError, match="invalid"):
        exchange(cloud, "BAD-CODE")


def test_exchange_unreachable_cloud():
    # Pick a port nothing is listening on.
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with pytest.raises(PairError, match="could not reach"):
        exchange(f"http://127.0.0.1:{port}", "ABC1-DEF2", timeout=0.5)


def test_main_prints_only_token_by_default(cloud, capsys):
    rc = main(["--cloud-url", cloud, "ABC1-DEF2"])
    assert rc == 0
    out = capsys.readouterr()
    assert out.out.strip() == "vyl_live_T0K3N"
    assert out.err == ""


def test_main_verbose_prints_summary(cloud, capsys):
    rc = main(["--cloud-url", cloud, "-v", "ABC1-DEF2"])
    assert rc == 0
    out = capsys.readouterr()
    assert "instance_id" in out.out
    assert "vyl_live_T0K3N" in out.out
    assert "VYLEN_INSTANCE_TOKEN=" in out.out


def test_main_returns_nonzero_on_bad_code(cloud, capsys):
    rc = main(["--cloud-url", cloud, "BAD-CODE"])
    assert rc == 1
    out = capsys.readouterr()
    assert "invalid" in out.err.lower()
