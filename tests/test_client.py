"""Tests for the gateway plugin client. Uses a tiny in-process `websockets`
server as the mock cloud — no Hermes, no Go binary, runs in pytest alone.
"""

from __future__ import annotations

import asyncio
import json
import socket
from typing import Any

import pytest
import websockets
from websockets.asyncio.server import serve as ws_serve

from hermes_vylen_gateway.client import (
    FRAME_HELLO,
    FRAME_READY,
    HandshakeError,
    HelloMeta,
    VylenGatewayClient,
)
from hermes_vylen_gateway.config import GatewayConfig


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _config_for(port: int, token: str = "vyl_live_test") -> GatewayConfig:
    return GatewayConfig(
        instance_token=token,
        cloud_url=f"http://127.0.0.1:{port}",
        websocket_url=f"ws://127.0.0.1:{port}/v1/gateway",
    )


async def _expect_authorized(ws) -> bool:
    auth = ws.request.headers.get("Authorization", "")
    return auth.startswith("Bearer ")


async def _mock_cloud(port: int, *, behavior: str = "ok"):
    """Start a mock cloud on `port`. `behavior` controls how it responds:

    - "ok":          accept hello, send ready
    - "error":       send an error frame instead of ready
    - "wrong_frame": send a non-ready, non-error frame
    - "no_reply":    accept hello but never reply
    """

    async def handler(ws):
        if not await _expect_authorized(ws):
            await ws.close(code=4001, reason="missing auth")
            return
        try:
            raw = await ws.recv()
        except websockets.ConnectionClosed:
            return
        frame: dict[str, Any] = json.loads(raw)
        if frame.get("type") != FRAME_HELLO:
            await ws.send(json.dumps({"type": "error", "message": "expected hello"}))
            return
        if behavior == "error":
            await ws.send(json.dumps({"type": "error", "code": "BAD", "message": "test error"}))
            return
        if behavior == "wrong_frame":
            await ws.send(json.dumps({"type": "something_else"}))
            return
        if behavior == "no_reply":
            await asyncio.sleep(5.0)
            return
        await ws.send(json.dumps({
            "type": FRAME_READY,
            "instance_id": "inst_test",
            "user_id": "giorgio",
            "server_time": "2026-05-13T00:00:00Z",
        }))
        # Hold the socket open until the client closes.
        try:
            async for _ in ws:
                pass
        except websockets.ConnectionClosed:
            pass

    return await ws_serve(handler, "127.0.0.1", port)


@pytest.mark.asyncio
async def test_handshake_ok():
    port = _free_port()
    server = await _mock_cloud(port)
    try:
        client = VylenGatewayClient(_config_for(port), meta=HelloMeta(hostname="ci"))
        ready = await client.connect(timeout=3.0)
        assert ready.instance_id == "inst_test"
        assert ready.user_id == "giorgio"
        await client.close()
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_handshake_error_frame_raises():
    port = _free_port()
    server = await _mock_cloud(port, behavior="error")
    try:
        client = VylenGatewayClient(_config_for(port))
        with pytest.raises(HandshakeError, match="test error"):
            await client.connect(timeout=3.0)
        await client.close()
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_handshake_wrong_frame_raises():
    port = _free_port()
    server = await _mock_cloud(port, behavior="wrong_frame")
    try:
        client = VylenGatewayClient(_config_for(port))
        with pytest.raises(HandshakeError, match="ready"):
            await client.connect(timeout=3.0)
        await client.close()
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_handshake_no_reply_times_out():
    port = _free_port()
    server = await _mock_cloud(port, behavior="no_reply")
    try:
        client = VylenGatewayClient(_config_for(port))
        with pytest.raises(HandshakeError, match="ready frame"):
            await client.connect(timeout=0.5)
        await client.close()
    finally:
        server.close()
        await server.wait_closed()
