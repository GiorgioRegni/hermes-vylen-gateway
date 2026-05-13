"""Voice transcription: cloud emits a `transcribe` frame with inline audio,
this module runs it through Hermes's bundled transcribe_audio and emits a
`transcribe_response` frame back.

Hermes ships faster-whisper (or OpenAI Whisper, depending on config) under
`tools.transcription_tools.transcribe_audio`. We call it via asyncio.to_thread
so we don't block the WS read loop. Imports are lazy so the plugin still
imports cleanly when Hermes isn't around (tests, doctor CLI).
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import tempfile
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

FRAME_TRANSCRIBE = "transcribe"
FRAME_TRANSCRIBE_RESPONSE = "transcribe_response"
FRAME_RESPONSE_ERROR = "response_error"


class Transcriber:
    """Owns the transcribe-frame lifecycle. Each request runs in a background
    task so the WS read loop stays free."""

    def __init__(self, send_frame: Callable[[dict[str, Any]], Awaitable[None]]):
        self._send = send_frame
        self._tasks: set[asyncio.Task] = set()

    async def handle(self, frame: dict[str, Any]) -> None:
        request_id = frame.get("request_id") or ""
        if not request_id:
            logger.warning("transcribe: frame missing request_id")
            return
        task = asyncio.create_task(self._run(request_id, frame))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def close(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        for task in list(self._tasks):
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

    async def _run(self, request_id: str, frame: dict[str, Any]) -> None:
        audio_b64 = frame.get("data") or ""
        fmt = (frame.get("format") or "m4a").lstrip(".")
        if not audio_b64:
            await self._error(request_id, "BAD_REQUEST", "no audio data")
            return
        try:
            audio_bytes = base64.b64decode(audio_b64)
        except Exception as exc:  # noqa: BLE001
            await self._error(request_id, "BAD_REQUEST", f"base64 decode: {exc}")
            return

        try:
            transcribe_audio = _import_transcribe()
        except ImportError:
            await self._error(
                request_id,
                "STT_UNAVAILABLE",
                "Hermes's transcription tools are not importable. "
                "Run `pip install faster-whisper` in the Hermes venv "
                "and enable stt in config.yaml.",
            )
            return

        path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as f:
                f.write(audio_bytes)
                path = f.name
            logger.info("transcribe: %s (%d bytes, fmt=%s)", request_id, len(audio_bytes), fmt)
            result = await asyncio.to_thread(transcribe_audio, path)
            if isinstance(result, dict) and result.get("success"):
                transcript = (result.get("transcript") or "").strip()
                await self._send({
                    "type": FRAME_TRANSCRIBE_RESPONSE,
                    "request_id": request_id,
                    "transcript": transcript,
                })
            else:
                err = "transcription failed"
                if isinstance(result, dict):
                    err = result.get("error") or err
                await self._error(request_id, "STT_FAILED", err)
        except Exception as exc:  # noqa: BLE001
            logger.exception("transcribe %s failed", request_id)
            await self._error(request_id, "STT_FAILED", str(exc))
        finally:
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass

    async def _error(self, request_id: str, code: str, message: str) -> None:
        await self._send({
            "type": FRAME_RESPONSE_ERROR,
            "request_id": request_id,
            "code": code,
            "message": message,
        })


def _import_transcribe():
    """Lazy import so the plugin remains importable outside of Hermes."""
    from tools.transcription_tools import transcribe_audio  # type: ignore
    return transcribe_audio
