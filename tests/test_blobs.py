from pathlib import Path
import base64

import pytest

from hermes_vylen_gateway.blobs import DEFAULT_MAX_BYTES, BlobRegistry
from hermes_vylen_gateway.relay import BLOB_PATH_PREFIX, FRAME_RESPONSE_CHUNK, FRAME_RESPONSE_END, HermesRelay

MAX_BLOB_CHUNK = 256 * 1024


@pytest.mark.asyncio
async def test_register_rejects_oversized_blob(tmp_path: Path) -> None:
    blob = tmp_path / "large.bin"
    with blob.open("wb") as f:
        f.seek(DEFAULT_MAX_BYTES)
        f.write(b"x")

    registry = BlobRegistry()

    assert await registry.register(blob) is None


@pytest.mark.asyncio
async def test_register_accepts_regular_blob(tmp_path: Path) -> None:
    blob = tmp_path / "image.png"
    blob.write_bytes(b"png")

    registry = BlobRegistry()
    registered = await registry.register(blob)

    assert registered is not None
    token, mime, filename = registered
    assert token
    assert mime == "image/png"
    assert filename == "image.png"


@pytest.mark.asyncio
async def test_relay_serves_blob_in_existing_chunk_size(tmp_path: Path) -> None:
    blob = tmp_path / "image.png"
    original = b"x" * (MAX_BLOB_CHUNK + 1)
    blob.write_bytes(original)
    registry = BlobRegistry()
    registered = await registry.register(blob)
    assert registered is not None
    token, _, _ = registered

    sent_frames: list[dict] = []

    async def send(frame):
        sent_frames.append(frame)

    relay = HermesRelay(send, blobs=registry)
    try:
        await relay._serve_blob("req_blob", "GET", BLOB_PATH_PREFIX + token)
    finally:
        await relay.close()

    chunks = [base64.b64decode(f["data"]) for f in sent_frames if f["type"] == FRAME_RESPONSE_CHUNK]
    assert [len(c) for c in chunks] == [MAX_BLOB_CHUNK, 1]
    assert b"".join(chunks) == original
    assert sent_frames[-1]["type"] == FRAME_RESPONSE_END
