from pathlib import Path

import pytest

from hermes_vylen_gateway.blobs import DEFAULT_MAX_BYTES, BlobRegistry


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
