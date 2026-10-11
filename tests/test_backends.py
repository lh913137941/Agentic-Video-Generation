import asyncio
import shutil

import httpx
import pytest

from agentic_video.backends import MockVideoBackend, VllmOmniVideoBackend
from agentic_video.composer import FFmpegComposer


def test_vllm_omni_async_job_protocol(tmp_path, monkeypatch):
    seen = []
    original = httpx.AsyncClient

    def handler(request):
        seen.append((request.method, request.url.path))
        if request.method == "POST":
            assert request.headers["content-type"].startswith("multipart/form-data")
            return httpx.Response(200, json={"id": "job-7", "status": "queued"})
        if request.url.path.endswith("/content"):
            return httpx.Response(200, content=b"some mp4 bytes")
        return httpx.Response(200, json={"status": "completed"})

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "agentic_video.backends.httpx.AsyncClient",
        lambda **kwargs: original(transport=transport, **kwargs)
    )
    path = tmp_path / "video.mp4"
    asyncio.run(VllmOmniVideoBackend(
        "http://localhost:8091", 832, 480, 16, 40, poll_interval=0
    ).generate("waves", path, 3.0))
    assert path.read_bytes() == b"some mp4 bytes"
    assert seen == [
        ("POST", "/v1/videos"), ("GET", "/v1/videos/job-7"),
        ("GET", "/v1/videos/job-7/content")
    ]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg missing")
def test_real_mock_clips_and_composer(tmp_path):
    async def scenario():
        backend = MockVideoBackend(fps=8, width=160, height=90)
        clips = []
        for i in range(2):
            path = tmp_path / f"shot-{i}.mp4"
            await backend.generate(f"test shot {i}", path, 1.0)
            clips.append((path, 1.0))
        output = tmp_path / "final.mp4"
        FFmpegComposer(width=160, height=90, fps=12).compose(clips, output)
        assert output.stat().st_size > 500
    asyncio.run(scenario())


def test_vllm_omni_v2v_sends_mp4_reference(tmp_path, monkeypatch):
    """The real adapter sends multipart file input_reference, not T2V-only text."""
    source = tmp_path / "source.mp4"
    source.write_bytes(b"uploaded-source-mp4")
    seen = []
    original = httpx.AsyncClient

    def handler(request):
        seen.append(request.url.path)
        if request.method == "POST":
            payload = request.content
            assert b'name="input_reference"' in payload
            assert b'filename="source.mp4"' in payload
            assert b"uploaded-source-mp4" in payload
            assert b'name="prompt"' in payload
            return httpx.Response(200, json={"id": "job-v2v", "status": "queued"})
        if request.url.path.endswith("/content"):
            return httpx.Response(200, content=b"edited-mp4")
        return httpx.Response(200, json={"status": "completed"})

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        "agentic_video.backends.httpx.AsyncClient",
        lambda **kwargs: original(transport=transport, **kwargs)
    )
    output = tmp_path / "edited.mp4"
    asyncio.run(VllmOmniVideoBackend(
        "http://127.0.0.1:8091", 832, 480, 16, 40, poll_interval=0
    ).generate("edit color", output, 2.0, input_reference=source))
    assert output.read_bytes() == b"edited-mp4"
    assert seen == [
        "/v1/videos", "/v1/videos/job-v2v", "/v1/videos/job-v2v/content"
    ]
