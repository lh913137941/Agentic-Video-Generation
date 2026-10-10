"""Upload-first smart-edit API: fail closed without V2V, preserve source, poll result."""
import time

import pytest
from fastapi.testclient import TestClient
from agentic_video.api import create_app
from agentic_video.composer import FFmpegComposer
from agentic_video.backends import MockVideoBackend
from agentic_video.parsers import RuleStoryboardParser
from agentic_video.service import VideoService
from agentic_video.store import FileProjectStore


class FakeV2V:
    async def generate(self, prompt, destination, duration, input_reference=None):
        assert input_reference is not None
        assert input_reference.name == "source.mp4"
        assert input_reference.is_file()
        assert "黄昏" in prompt
        destination.write_bytes(b"v2v-result")


def fake_service(tmp_path, backend):
    return VideoService(
        FileProjectStore(tmp_path), RuleStoryboardParser(), backend,
        FFmpegComposer(), concurrency=1,
    )


@pytest.mark.skipif(__import__("shutil").which("ffmpeg") is None, reason="FFmpeg absent")
def test_upload_edit_poll_and_download(tmp_path):
    import asyncio
    original = tmp_path / "original.mp4"
    asyncio.run(MockVideoBackend(fps=8, width=160, height=90).generate(
        "input scene", original, 1.0
    ))
    app = create_app(fake_service(tmp_path / "db", FakeV2V()))
    app.state.video_editor.enabled = True
    with TestClient(app) as client:
        with original.open("rb") as source:
            res = client.post("/api/video-edits", files={
                "file": ("test.mp4", source, "video/mp4")
            }, data={"prompt": "变成黄昏"})
        assert res.status_code == 202, res.text
        job_id = res.json()["job_id"]
        for _ in range(150):
            result = client.get(f"/api/video-edits/{job_id}").json()
            if result["status"] in {"READY", "FAILED"}:
                break
            time.sleep(.02)
        assert result["status"] == "READY", result
        assert client.get(
            f"/api/video-edits/{job_id}/source"
        ).content == original.read_bytes()
        inline = client.get(f"/api/video-edits/{job_id}/result")
        attachment = client.get(f"/api/video-edits/{job_id}/result?download=1")
        assert inline.content == b"v2v-result"
        assert inline.headers["content-disposition"].startswith("inline;")
        assert attachment.content == b"v2v-result"
        assert attachment.headers["content-disposition"].startswith("attachment;")
        assert client.get("/api/video-edits/no-such-job").status_code == 404


def test_mock_mode_disallows_misleading_edit(tmp_path):
    app = create_app(fake_service(tmp_path, MockVideoBackend()))
    with TestClient(app) as client:
        info = client.get("/api/capabilities").json()
        assert info["video_edit_upload"] is False
        assert info["spatial_masks"] is False
        assert info["temporal_range_control"] is False
        res = client.post("/api/video-edits", files={
            "file": ("sample.mp4", b"fake", "video/mp4")
        }, data={"prompt": "change shirt"})
        assert res.status_code == 409
        assert "V2V" in res.text
        assert client.get("/api/video-edits/bad-id/result").status_code == 404


def test_rejects_invalid_mp4_without_leaving_partial_job(tmp_path):
    app = create_app(fake_service(tmp_path, FakeV2V()))
    app.state.video_editor.enabled = True
    with TestClient(app) as client:
        res = client.post("/api/video-edits", files={
            "file": ("unsafe.mp4", b"not-really-a-video", "video/mp4")
        }, data={"prompt": "test"})
        assert res.status_code == 422
        assert list((tmp_path / "video_edits").glob("*")) == []
