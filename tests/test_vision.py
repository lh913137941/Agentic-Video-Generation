"""Video understanding API + sampled image contract, without real API credentials."""
import asyncio
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from agentic_video.api import create_app
from agentic_video.backends import MockVideoBackend
from agentic_video.models import ShotSpec
from agentic_video.parsers import RuleStoryboardParser
from agentic_video.service import VideoService
from agentic_video.store import FileProjectStore
from agentic_video.vision import (
    OpenAICompatibleVisionAnalyzer, VisionConfig, VisionError, VisionPlan,
    sample_seconds,
)


class DummyComposer:
    def compose(self, clips, output):
        output.write_bytes(b"composed")


class DummyBackend:
    async def generate(self, prompt, destination, duration):
        destination.write_bytes(b"clip")


def service(tmp_path):
    return VideoService(
        FileProjectStore(tmp_path), RuleStoryboardParser(),
        DummyBackend(), DummyComposer(), max_shots=5,
    )


def plan():
    return VisionPlan(
        summary="孩子在树荫下的街道行走",
        observations=["孩子在道路上行走", "背景有树木和建筑"],
        shots=[
            ShotSpec(prompt="街道远景，孩子慢慢向前走", duration=2.0),
            ShotSpec(prompt="孩子在树下的中景", duration=3.0),
        ],
    )


class DummyVision:
    config = VisionConfig(
        "https://api.example.test", "test-vision-model", "secret"
    )

    async def analyze(self, file, instruction, shot_count, style):
        assert instruction == "根据视频生成两镜头"
        assert shot_count == 2
        assert style == "cinematic"
        await file.close()
        return plan(), [0.5, 1.5]


def test_understand_creates_exact_shots_without_text_reparse(tmp_path):
    app = create_app(service(tmp_path), vision_analyzer=DummyVision())
    with TestClient(app) as client:
        capabilities = client.get("/api/capabilities").json()
        assert capabilities["video_understanding"] is True
        assert capabilities["vision_model"] == "test-vision-model"
        assert "secret" not in str(capabilities)
        assert capabilities["video_edit_upload"] is False
        response = client.post(
            "/api/video-understand",
            files={"file": ("source.mp4", b"some-bytes", "video/mp4")},
            data={
                "instruction": "根据视频生成两镜头",
                "shot_count": "2", "theme_style": "cinematic",
            },
        )
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["summary"].startswith("孩子")
        assert data["sampled_seconds"] == [0.5, 1.5]
        assert len(data["project"]["shots"]) == 2
        assert data["project"]["shots"][0]["prompt"] == plan().shots[0].prompt
        assert data["project"]["status"] == "DRAFT"
        pid = data["project"]["project_id"]
        assert client.get(f"/api/projects/{pid}").status_code == 200
        assert client.post(f"/api/projects/{pid}/generate").status_code == 202


def test_vision_unconfigured_does_not_unlock_edit(tmp_path, monkeypatch):
    monkeypatch.delenv("AVG_VISION_MODEL", raising=False)
    with TestClient(create_app(service(tmp_path))) as client:
        cap = client.get("/api/capabilities").json()
        assert cap["video_understanding"] is False
        assert cap["video_edit_upload"] is False
        result = client.post("/api/video-understand",
            files={"file": ("clip.mp4", b"x", "video/mp4")})
        assert result.status_code == 409
        assert "AVG_VISION_MODEL" in result.text


def test_vision_config_rejects_plain_http_with_credentials():
    with pytest.raises(ValueError, match="HTTPS"):
        VisionConfig("http://example.com", "vision-model", "secret")
    assert VisionConfig("http://localhost:8001/v1", "vision").model == "vision"
    with pytest.raises(ValueError):
        VisionConfig("https://api.example.com", "vision", max_frames=100)


def test_sample_times_not_at_video_ends():
    assert sample_seconds(4, 4) == [0.5, 1.5, 2.5, 3.5]


@pytest.mark.skipif(
    __import__("shutil").which("ffmpeg") is None or
    __import__("shutil").which("ffprobe") is None,
    reason="FFmpeg required for keyframe extraction",
)
def test_model_receives_jpeg_frames_not_mp4(tmp_path, monkeypatch):
    original = tmp_path / "source.mp4"
    asyncio.run(MockVideoBackend(fps=8, width=160, height=90).generate(
        "a boy in sunlight", original, 3.0
    ))
    payloads = []
    client_type = httpx.AsyncClient

    def handler(request):
        body = json.loads(request.content)
        payloads.append(body)
        assert request.headers["authorization"] == "Bearer test-only-key"
        images = [
            c["image_url"]["url"] for c in body["messages"][1]["content"]
            if c["type"] == "image_url"
        ]
        assert len(images) == 3
        assert all(img.startswith("data:image/jpeg;base64,") for img in images)
        assert body["model"] == "vision-test"
        return httpx.Response(200, json={
            "choices": [{"message": {
                "content": plan().model_dump_json()
            }}]
        })

    monkeypatch.setattr(
        "agentic_video.vision.httpx.AsyncClient",
        lambda **kwargs: client_type(
            transport=httpx.MockTransport(handler), **kwargs
        ),
    )
    config = VisionConfig(
        "https://vision.provider.test/v1", "vision-test",
        "test-only-key", max_frames=3
    )
    analyzer = OpenAICompatibleVisionAnalyzer(config)
    from fastapi import UploadFile
    with original.open("rb") as handle:
        result, times = asyncio.run(analyzer.analyze(
            UploadFile(handle, filename="clip.mp4"),
            "分析视频", 2, "cinematic"
        ))
    assert len(result.shots) == 2
    assert len(times) == 3
    assert len(payloads) == 1


def test_provider_invalid_json_is_sanitized(monkeypatch):
    async_client = httpx.AsyncClient

    def handler(request):
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "not json"}}]
        })
    monkeypatch.setattr(
        "agentic_video.vision.httpx.AsyncClient",
        lambda **kwargs: async_client(
            transport=httpx.MockTransport(handler), **kwargs
        ),
    )
    monkeypatch.setattr(
        "agentic_video.vision.inspect_source", lambda path: 2.0
    )
    monkeypatch.setattr(
        "agentic_video.vision.extract_frame",
        lambda path, sec, edge: b"\xff\xd8fake-jpg",
    )
    from fastapi import UploadFile
    from io import BytesIO
    analyzer = OpenAICompatibleVisionAnalyzer(
        VisionConfig("https://api.example.com", "vision", max_frames=1)
    )
    with pytest.raises(VisionError, match="JSON"):
        asyncio.run(analyzer.analyze(
            UploadFile(BytesIO(b"fake-mp4"), filename="input.mp4"),
            "理解场景", 2, "cinematic"
        ))
