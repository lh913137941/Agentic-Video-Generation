"""Safe local provider selection and a mocked fal.ai edit-video contract."""
import asyncio
import json
import os
import stat
import sys
import types
from pathlib import Path

import httpx
import pytest

from agentic_video.v2v_config import (
    FAL_EDIT_MODEL, load_local_v2v, save_local_v2v, ui_provider_status,
)
from agentic_video.fal_video import (
    FalWanEditBackend, _trusted_output_url,
)
from agentic_video.backends import BackendError
from agentic_video.config import Settings


def test_local_v2v_settings_0600_and_secret_not_in_ui(tmp_path):
    path = tmp_path / "v2v.json"
    save_local_v2v("fal", "http://127.0.0.1:8091",
                   FAL_EDIT_MODEL, "private-test-key", path)
    assert json.loads(path.read_text())["api_key"] == "private-test-key"
    if os.name != "nt":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    provider, endpoint, model, key_status = ui_provider_status(path)
    assert provider == "fal"
    assert model == FAL_EDIT_MODEL
    assert key_status == "已设置"
    assert "private" not in key_status
    # Leaving key input empty preserves the previous saved secret.
    save_local_v2v("fal", endpoint, model, "", path)
    assert load_local_v2v(path)["api_key"] == "private-test-key"
    save_local_v2v("mock", endpoint, model, "", path)
    assert load_local_v2v(path)["api_key"] == ""


@pytest.mark.parametrize("provider,url,model,key", [
    ("fal", "http://localhost:8091", "other-model", "key"),
    ("fal", "http://localhost:8091", FAL_EDIT_MODEL, ""),
    ("omni", "http://remote.example:8091", FAL_EDIT_MODEL, ""),
    ("omni", "https://user:password@remote.example", FAL_EDIT_MODEL, ""),
    ("fake", "https://example.com", FAL_EDIT_MODEL, ""),
])
def test_unsafe_provider_settings_rejected(tmp_path, provider, url, model, key):
    with pytest.raises(ValueError):
        save_local_v2v(provider,url,model,key,tmp_path/"v2v.json")


def test_settings_use_ui_file_unless_env_override(tmp_path, monkeypatch):
    import agentic_video.v2v_config as conf
    monkeypatch.setattr(conf, "CONFIG_FILE", tmp_path/"v2v.json")
    monkeypatch.delenv("AVG_BACKEND", raising=False)
    monkeypatch.delenv("FAL_KEY", raising=False)
    monkeypatch.delenv("AVG_FAL_MODEL", raising=False)
    save_local_v2v("fal", "http://localhost:8091",
                   FAL_EDIT_MODEL, "test-key", conf.CONFIG_FILE)
    assert Settings.from_env().backend == "fal"
    assert Settings.from_env().fal_api_key == "test-key"
    monkeypatch.setenv("AVG_BACKEND", "mock")
    assert Settings.from_env().backend == "mock"


def test_fal_backend_rejects_missing_video_and_bad_time(tmp_path):
    backend = FalWanEditBackend("hidden-key")
    with pytest.raises(BackendError, match="原视频"):
        asyncio.run(backend.generate("do something", tmp_path/"out.mp4", 5))
    with pytest.raises(BackendError, match="2～10"):
        asyncio.run(backend.generate(
            "do something", tmp_path/"out.mp4", 30,
            input_reference=tmp_path/"original.mp4",
        ))


@pytest.mark.parametrize("link", [
    "http://v3b.fal.media/video.mp4",
    "https://fal.media.attacker.invalid/out.mp4",
    "https://localhost/internal.mp4",
    "file:///etc/passwd",
])
def test_untrusted_result_urls_rejected(link):
    with pytest.raises(BackendError):
        _trusted_output_url(link)


def test_mocked_fal_upload_queue_and_output(monkeypatch, tmp_path):
    import agentic_video.fal_video as mod
    source = tmp_path/"source.mp4"
    source.write_bytes(b"input")
    calls = []
    class Handle:
        async def get(self):
            calls.append("get")
            return {"video": {"url": "https://v3b.fal.media/files/test.mp4"}}
    class FakeFal:
        def __init__(self, key):
            assert key == "hidden-key"
        async def upload_file(self, filename):
            assert filename == str(source)
            calls.append("upload")
            return "https://storage.fal.media/source.mp4"
        async def submit(self, model, arguments):
            assert model == FAL_EDIT_MODEL
            assert arguments["video_url"] == "https://storage.fal.media/source.mp4"
            assert arguments["prompt"] == "change lighting"
            calls.append("submit")
            return Handle()
    monkeypatch.setitem(sys.modules, "fal_client",
                        types.SimpleNamespace(AsyncClient=FakeFal))
    original_http = httpx.AsyncClient
    def handler(request):
        assert request.url.host == "v3b.fal.media"
        # Minimal MP4 magic; the backend validates the container signature.
        return httpx.Response(200, content=b"\x00\x00\x00\x18ftypisomtest-video")
    monkeypatch.setattr(
        mod.httpx, "AsyncClient",
        lambda **options: original_http(
            transport=httpx.MockTransport(handler), **options
        ),
    )
    result = tmp_path/"result.mp4"
    asyncio.run(FalWanEditBackend("hidden-key").generate(
        "change lighting", result, 5, source,
    ))
    assert result.exists()
    assert result.read_bytes()[4:8] == b"ftyp"
    assert calls == ["upload", "submit", "get"]


def test_no_provider_api_key_in_public_capabilities(tmp_path, monkeypatch):
    import agentic_video.v2v_config as conf
    from agentic_video.api import create_app, build_service
    from fastapi.testclient import TestClient
    monkeypatch.setattr(conf, "CONFIG_FILE", tmp_path/"v2v.json")
    for key in ["AVG_BACKEND", "FAL_KEY", "AVG_FAL_MODEL"]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("AVG_DATA_DIR", str(tmp_path/"data"))
    save_local_v2v("fal", "http://127.0.0.1:8091",
                   FAL_EDIT_MODEL, "secret-private-key", conf.CONFIG_FILE)
    with TestClient(create_app(build_service())) as client:
        response = client.get("/api/capabilities")
        assert response.status_code == 200
        assert "secret-private-key" not in response.text
        data = response.json()
        assert data["video_edit_backend"] == "fal-wan-edit"
        assert data["video_edit_model"] == FAL_EDIT_MODEL
        assert data["video_edit_max_seconds"] == 10
