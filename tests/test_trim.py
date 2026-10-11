"""Separate FFmpeg trim flow: no dependence on Gradio's built-in scissors UI."""
import shutil
from pathlib import Path

import pytest

from agentic_video.trim import inspect_source, trim_source_video


@pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
                    reason="FFmpeg/ffprobe missing")
def test_trim_mp4_uses_explicit_range_and_preserves_original(tmp_path):
    import asyncio
    from agentic_video.backends import MockVideoBackend

    original = tmp_path / "original.mp4"
    asyncio.run(MockVideoBackend(
        fps=16, width=160, height=90
    ).generate("scene", original, 3.0))
    size_before = original.stat().st_size
    assert 2.8 < inspect_source(original) < 3.2
    clip = Path(trim_source_video(original, 0.5, 1.75))
    assert clip.is_file()
    assert 1.10 < inspect_source(clip) < 1.40
    assert original.stat().st_size == size_before
    assert clip != original
    assert clip.stat().st_size > 500

    with pytest.raises(ValueError, match="至少"):
        trim_source_video(original, 1.2, 1.25)
    with pytest.raises(ValueError, match="裁剪时间"):
        trim_source_video(original, -1, 1)
    with pytest.raises(ValueError, match="裁剪时间"):
        trim_source_video(original, 1, 40)


def test_invalid_video_and_path(tmp_path):
    with pytest.raises(ValueError, match="MP4"):
        inspect_source(tmp_path / "nonexistent.mp4")
    path = tmp_path / "x.mp4"
    path.write_bytes(b"broken")
    with pytest.raises(ValueError):
        inspect_source(path)


def test_missing_ffmpeg_is_actionable(monkeypatch, tmp_path):
    from agentic_video import trim
    path = tmp_path / "input.mp4"
    path.write_bytes(b"fake file")
    monkeypatch.setattr(trim.shutil, "which", lambda tool: None)
    with pytest.raises(trim.MediaToolsUnavailable) as exc:
        trim.inspect_source(path)
    text = str(exc.value)
    assert "ffmpeg/ffprobe" in text
    assert "winget install" in text
    assert "重启 Gradio" in text


def test_ffprobe_missing_only_is_reported(monkeypatch):
    from agentic_video import trim
    monkeypatch.setattr(
        trim.shutil, "which",
        lambda tool: "C:/ffmpeg/bin/ffmpeg.exe" if tool == "ffmpeg" else None
    )
    with pytest.raises(trim.MediaToolsUnavailable, match="ffprobe"):
        trim.check_media_tools()
